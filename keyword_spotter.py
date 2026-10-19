import os
import re
import json
import time
import logging
import numpy as np

try:
    import vosk
except ImportError:
    vosk = None

logger = logging.getLogger(__name__)

def normalize_spoken_text(text: str) -> str:
    """Strips punctuation and normalizes whitespace for robust phonetic and command matching."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def is_intentional_clip_command(text: str) -> bool:
    """
    Evaluates whether transcribed text contains an intentional clip command (e.g. 'bot clip that', 'clip that').
    Handles punctuation (e.g. 'Bot, clip that.') and rejects casual phrases ('paper clip', 'hair clip').
    """
    norm = normalize_spoken_text(text)
    if not norm:
        return False

    # Block common conversational phrases using 'clip' as a noun/non-command
    non_command_phrases = [
        "paper clip",
        "hair clip",
        "binder clip",
        "nail clip",
        "clip on",
        "magazine clip",
        "money clip"
    ]
    if any(p in norm for p in non_command_phrases):
        return False

    # Positive command patterns:
    # 'bot clip that', 'bot, clip that', 'hey bot clip that', 'bot please clip that', 'clip that', 'clip this'
    explicit_patterns = [
        r"\bbot\b.*?\bclip\b.*?\b(that|this|it)\b",       # 'bot clip that', 'bot, please clip that'
        r"\b(please |can you )?clip\b.*?\b(that|this|it)\b", # 'clip that', 'please clip that', 'can you clip that'
        r"\bclip that\b",                                # 'clip that'
        r"\bclip this\b",                                # 'clip this'
        r"^clip$"                                        # exact single-word command
    ]
    for pattern in explicit_patterns:
        if re.search(pattern, norm):
            return True

    return False

# Stage 1 Vosk candidate regex: catches potential speech commands for Whisper verification
VOSK_CANDIDATE_REGEX = re.compile(
    r"\b(bot|but|boat|bought|bad|pot)\b.*?\bclip|"
    r"\bclip\b.*?\b(that|this|it)|"
    r"\bclip that\b|\bclip this\b|"
    r"^clip$",
    re.IGNORECASE
)

class KeywordSpotter:
    """
    Offline English Keyword Spotting (KWS) module using open-vocabulary Vosk
    backed by a Two-Stage verification layer via faster-whisper.
    Continuously listens for intentional activation phrases ('bot clip that')
    while rejecting false alarms from conversational speech.
    """
    def __init__(
        self,
        model_path=None,
        target_phrases=None,
        sample_rate=16000,
        cooldown_seconds=8.0,
        trigger_callback=None,
        asr_engine=None
    ):
        self.sample_rate = sample_rate
        self.cooldown_seconds = cooldown_seconds
        self.trigger_callback = trigger_callback
        self.asr_engine = asr_engine
        self.last_trigger_time = 0.0

        self.model = None
        self.recognizer = None

        if model_path and os.path.exists(model_path) and vosk:
            try:
                vosk.SetLogLevel(-1)
                self.model = vosk.Model(model_path)
                # Open-vocabulary KaldiRecognizer allows natural recognition of full English sentences
                self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
                logger.info("Vosk English KWS initialized with open vocabulary")
            except Exception as e:
                logger.warning(f"Failed to load Vosk model: {e}")

        # Rolling buffer for Two-Stage Whisper Verification and Fallback Spotting
        self.whisper_buffer = []
        self.whisper_samples_target = int(self.sample_rate * 2.5)
        self.last_whisper_check = time.time()
        self.last_partial_logged = ""

    def _check_candidate(self, text: str) -> bool:
        """Stage 1: Checks if text might be a candidate clip command."""
        norm = normalize_spoken_text(text)
        if not norm:
            return False
        # Block known non-commands immediately
        if any(p in norm for p in ["paper clip", "hair clip", "binder clip", "nail clip", "clip on", "magazine clip", "money clip"]):
            return False
        if is_intentional_clip_command(norm):
            return True
        return bool(VOSK_CANDIDATE_REGEX.search(norm))

    def _verify_with_whisper(self, candidate_text: str) -> tuple[bool, str]:
        """
        Stage 2 Verification:
        When Vosk suspects a wake word, verify against Whisper on the recent audio buffer.
        Suppresses false alarms (e.g. Vosk hearing noise when user said something unrelated).
        Returns tuple: (is_confirmed, final_transcription).
        """
        if not self.asr_engine or len(self.whisper_buffer) == 0:
            return True, candidate_text  # Standalone mode or tests without ASR engine

        try:
            recent_pcm = np.concatenate(self.whisper_buffer)
            audio_float = recent_pcm.astype(np.float32) / 32768.0

            # Ignore absolute silence
            rms = np.sqrt(np.mean(audio_float ** 2))
            if rms < 0.002:
                return False, ""

            whisper_text, _ = self.asr_engine.transcribe(audio_float, language="en")
            confirmed = is_intentional_clip_command(whisper_text)

            if confirmed:
                logger.info(f"✅ Two-Stage KWS Verified: Vosk candidate '{candidate_text}' confirmed by Whisper ('{whisper_text}')")
                return True, whisper_text
            else:
                logger.warning(
                    f"🛡️ Two-Stage KWS Suppressed False Alarm: Vosk heard '{candidate_text}', "
                    f"but Whisper verified '{whisper_text}'."
                )
                return False, whisper_text
        except Exception as e:
            logger.error(f"Error during Whisper verification: {e}")
            return True, candidate_text

    def _check_and_trigger(self, recognized_text: str) -> bool:
        recognized_text = recognized_text.strip()
        if not recognized_text:
            return False

        current_time = time.time()
        if current_time - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = current_time
            logger.info(f"🎯 Activation phrase triggered: '{recognized_text}'")
            if self.trigger_callback:
                try:
                    self.trigger_callback(recognized_text)
                except Exception as e:
                    logger.error(f"Error in trigger_callback: {e}")
            return True
        else:
            logger.debug("Activation phrase ignored due to debounce cooldown.")
        return False

    def process_chunk(self, pcm_bytes: bytes) -> bool:
        """
        Receives 16 kHz 16-bit mono PCM bytes and feeds to:
        1. Fast Vosk Open-Vocabulary KWS with Two-Stage Whisper Verification.
        2. Rolling Whisper Spotter Fallback.
        Returns True if activation phrase is confirmed and debounce passes.
        """
        if not pcm_bytes:
            return False

        # Continuously maintain recent rolling audio for Whisper verification
        chunk_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        self.whisper_buffer.append(chunk_int16)
        max_chunks = int(self.sample_rate * 3.0 / len(chunk_int16)) if len(chunk_int16) > 0 else 30
        if len(self.whisper_buffer) > max_chunks:
            self.whisper_buffer.pop(0)

        # 1. Primary: Fast Vosk Open-Vocabulary KWS with Whisper Verification
        if self.recognizer:
            if self.recognizer.AcceptWaveform(pcm_bytes):
                res = json.loads(self.recognizer.Result())
                text = res.get("text", "")
                if text:
                    logger.info(f"🎙️ [Vosk KWS Final]: \"{text}\"")
                if self._check_candidate(text):
                    verified, confirmed_text = self._verify_with_whisper(text)
                    if verified and self._check_and_trigger(confirmed_text or text):
                        if hasattr(self.recognizer, "Reset"):
                            self.recognizer.Reset()
                        self.whisper_buffer.clear()
                        return True
            else:
                partial = json.loads(self.recognizer.PartialResult())
                text = partial.get("partial", "").strip()
                if text and text != "[unk]" and text != self.last_partial_logged:
                    self.last_partial_logged = text
                    logger.info(f"🎙️ [Vosk KWS Hearing]: \"{text}\"")
                if self._check_candidate(text):
                    verified, confirmed_text = self._verify_with_whisper(text)
                    if verified and self._check_and_trigger(confirmed_text or text):
                        if hasattr(self.recognizer, "Reset"):
                            self.recognizer.Reset()
                        self.whisper_buffer.clear()
                        return True

        # 2. Secondary: Periodic Rolling Whisper Spotter Fallback (runs every 1.8s)
        if self.asr_engine:
            total_samples = sum(len(c) for c in self.whisper_buffer)
            now = time.time()
            if total_samples >= self.whisper_samples_target and (now - self.last_whisper_check >= 1.8):
                self.last_whisper_check = now
                audio_concat = np.concatenate(self.whisper_buffer).astype(np.float32) / 32768.0

                rms = np.sqrt(np.mean(audio_concat ** 2))
                if rms > 0.002:
                    whisper_text, _ = self.asr_engine.transcribe(audio_concat, language="en")
                    if whisper_text:
                        logger.info(f"🔍 [Whisper Spotter]: \"{whisper_text}\"")
                        if is_intentional_clip_command(whisper_text) and self._check_and_trigger(whisper_text):
                            if self.recognizer and hasattr(self.recognizer, "Reset"):
                                self.recognizer.Reset()
                            self.whisper_buffer.clear()
                            return True

        return False

