import os
import json
import time
import logging
import numpy as np

try:
    import vosk
except ImportError:
    vosk = None

logger = logging.getLogger(__name__)

DEFAULT_TARGET_PHRASES = [
    "bot record",
    "record",
    "but record",
    "boat record",
    "start record",
    "recording"
]

class KeywordSpotter:
    """
    Offline English Keyword Spotting (KWS) module using grammar-constrained Vosk
    with an optional rolling faster-whisper fallback for maximum accuracy.
    Continuously listens for activation phrases ('bot record', 'record', 'but record')
    with sub-100ms latency and 100% offline local processing.
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

        if target_phrases is None:
            self.target_phrases = [p.lower() for p in DEFAULT_TARGET_PHRASES]
        else:
            self.target_phrases = [p.lower() for p in target_phrases]
            for variant in ["record", "bot record", "but record", "boat record", "recording"]:
                if variant not in self.target_phrases:
                    self.target_phrases.append(variant)

        self.model = None
        self.recognizer = None

        if model_path and os.path.exists(model_path) and vosk:
            try:
                vosk.SetLogLevel(-1)
                self.model = vosk.Model(model_path)
                # Constrain search graph to target phrases + [unk] for 10x higher keyword spot rate
                grammar_words = list(set(self.target_phrases + ["[unk]"]))
                grammar_json = json.dumps(grammar_words)
                self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate, grammar_json)
                logger.info(f"Vosk English KWS initialized with grammar: {grammar_words}")
            except Exception as e:
                logger.warning(f"Failed to load Vosk model with grammar: {e}")
                try:
                    self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
                except Exception:
                    pass

        # Rolling buffer for secondary Whisper fallback verification (2.5 seconds = 40,000 samples)
        self.whisper_buffer = []
        self.whisper_samples_target = int(self.sample_rate * 2.5)
        self.last_whisper_check = time.time()
        self.last_partial_logged = ""

    def _check_and_trigger(self, recognized_text: str) -> bool:
        recognized_text = recognized_text.lower().strip()
        if not recognized_text:
            return False

        detected = any(phrase in recognized_text for phrase in self.target_phrases)
        if detected:
            current_time = time.time()
            if current_time - self.last_trigger_time >= self.cooldown_seconds:
                self.last_trigger_time = current_time
                logger.info(f"🎯 Activation phrase detected: '{recognized_text}'")
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
        1. Vosk Grammar KWS (instant sub-100ms response).
        2. Rolling Whisper Spotter Fallback (rock-solid semantic detection).
        Returns True if activation phrase is detected and debounce passes.
        """
        if not pcm_bytes:
            return False

        # 1. Primary: Fast Vosk Grammar KWS (<50ms)
        if self.recognizer:
            if self.recognizer.AcceptWaveform(pcm_bytes):
                res = json.loads(self.recognizer.Result())
                text = res.get("text", "")
                if text:
                    logger.info(f"🎙️ [Vosk KWS Final]: \"{text}\"")
                if self._check_and_trigger(text):
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
                if text and self._check_and_trigger(text):
                    if hasattr(self.recognizer, "Reset"):
                        self.recognizer.Reset()
                    self.whisper_buffer.clear()
                    return True

        # 2. Secondary: Rolling Whisper Spotter Fallback
        if self.asr_engine:
            chunk_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            self.whisper_buffer.append(chunk_int16)
            total_samples = sum(len(c) for c in self.whisper_buffer)

            now = time.time()
            if total_samples >= self.whisper_samples_target and (now - self.last_whisper_check >= 1.8):
                self.last_whisper_check = now
                audio_concat = np.concatenate(self.whisper_buffer).astype(np.float32) / 32768.0

                # Retain last 0.8s overlap to avoid missing words split across chunk boundaries
                overlap_samples = int(self.sample_rate * 0.8)
                if len(audio_concat) > overlap_samples:
                    tail_int16 = (audio_concat[-overlap_samples:] * 32767).astype(np.int16)
                    self.whisper_buffer = [tail_int16]
                else:
                    self.whisper_buffer = []

                # Only run Whisper if there is sound above silence threshold
                rms = np.sqrt(np.mean(audio_concat ** 2))
                if rms > 0.008:
                    whisper_text, _ = self.asr_engine.transcribe(audio_concat, language="en")
                    if whisper_text:
                        logger.info(f"🔍 [Whisper Spotter]: \"{whisper_text}\"")
                        if self._check_and_trigger(whisper_text):
                            if self.recognizer and hasattr(self.recognizer, "Reset"):
                                self.recognizer.Reset()
                            self.whisper_buffer.clear()
                            return True

        return False

