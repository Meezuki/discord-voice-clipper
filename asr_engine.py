import time
import logging
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)

class LocalASREngine:
    """
    Local Speech Recognition engine powered by faster-whisper (CTranslate2).
    100% offline with zero external API dependencies, optimized with int8
    quantization on CPU / float16 on CUDA GPU.
    """
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        # Model is multilingual (e.g. 'base' or 'small') supporting English ('en') and Indonesian ('id')
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

        if model_size and WhisperModel:
            logger.info(f"Loading faster-whisper multilingual model ({model_size}, {device}, {compute_type})...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data: np.ndarray, language=None):
        """
        Transcribes a 16 kHz float32 mono audio array.
        Language: None (auto-detects English vs Indonesian), or 'en', 'id'.
        Returns tuple: (transcription_text, latency_seconds).
        """
        if audio_data is None or len(audio_data) == 0:
            return "", 0.0

        if not self.model:
            return "(ASR model not loaded)", 0.0

        start_time = time.perf_counter()

        target_lang = language
        if target_lang is None and hasattr(self.model, "detect_language"):
            # Constrain language detection to supported bilingual domain (Indonesian vs English)
            # Prevents erroneous auto-detection into ungrounded languages (e.g. 'ms', 'nn', 'cy')
            try:
                detect_res = self.model.detect_language(audio_data, vad_filter=True)
                if isinstance(detect_res, (list, tuple)) and len(detect_res) >= 3:
                    prob_dict = dict(detect_res[2])
                    id_prob = prob_dict.get("id", 0.0) + prob_dict.get("ms", 0.0) * 0.5
                    en_prob = prob_dict.get("en", 0.0)
                    target_lang = "id" if id_prob > 0.5 and id_prob > en_prob else "en"
                else:
                    target_lang = "en"
            except Exception as e:
                logger.warning(f"Bilingual language detection fallback to 'en': {e}")
                target_lang = "en"
        
        # faster-whisper accepts float32 numpy array normalized to [-1.0, 1.0]
        segments, info = self.model.transcribe(
            audio_data,
            language=target_lang,
            beam_size=5,
            initial_prompt="Meeting discussion notes and conversation.",
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=250,
                min_silence_duration_ms=400,
                speech_pad_ms=300
            ),
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4],
            no_repeat_ngram_size=0,
            repetition_penalty=1.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=1.5
        )
        
        # Concatenate all transcription segments
        text = " ".join([segment.text.strip() for segment in segments]).strip()
        latency = time.perf_counter() - start_time
        
        detected_lang = getattr(info, 'language', target_lang or 'unknown')
        logger.info(f"Transcription ({detected_lang}) finished in {latency:.3f}s: '{text}'")
        return text, latency