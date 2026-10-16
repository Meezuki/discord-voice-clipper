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
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

        if model_size and WhisperModel:
            logger.info(f"Loading faster-whisper model ({model_size}, {device}, {compute_type})...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data: np.ndarray, language="en"):
        """
        Transcribes a 16 kHz float32 mono audio array.
        Returns tuple: (transcription_text, latency_seconds).
        """
        if audio_data is None or len(audio_data) == 0:
            return "", 0.0

        if not self.model:
            return "(ASR model not loaded)", 0.0

        start_time = time.perf_counter()
        
        # faster-whisper accepts float32 numpy array normalized to [-1.0, 1.0]
        segments, info = self.model.transcribe(
            audio_data,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Concatenate all transcription segments
        text = " ".join([segment.text.strip() for segment in segments]).strip()
        latency = time.perf_counter() - start_time
        
        logger.info(f"Transcription finished in {latency:.3f}s: '{text}'")
        return text, latency