import os
import json
import time
import logging
import numpy as np
import onnxruntime as ort

from download_model import (
    MEL_MODEL_PATH,
    EMB_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_CONFIG_PATH,
)
from wakeword_features import WakeWordFeatureExtractor, StreamingFeatureBuffer

logger = logging.getLogger(__name__)


class KeywordSpotter:
    """
    Streaming local Wake-Word / Keyword Spotting (KWS) engine for 'bot clip that'.
    Runs a trained PyTorch-to-ONNX BiGRU classifier head on top of frozen ONNX
    speech embeddings (melspectrogram.onnx + embedding_model.onnx) with <20 ms CPU latency.
    """

    def __init__(
        self,
        model_path: str | None = CLASSIFIER_MODEL_PATH,
        mel_model_path: str = MEL_MODEL_PATH,
        emb_model_path: str = EMB_MODEL_PATH,
        config_path: str | None = None,
        target_phrases=None,
        sample_rate: int = 16000,
        cooldown_seconds: float = 8.0,
        threshold: float | None = None,
        patience_frames: int | None = None,
        trigger_callback=None,
        asr_engine=None,  # Kept for signature compatibility; wake-word detection is standalone
    ):
        self.sample_rate = sample_rate
        self.cooldown_seconds = cooldown_seconds
        self.trigger_callback = trigger_callback
        self.last_trigger_time = 0.0
        self.last_probability = 0.0
        self.consecutive_hits = 0

        # Resolve model config (threshold & patience) if available
        resolved_cfg_path = config_path
        if resolved_cfg_path is None and model_path:
            candidate = os.path.join(os.path.dirname(model_path), "model_config.json")
            resolved_cfg_path = candidate if os.path.exists(candidate) else CLASSIFIER_CONFIG_PATH

        cfg = {}
        if resolved_cfg_path and os.path.exists(resolved_cfg_path):
            try:
                with open(resolved_cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read wake-word config {resolved_cfg_path}: {e}")

        self.threshold = float(threshold if threshold is not None else cfg.get("threshold", 0.5))
        self.patience_frames = int(
            patience_frames if patience_frames is not None else cfg.get("patience_frames", 2)
        )

        self.extractor = None
        self.feature_buffer = None
        self.classifier_session = None
        self.classifier_input_name = None

        if model_path and os.path.exists(model_path) and os.path.exists(mel_model_path) and os.path.exists(emb_model_path):
            try:
                self.extractor = WakeWordFeatureExtractor(mel_model_path, emb_model_path)
                self.feature_buffer = StreamingFeatureBuffer(self.extractor)

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 2
                self.classifier_session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                self.classifier_input_name = self.classifier_session.get_inputs()[0].name
                logger.info(
                    f"Trained ONNX Wake-Word Spotter loaded ({os.path.basename(model_path)}, "
                    f"threshold={self.threshold:.3f}, patience={self.patience_frames})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ONNX wake-word spotter: {e}")
                self.classifier_session = None
        elif model_path:
            logger.warning(
                f"Trained wake-word model not found at '{model_path}'. "
                "Voice trigger disabled until you run train_wakeword.py (manual !clip still works)."
            )

    @property
    def is_ready(self) -> bool:
        return self.classifier_session is not None and self.feature_buffer is not None

    def predict_window(self, embedding_window: np.ndarray) -> float:
        """Runs ONNX classifier on a single (16, 96) embedding window -> probability [0, 1]."""
        if self.classifier_session is None:
            return 0.0
        inp = np.asarray(embedding_window, dtype=np.float32)
        if inp.ndim == 2:
            inp = inp[None, ...]
        out = self.classifier_session.run(None, {self.classifier_input_name: inp})[0]
        return float(out.reshape(-1)[0])

    def _check_and_trigger(self, confidence: float) -> bool:
        now = time.time()
        if now - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = now
            label = f"bot clip that (conf={confidence:.2f})"
            logger.info(f"🎯 Wake-word triggered: {label}")
            if self.trigger_callback:
                try:
                    self.trigger_callback(label)
                except Exception as e:
                    logger.error(f"Error in trigger_callback: {e}")
            return True
        logger.debug("Wake-word ignored due to debounce cooldown.")
        return False

    def process_chunk(self, pcm_bytes: bytes) -> bool:
        """
        Ingests 16 kHz 16-bit mono PCM bytes, extracts streaming speech embeddings
        every 80 ms (1280 samples), and runs the trained ONNX wake-word classifier.
        Returns True if 'bot clip that' is detected and debounce cooldown passes.
        """
        if not pcm_bytes or not self.is_ready:
            return False

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return False

        windows = self.feature_buffer.push_audio(samples)
        for win in windows:
            prob = self.predict_window(win)
            self.last_probability = prob

            if prob >= self.threshold:
                self.consecutive_hits += 1
            else:
                self.consecutive_hits = 0

            if self.consecutive_hits >= self.patience_frames:
                self.consecutive_hits = 0
                if self._check_and_trigger(prob):
                    self.feature_buffer.reset()
                    return True

        return False
