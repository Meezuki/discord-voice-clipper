import os
import wave
import collections
import numpy as np
import onnxruntime as ort
from scipy import signal

SAMPLE_RATE = 16000
MEL_STEP_SAMPLES = 1280       # 80 ms per step (8 mel frames)
MEL_CONTEXT_SAMPLES = 480     # 30 ms overlap matching offline melspectrogram.onnx
MEL_WINDOW_FRAMES = 76        # 76 mel frames -> 1 embedding vector
MEL_BINS = 32
EMBEDDING_FRAMES_PER_STEP = 8 # 8 mel frames per 80 ms step
SEQUENCE_LENGTH = 16          # 16 embedding steps (~2.0 s receptive field)
EMBEDDING_DIM = 96
WINDOW_SAMPLES = 32000        # 2.0 s @ 16 kHz -> 197 mel frames -> 16 embeddings


def load_wav_int16_scale(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Loads a WAV file as float32 mono scaled to int16 range [-32768.0, 32767.0],
    which is required by openWakeWord's melspectrogram.onnx backbone.
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV (sampwidth=2), got {sampwidth} in {file_path}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if framerate != target_sr and len(audio) > 0:
        num_target = int(round(len(audio) * float(target_sr) / framerate))
        audio = signal.resample(audio, num_target).astype(np.float32)

    return audio


def to_int16_scale(audio: np.ndarray) -> np.ndarray:
    """Normalizes int16 or [-1.0, 1.0] float32 waveform to float32 in int16 scale."""
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return arr
    if arr.dtype == np.int16 or np.max(np.abs(arr)) > 2.0:
        return arr.astype(np.float32)
    return (arr * 32767.0).astype(np.float32)


def extract_speech_centered_window(
    audio: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    jitter_samples: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Extracts a fixed-length window centered around the peak speech energy region
    so wake-word utterances inside 2.5-3.5s clips are preserved cleanly.
    """
    audio = to_int16_scale(audio)
    n = len(audio)
    if n == window_samples:
        return audio
    if n < window_samples:
        pad_total = window_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant")

    # Find center of energy using a 1.0s sliding RMS window
    energy_win = min(SAMPLE_RATE, n)
    sq = audio ** 2
    kernel = np.ones(energy_win, dtype=np.float32)
    energy = np.convolve(sq, kernel, mode="valid")
    peak_start = int(np.argmax(energy))
    peak_center = peak_start + energy_win // 2

    if jitter_samples > 0 and rng is not None:
        peak_center += int(rng.integers(-jitter_samples, jitter_samples + 1))

    start = max(0, min(n - window_samples, peak_center - window_samples // 2))
    return audio[start : start + window_samples]


class WakeWordFeatureExtractor:
    """
    Shared ONNX feature extractor converting 16 kHz audio into 96-dim speech
    embeddings using frozen melspectrogram.onnx + embedding_model.onnx.
    Used identically during training, evaluation, and live streaming inference.
    """

    def __init__(self, mel_model_path: str, emb_model_path: str):
        if not os.path.exists(mel_model_path):
            raise FileNotFoundError(f"Mel-spectrogram ONNX model not found: {mel_model_path}")
        if not os.path.exists(emb_model_path):
            raise FileNotFoundError(f"Embedding ONNX model not found: {emb_model_path}")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        self.mel_session = ort.InferenceSession(
            mel_model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.emb_session = ort.InferenceSession(
            emb_model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.mel_input_name = self.mel_session.get_inputs()[0].name
        self.emb_input_name = self.emb_session.get_inputs()[0].name

    def waveform_to_mel(self, audio_int16_scale: np.ndarray) -> np.ndarray:
        """Computes normalized log-mel frames of shape (T_mel, 32)."""
        wav = to_int16_scale(audio_int16_scale)
        if len(wav) < 512:
            wav = np.pad(wav, (0, 512 - len(wav)))
        raw_mel = self.mel_session.run(None, {self.mel_input_name: wav[None, :]})[0]
        mel = (raw_mel / 10.0 + 2.0).squeeze(axis=0).squeeze(axis=0)
        return mel.astype(np.float32)

    def mel_to_embeddings(self, mel: np.ndarray) -> np.ndarray:
        """Slides a 76-frame window with stride 8 across mel (T_mel, 32) -> (T_emb, 96)."""
        if len(mel) < MEL_WINDOW_FRAMES:
            pad = MEL_WINDOW_FRAMES - len(mel)
            mel = np.pad(mel, ((pad, 0), (0, 0)), mode="constant")

        windows = np.stack(
            [
                mel[i : i + MEL_WINDOW_FRAMES]
                for i in range(0, len(mel) - MEL_WINDOW_FRAMES + 1, EMBEDDING_FRAMES_PER_STEP)
            ],
            axis=0,
        )[..., None].astype(np.float32)

        embs = self.emb_session.run(None, {self.emb_input_name: windows})[0]
        return embs.reshape(len(windows), EMBEDDING_DIM).astype(np.float32)

    def extract_sequence(
        self, audio: np.ndarray, sequence_length: int = SEQUENCE_LENGTH
    ) -> np.ndarray:
        """
        Extracts a fixed (sequence_length, 96) embedding tensor from a short clip.
        Pads or crops symmetrically if needed.
        """
        wav = extract_speech_centered_window(audio, window_samples=WINDOW_SAMPLES)
        mel = self.waveform_to_mel(wav)
        embs = self.mel_to_embeddings(mel)
        if len(embs) < sequence_length:
            pad = sequence_length - len(embs)
            embs = np.pad(embs, ((pad, 0), (0, 0)), mode="constant")
        elif len(embs) > sequence_length:
            embs = embs[-sequence_length:]
        return embs.astype(np.float32)

    def extract_all_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """Extracts sequential 80ms-hop embeddings (N, 96) for continuous long audio."""
        mel = self.waveform_to_mel(audio)
        return self.mel_to_embeddings(mel)


class StreamingFeatureBuffer:
    """
    Incremental streaming buffer converting incoming 16 kHz PCM samples into
    rolling (16, 96) embedding windows with exact equivalence to offline mel extraction.
    """

    def __init__(self, extractor: WakeWordFeatureExtractor, sequence_length: int = SEQUENCE_LENGTH):
        self.extractor = extractor
        self.sequence_length = sequence_length
        self.pcm_remainder = np.zeros(0, dtype=np.float32)
        self.mel_context = np.zeros(MEL_CONTEXT_SAMPLES, dtype=np.float32)
        self.mel_buffer = collections.deque(maxlen=MEL_WINDOW_FRAMES)
        self.emb_buffer = collections.deque(maxlen=sequence_length)

    def reset(self):
        self.pcm_remainder = np.zeros(0, dtype=np.float32)
        self.mel_context = np.zeros(MEL_CONTEXT_SAMPLES, dtype=np.float32)
        self.mel_buffer.clear()
        self.emb_buffer.clear()

    def push_audio(self, samples_int16_scale: np.ndarray) -> list[np.ndarray]:
        """
        Appends new 16 kHz samples and returns a list of (sequence_length, 96)
        embedding windows whenever enough frames are available.
        """
        wav = to_int16_scale(samples_int16_scale)
        if len(wav) == 0:
            return []

        self.pcm_remainder = np.concatenate([self.pcm_remainder, wav])
        ready_windows = []

        while len(self.pcm_remainder) >= MEL_STEP_SAMPLES:
            step = self.pcm_remainder[:MEL_STEP_SAMPLES]
            self.pcm_remainder = self.pcm_remainder[MEL_STEP_SAMPLES:]

            mel_input = np.concatenate([self.mel_context, step])[None, :]
            self.mel_context = step[-MEL_CONTEXT_SAMPLES:]

            raw_mel = self.extractor.mel_session.run(
                None, {self.extractor.mel_input_name: mel_input}
            )[0]
            mel_frames = (raw_mel / 10.0 + 2.0).squeeze(axis=0).squeeze(axis=0)
            for frame in mel_frames[-EMBEDDING_FRAMES_PER_STEP:]:
                self.mel_buffer.append(frame)

            if len(self.mel_buffer) == MEL_WINDOW_FRAMES:
                mel_win = np.stack(self.mel_buffer, axis=0)[None, ..., None].astype(np.float32)
                emb = self.extractor.emb_session.run(
                    None, {self.extractor.emb_input_name: mel_win}
                )[0].reshape(EMBEDDING_DIM)
                self.emb_buffer.append(emb.astype(np.float32))

                if len(self.emb_buffer) == self.sequence_length:
                    ready_windows.append(np.stack(self.emb_buffer, axis=0))

        return ready_windows
