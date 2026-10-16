import io
import time
import wave
import threading
import collections
import logging
import numpy as np
from scipy import signal
from discord.ext import voice_recv

logger = logging.getLogger("AudioSink")

class SpeechSegmenter:
    """
    Per-speaker Voice Activity Detector and sentence segmenter.
    Buffers incoming 20ms frames per user, detects speech vs silence,
    and flushes completed sentences when a pause (>= silence_timeout_sec) occurs.
    """
    def __init__(
        self,
        segment_callback=None,
        speech_threshold=0.012,
        silence_timeout_sec=0.7,
        min_duration_sec=0.8,
        max_duration_sec=4.5
    ):
        self.segment_callback = segment_callback
        self.speech_threshold = speech_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

        self.silence_frames_threshold = int(silence_timeout_sec / 0.02)
        self.max_frames = int(max_duration_sec / 0.02)
        self.min_frames = int(min_duration_sec / 0.02)

        self.lock = threading.Lock()
        self.speakers = {}

    def add_frame(self, user_id, speaker_name: str, pcm_16k_float32: np.ndarray):
        if len(pcm_16k_float32) == 0:
            return

        rms = np.sqrt(np.mean(pcm_16k_float32 ** 2))
        is_speech = rms >= self.speech_threshold
        to_flush = None

        with self.lock:
            if user_id not in self.speakers:
                self.speakers[user_id] = {
                    "frames": [],
                    "silence_count": 0,
                    "name": speaker_name or f"User_{user_id}"
                }

            sp = self.speakers[user_id]
            if speaker_name:
                sp["name"] = speaker_name

            if is_speech:
                sp["frames"].append(pcm_16k_float32)
                sp["silence_count"] = 0
                if len(sp["frames"]) >= self.max_frames:
                    concat_audio = np.concatenate(sp["frames"])
                    to_flush = (user_id, sp["name"], concat_audio)
                    sp["frames"] = []
                    sp["silence_count"] = 0
            else:
                if sp["frames"]:
                    sp["silence_count"] += 1
                    # Retain subtle silence padding (up to 5 frames = 100ms) to avoid clipping word ends
                    if sp["silence_count"] <= 5:
                        sp["frames"].append(pcm_16k_float32)

                    if sp["silence_count"] >= self.silence_frames_threshold:
                        if len(sp["frames"]) >= self.min_frames:
                            concat_audio = np.concatenate(sp["frames"])
                            to_flush = (user_id, sp["name"], concat_audio)
                        sp["frames"] = []
                        sp["silence_count"] = 0

        if to_flush and self.segment_callback:
            u_id, name, audio = to_flush
            try:
                self.segment_callback(u_id, name, audio)
            except Exception as e:
                logger.error(f"Error in speech segment callback: {e}")

    def cleanup(self):
        with self.lock:
            self.speakers.clear()

class SynchronizedRollingSink(voice_recv.AudioSink):
    """
    AudioSink that ingests Discord RTP audio packets (48 kHz stereo 16-bit PCM),
    performs exact 3:1 integer decimation down to 16 kHz mono, mixes multi-speaker
    audio with non-linear soft-clipping (np.tanh) to prevent integer overflow distortion,
    and supplies a continuous streaming chunk feed (100-200ms) to the Vosk KWS engine.
    """
    def __init__(self, buffer_seconds=30, chunk_callback=None, chunk_ms=100, segmenter=None):
        super().__init__()
        self.buffer_seconds = buffer_seconds
        self.chunk_callback = chunk_callback
        self.chunk_ms = chunk_ms
        self.segmenter = segmenter
        self.source_sr = 48000
        self.target_sr = 16000
        self.decimation_factor = 3  # 48000 / 16000

        # 1 Discord packet = 20 ms = 960 stereo samples (48 kHz) -> 320 mono samples (16 kHz)
        self.samples_per_packet = int(self.target_sr * 0.02)  # 320
        self.max_packets = int((buffer_seconds * 1000) / 20)  # 1500 packets for 30 seconds
        
        self.lock = threading.Lock()
        # Deque containing normalized float32 chunks of 20 ms duration (320 samples)
        self.rolling_buffer = collections.deque(maxlen=self.max_packets)

        # Per-user queue for timeline synchronization
        self.user_buffers = collections.defaultdict(lambda: collections.deque(maxlen=50))

        # Accumulator for Vosk streaming chunks
        self.chunk_samples_needed = int(self.target_sr * (chunk_ms / 1000.0))
        self.chunk_accumulator = []
        self.packet_count = 0

    def wants_opus(self) -> bool:
        return False  # Discord decodes to 16-bit PCM

    def cleanup(self):
        with self.lock:
            self.rolling_buffer.clear()
            self.user_buffers.clear()
            self.chunk_accumulator.clear()
            if self.segmenter:
                self.segmenter.cleanup()


    def _decimate_and_mono(self, pcm_bytes: bytes) -> np.ndarray:
        """Converts 48kHz stereo int16 PCM to 16kHz mono float32."""
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if len(audio_int16) % 2 != 0:
            audio_int16 = audio_int16[:-1]
        
        stereo = audio_int16.reshape(-1, 2)
        # Average stereo channels to mono float32 (-1.0 to 1.0)
        mono_48k = stereo.mean(axis=1).astype(np.float32) / 32768.0
        
        # 3:1 integer decimation from 48 kHz to 16 kHz using anti-aliasing polyphase filter
        mono_16k = signal.resample_poly(mono_48k, 1, self.decimation_factor).astype(np.float32)
        return mono_16k

    def write(self, user, data: voice_recv.VoiceData):
        if not data or not data.pcm:
            return

        user_id = user.id if user else (getattr(data.packet, 'ssrc', 0) if hasattr(data, 'packet') else 0)
        speaker_name = getattr(user, 'display_name', getattr(user, 'name', f"User_{user_id}"))

        mono_16k = self._decimate_and_mono(data.pcm)
        if len(mono_16k) == 0:
            return

        if self.segmenter:
            self.segmenter.add_frame(user_id, speaker_name, mono_16k)


        with self.lock:
            # Append to user queue
            self.user_buffers[user_id].append(mono_16k)

            # If audio is present from multiple users, mix at 20ms slot with soft-clipping
            active_packets = []
            for u_id in list(self.user_buffers.keys()):
                if self.user_buffers[u_id]:
                    active_packets.append(self.user_buffers[u_id].popleft())

            if not active_packets:
                return

            # Linear float32 summation of multi-speaker audio + non-linear soft-clipping (np.tanh)
            combined = np.sum(active_packets, axis=0)
            soft_clipped = np.tanh(combined).astype(np.float32)

            self.rolling_buffer.append(soft_clipped)
            self.packet_count += 1
            if self.packet_count % 100 == 1:
                logger.info(f"🔊 [Audio Sink]: Active Discord audio stream ({len(self.rolling_buffer)} / {self.max_packets} slots)")

            # Feed to Vosk streaming chunk callback if registered
            if self.chunk_callback:
                self.chunk_accumulator.append(soft_clipped)
                total_samples = sum(len(c) for c in self.chunk_accumulator)
                if total_samples >= self.chunk_samples_needed:
                    chunk_float = np.concatenate(self.chunk_accumulator)
                    # Retain leftover samples for next chunk
                    if len(chunk_float) > self.chunk_samples_needed:
                        extra = chunk_float[self.chunk_samples_needed:]
                        chunk_float = chunk_float[:self.chunk_samples_needed]
                        self.chunk_accumulator = [extra]
                    else:
                        self.chunk_accumulator = []

                    # Convert float32 [-1.0, 1.0] to int16 PCM bytes for Vosk
                    chunk_int16 = (chunk_float * 32767).astype(np.int16)
                    self.chunk_callback(chunk_int16.tobytes())

    def get_audio_clip(self, user_id=None) -> np.ndarray:
        """
        Retrieves atomic snapshot of the last 30 seconds of audio
        as a float32 numpy array (16 kHz mono).
        """
        with self.lock:
            if not self.rolling_buffer:
                return np.array([], dtype=np.float32)
            # Atomic list copy from circular deque
            snapshot = list(self.rolling_buffer)

        return np.concatenate(snapshot)

    def get_wav_bytes(self) -> bytes:
        """Exports atomic snapshot of the last 30s audio to an in-memory WAV file."""
        audio_float = self.get_audio_clip()
        if len(audio_float) == 0:
            return b""

        audio_int16 = (audio_float * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.target_sr)  # 16000 Hz
            wf.writeframes(audio_int16.tobytes())
        wav_io.seek(0)
        return wav_io.read()

# Compatibility alias
RollingAudioSink = SynchronizedRollingSink