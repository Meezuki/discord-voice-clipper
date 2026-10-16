import time
import numpy as np
import pytest
from audio_sink import SynchronizedRollingSink

class DummyUser:
    def __init__(self, user_id):
        self.id = user_id

class DummyVoiceData:
    def __init__(self, pcm_bytes):
        self.pcm = pcm_bytes

def generate_pcm_sine(freq=440, duration_sec=0.02, sample_rate=48000, volume=1.0):
    """Creates a 20ms audio packet in 48kHz stereo int16 PCM format."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    # Stereo: 2 channels
    sine = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
    stereo = np.column_stack((sine, sine)).flatten()
    return stereo.tobytes()

def test_single_speaker_decimation_and_length():
    """Verify 48kHz stereo to 16kHz mono decimation (3:1 integer ratio)."""
    sink = SynchronizedRollingSink(buffer_seconds=5)
    user = DummyUser(1001)

    # Send 50 packets @ 20ms = 1.0 second of audio
    for _ in range(50):
        packet = generate_pcm_sine(freq=440, duration_sec=0.02, sample_rate=48000, volume=0.5)
        sink.write(user, DummyVoiceData(packet))

    clip = sink.get_audio_clip()
    # 1.0 second at 16000 Hz = ~16000 samples
    assert len(clip) > 0
    assert abs(len(clip) - 16000) <= 100
    assert clip.dtype == np.float32
    assert np.max(np.abs(clip)) <= 1.0

def test_multi_speaker_mixing_and_soft_clipping():
    """Verify summing 2 full-amplitude speakers (0 dB) does not cause integer wrap-around overflow."""
    sink = SynchronizedRollingSink(buffer_seconds=2)
    user1 = DummyUser(101)
    user2 = DummyUser(102)

    # Two loud simultaneous voices (volume 0.95)
    packet1 = generate_pcm_sine(freq=400, duration_sec=0.02, sample_rate=48000, volume=0.95)
    packet2 = generate_pcm_sine(freq=400, duration_sec=0.02, sample_rate=48000, volume=0.95)

    for _ in range(10):
        sink.write(user1, DummyVoiceData(packet1))
        sink.write(user2, DummyVoiceData(packet2))

    clip = sink.get_audio_clip()
    # Soft clipping must bound values in [-1.0, 1.0] without overflow wrap-around
    assert np.max(clip) <= 1.0
    assert np.min(clip) >= -1.0

def test_streaming_chunk_callback():
    """Verify streaming chunks (100-200ms) are emitted in real-time for the KWS module."""
    received_chunks = []
    def on_chunk(chunk_bytes):
        received_chunks.append(chunk_bytes)

    sink = SynchronizedRollingSink(buffer_seconds=5, chunk_callback=on_chunk, chunk_ms=100)
    user = DummyUser(201)

    # 5 packets @ 20ms = 100ms -> should trigger callback 1x
    for _ in range(5):
        packet = generate_pcm_sine(freq=300, duration_sec=0.02, sample_rate=48000)
        sink.write(user, DummyVoiceData(packet))

    assert len(received_chunks) == 1
    # 100ms at 16kHz 16-bit mono = 1600 samples * 2 bytes = 3200 bytes
    assert len(received_chunks[0]) == 3200

def test_buffer_freezing_snapshot():
    """Verify 30-second memory snapshot without clearing or corrupting ongoing capture."""
    sink = SynchronizedRollingSink(buffer_seconds=30)
    user = DummyUser(301)

    for _ in range(25):
        packet = generate_pcm_sine(freq=500, duration_sec=0.02)
        sink.write(user, DummyVoiceData(packet))

    snapshot1 = sink.get_audio_clip()
    # Continuous recording continues
    for _ in range(10):
        packet = generate_pcm_sine(freq=500, duration_sec=0.02)
        sink.write(user, DummyVoiceData(packet))

    snapshot2 = sink.get_audio_clip()
    assert len(snapshot2) > len(snapshot1)

