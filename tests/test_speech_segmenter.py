import pytest
import numpy as np
from audio_sink import SpeechSegmenter

def generate_float32_sine(freq=440, duration_sec=0.02, sample_rate=16000, volume=0.1):
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * volume).astype(np.float32)

def test_speech_segmenter_detects_speech_and_flushes_after_silence():
    """Verify that speech followed by silence flushes a complete audio segment."""
    flushed_segments = []

    def on_segment(user_id, speaker_name, audio_data):
        flushed_segments.append((user_id, speaker_name, audio_data))

    segmenter = SpeechSegmenter(
        segment_callback=on_segment,
        speech_threshold=0.01,
        silence_timeout_sec=0.4,
        min_duration_sec=0.5
    )

    # 1. Feed 50 frames @ 20ms of speech = 1.0 second
    speech_frame = generate_float32_sine(volume=0.2)
    for _ in range(50):
        segmenter.add_frame(101, "Alice", speech_frame)

    assert len(flushed_segments) == 0  # Still speaking

    # 2. Feed 25 frames @ 20ms of silence = 0.5s (> 0.4s timeout)
    silence_frame = np.zeros(320, dtype=np.float32)
    for _ in range(25):
        segmenter.add_frame(101, "Alice", silence_frame)

    # Must flush now
    assert len(flushed_segments) == 1
    u_id, name, audio = flushed_segments[0]
    assert u_id == 101
    assert name == "Alice"
    assert len(audio) >= 50 * 320  # At least 1.0 second of audio + padding


def test_speech_segmenter_ignores_pure_silence():
    """Verify that pure silence frames never trigger a segment flush."""
    flushed = []
    segmenter = SpeechSegmenter(segment_callback=lambda u, n, a: flushed.append(a))

    silence = np.zeros(320, dtype=np.float32)
    for _ in range(100):
        segmenter.add_frame(102, "Bob", silence)

    assert len(flushed) == 0

def test_speech_segmenter_max_duration_split():
    """Verify that continuous speech exceeding max_duration_sec is automatically flushed."""
    flushed = []
    segmenter = SpeechSegmenter(
        segment_callback=lambda u, n, a: flushed.append(a),
        max_duration_sec=2.0
    )

    speech = generate_float32_sine(volume=0.2)
    # 120 frames @ 20ms = 2.4s (> 2.0s max duration)
    for _ in range(120):
        segmenter.add_frame(103, "Charlie", speech)

    assert len(flushed) >= 1
    assert len(flushed[0]) == 100 * 320  # exactly 2.0 seconds
