import numpy as np
from unittest.mock import MagicMock
from keyword_spotter import KeywordSpotter


def _make_mock_spotter(probs_sequence, threshold=0.5, patience_frames=2, cooldown_seconds=2.0, callback=None):
    spotter = KeywordSpotter(
        model_path=None,
        threshold=threshold,
        patience_frames=patience_frames,
        cooldown_seconds=cooldown_seconds,
        trigger_callback=callback,
    )
    spotter.classifier_session = MagicMock()
    spotter.classifier_input_name = "embeddings"
    spotter.feature_buffer = MagicMock()
    # Each process_chunk call emits one dummy (16, 96) window
    spotter.feature_buffer.push_audio.return_value = [np.zeros((16, 96), dtype=np.float32)]

    prob_iter = iter(probs_sequence)
    spotter.classifier_session.run.side_effect = lambda *args, **kwargs: [np.array([[next(prob_iter)]], dtype=np.float32)]
    return spotter


def test_keyword_spotter_missing_model_graceful_noop():
    """Verify missing ONNX model disables voice trigger gracefully without crashing."""
    spotter = KeywordSpotter(model_path="nonexistent_model.onnx")
    assert spotter.is_ready is False
    dummy_pcm = (np.ones(1600, dtype=np.int16) * 500).tobytes()
    assert spotter.process_chunk(dummy_pcm) is False


def test_keyword_spotter_patience_and_trigger():
    """Verify consecutive frames >= threshold (patience=2) trigger once and invoke callback."""
    triggered = []
    spotter = _make_mock_spotter(
        probs_sequence=[0.20, 0.85, 0.91],
        threshold=0.5,
        patience_frames=2,
        cooldown_seconds=8.0,
        callback=lambda label: triggered.append(label),
    )
    pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()

    assert spotter.process_chunk(pcm) is False  # 0.20 < 0.5
    assert spotter.process_chunk(pcm) is False  # 0.85 >= 0.5 (hit 1/2)
    assert spotter.process_chunk(pcm) is True   # 0.91 >= 0.5 (hit 2/2 -> trigger!)
    assert len(triggered) == 1
    assert "bot clip that" in triggered[0]


def test_keyword_spotter_resets_patience_on_drop():
    """Verify a single transient spike below patience_frames does NOT trigger."""
    spotter = _make_mock_spotter(
        probs_sequence=[0.88, 0.30, 0.89],
        threshold=0.5,
        patience_frames=2,
    )
    pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()

    assert spotter.process_chunk(pcm) is False  # hit 1
    assert spotter.process_chunk(pcm) is False  # drop -> reset to 0
    assert spotter.process_chunk(pcm) is False  # hit 1 again, not 2


def test_keyword_spotter_debounce_cooldown():
    """Verify debounce blocks rapid re-triggers until cooldown_seconds elapses."""
    spotter = _make_mock_spotter(
        probs_sequence=[0.95, 0.96, 0.97, 0.98, 0.99, 0.99],
        threshold=0.5,
        patience_frames=2,
        cooldown_seconds=5.0,
    )
    pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()

    assert spotter.process_chunk(pcm) is False
    assert spotter.process_chunk(pcm) is True   # First trigger

    # Immediate subsequent hits within cooldown must return False
    assert spotter.process_chunk(pcm) is False
    assert spotter.process_chunk(pcm) is False

    # Simulate cooldown expiry
    spotter.last_trigger_time -= 5.5
    assert spotter.process_chunk(pcm) is False  # hit 1
    assert spotter.process_chunk(pcm) is True   # hit 2 -> Second trigger
