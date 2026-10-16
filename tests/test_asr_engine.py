import numpy as np
import pytest
from unittest.mock import MagicMock
from asr_engine import LocalASREngine

def test_asr_engine_empty_audio():
    """Verify that empty audio returns empty text and 0.0 latency."""
    asr = LocalASREngine(model_size=None)
    text, latency = asr.transcribe(np.array([], dtype=np.float32))
    assert text == ""
    assert latency == 0.0

def test_asr_engine_transcription_mock():
    """Verify transcription process and latency calculation."""
    asr = LocalASREngine(model_size=None)
    
    # Mock model
    mock_model = MagicMock()
    mock_segment1 = MagicMock()
    mock_segment1.text = "The coordination meeting"
    mock_segment2 = MagicMock()
    mock_segment2.text = "finished on schedule."
    mock_model.transcribe.return_value = ([mock_segment1, mock_segment2], None)

    asr.model = mock_model
    dummy_audio = np.zeros(16000, dtype=np.float32)
    text, latency = asr.transcribe(dummy_audio)

    assert "The coordination meeting finished on schedule." in text
    assert latency >= 0.0

