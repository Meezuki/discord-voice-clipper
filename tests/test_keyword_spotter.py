import time
import numpy as np
import pytest
from unittest.mock import MagicMock
from keyword_spotter import KeywordSpotter

def test_keyword_spotter_debounce():
    """Verify that debounce locks the trigger for cooldown_seconds after wake word detection."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    # Simulate detecting "bot record"
    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "please bot record this"}'

    dummy_pcm = b"\x00" * 3200 # 100ms audio chunk
    detected1 = spotter.process_chunk(dummy_pcm)
    assert detected1 is True

    # Immediate subsequent call within debounce cooldown must return False
    detected2 = spotter.process_chunk(dummy_pcm)
    assert detected2 is False

    # After cooldown expires
    spotter.last_trigger_time -= 2.1
    detected3 = spotter.process_chunk(dummy_pcm)
    assert detected3 is True

def test_keyword_spotter_record_phrase():
    """Verify that saying just 'record' also triggers the wake word."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "record"}'

    dummy_pcm = b"\x00" * 3200
    detected = spotter.process_chunk(dummy_pcm)
    assert detected is True

def test_keyword_spotter_but_record_phonetic_variant():
    """Verify that acoustic variant 'but record' triggers the wake word."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "but record"}'

    dummy_pcm = b"\x00" * 3200
    detected = spotter.process_chunk(dummy_pcm)
    assert detected is True

def test_keyword_spotter_whisper_fallback():
    """Verify that when Vosk returns unk or empty, Whisper spotter fallback catches 'record'."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ("hello everyone please record now", 0.05)

    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0, asr_engine=mock_asr)
    spotter.recognizer = MagicMock()
    spotter.recognizer.AcceptWaveform.return_value = False
    spotter.recognizer.PartialResult.return_value = '{"partial": "[unk]"}'

    # Force last_whisper_check to past
    spotter.last_whisper_check = 0.0

    # Ingest enough audio to exceed whisper_samples_target (e.g. 2.5s with non-zero sound)
    # 2.5s = 25 chunks @ 100ms (1600 samples)
    detected = False
    # Non-silent chunk
    chunk = (np.ones(1600, dtype=np.int16) * 5000).tobytes()
    for _ in range(26):
        if spotter.process_chunk(chunk):
            detected = True
            break

    assert detected is True
    mock_asr.transcribe.assert_called()

def test_keyword_spotter_ignores_non_trigger_words():
    """Verify that non-trigger sentences do not fire."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "good morning everyone let us begin"}'

    dummy_pcm = b"\x00" * 3200
    detected = spotter.process_chunk(dummy_pcm)
    assert detected is False

