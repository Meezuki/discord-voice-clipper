import time
import numpy as np
import pytest
from unittest.mock import MagicMock
from keyword_spotter import KeywordSpotter

def test_keyword_spotter_debounce():
    """Verify that debounce locks the trigger for cooldown_seconds after wake word detection."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    # Simulate detecting "bot clip that"
    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "please bot clip that"}'

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

def test_keyword_spotter_bot_word_alone_does_not_trigger():
    """Verify that saying just 'bot' or 'paper clip' does NOT trigger the wake word."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "hey bot"}'

    dummy_pcm = b"\x00" * 3200
    detected = spotter.process_chunk(dummy_pcm)
    assert detected is False

    # Also non-command 'paper clip' should not trigger
    spotter.recognizer.Result.return_value = '{"text": "bring me a paper clip"}'
    assert spotter.process_chunk(dummy_pcm) is False

def test_keyword_spotter_clip_that_trigger():
    """Verify that intentional phrase 'bot clip that' triggers the wake word."""
    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0)
    spotter.recognizer = MagicMock()

    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "bot clip that"}'

    dummy_pcm = b"\x00" * 3200
    detected = spotter.process_chunk(dummy_pcm)
    assert detected is True

def test_keyword_spotter_whisper_fallback():
    """Verify that when Vosk returns unk or empty, Whisper spotter fallback catches 'bot clip that'."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ("hello everyone please bot clip that now", 0.05)

    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0, asr_engine=mock_asr)
    spotter.recognizer = MagicMock()
    spotter.recognizer.AcceptWaveform.return_value = False
    spotter.recognizer.PartialResult.return_value = '{"partial": "[unk]"}'

    # Force last_whisper_check to past
    spotter.last_whisper_check = 0.0

    # Ingest enough audio to exceed whisper_samples_target (e.g. 2.5s with non-zero sound)
    detected = False
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

def test_keyword_spotter_two_stage_verification_rejects_false_positive():
    """Verify that if Vosk misrecognizes conversational speech as a trigger, Whisper verification rejects it."""
    mock_asr = MagicMock()
    # Whisper reveals user actually said conversational speech without any clip command
    mock_asr.transcribe.return_value = ("what is listening in the voice channel general", 0.05)

    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0, asr_engine=mock_asr)
    spotter.recognizer = MagicMock()
    spotter.recognizer.AcceptWaveform.return_value = True
    # Vosk erroneously heard 'clip'
    spotter.recognizer.Result.return_value = '{"text": "clip"}'

    # Provide audible chunk so verification evaluates
    chunk = (np.ones(1600, dtype=np.int16) * 5000).tobytes()
    detected = spotter.process_chunk(chunk)

    assert detected is False  # Suppressed by Whisper verification!
    mock_asr.transcribe.assert_called()

def test_keyword_spotter_two_stage_verification_confirms_true_positive():
    """Verify that if Vosk detects trigger and Whisper confirms it, the trigger fires."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ("please bot clip that meeting", 0.05)

    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0, asr_engine=mock_asr)
    spotter.recognizer = MagicMock()
    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "bot clip that"}'

    chunk = (np.ones(1600, dtype=np.int16) * 5000).tobytes()
    detected = spotter.process_chunk(chunk)

    assert detected is True  # Confirmed!
    mock_asr.transcribe.assert_called()

def test_keyword_spotter_whisper_punctuation_handling():
    """Verify that Whisper output containing punctuation like 'Bot, clip that.' triggers properly."""
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = ("Bot, clip that.", 0.05)

    spotter = KeywordSpotter(model_path=None, cooldown_seconds=2.0, asr_engine=mock_asr)
    spotter.recognizer = MagicMock()
    spotter.recognizer.AcceptWaveform.return_value = True
    spotter.recognizer.Result.return_value = '{"text": "bot clip that"}'

    chunk = (np.ones(1600, dtype=np.int16) * 5000).tobytes()
    detected = spotter.process_chunk(chunk)

    assert detected is True


