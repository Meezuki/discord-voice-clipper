import pytest
import numpy as np
from unittest.mock import MagicMock
from evaluate_wer import WERBenchmarkEvaluator

def test_wer_evaluation_calculation():
    """Verify calculation of WER, substitutions, deletions, and insertions."""
    mock_asr = MagicMock()
    # Mocking translation with 1 substitution: "today" -> "now"
    mock_asr.transcribe.return_value = ("the team meeting starts now", 0.8)

    evaluator = WERBenchmarkEvaluator(asr_engine=mock_asr)
    dummy_audio = np.zeros(32000, dtype=np.float32)  # 2.0 seconds

    reference = "the team meeting starts today"
    res = evaluator.evaluate_sample(dummy_audio, reference)

    assert res["total_words"] == 5
    assert res["substitutions"] == 1
    assert res["deletions"] == 0
    assert res["insertions"] == 0
    assert abs(res["wer"] - (1.0 / 5.0)) < 1e-4
    assert res["duration_sec"] == 2.0
    assert res["latency_sec"] == 0.8
    assert res["rtf"] == 0.4  # 0.8 / 2.0

def test_markdown_report_generation():
    """Verify markdown report generation formatting."""
    evaluator = WERBenchmarkEvaluator(asr_engine=MagicMock())
    results = [
        {
            "reference": "hello everyone",
            "hypothesis": "hello everyone",
            "duration_sec": 1.0,
            "latency_sec": 0.2,
            "rtf": 0.2,
            "wer": 0.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "hits": 2,
            "total_words": 2
        }
    ]
    report = evaluator.generate_markdown_report(results)
    assert "Word Error Rate" in report
    assert "Real-Time Factor" in report
    assert "hello everyone" in report

def test_load_wav_file_resampling(tmp_path):
    """Verify that load_wav_file correctly resamples 44.1kHz stereo to 16kHz mono float32."""
    import wave
    from evaluate_wer import load_wav_file

    wav_file = tmp_path / "test_44k.wav"
    sr = 44100
    duration = 1.0  # 1 second = 44100 samples
    samples = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, sr)) * 10000).astype(np.int16)
    stereo_samples = np.column_stack((samples, samples)).flatten()

    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo_samples.tobytes())

    loaded = load_wav_file(str(wav_file), target_sr=16000)
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.float32
    # 1.0 second at 16000 Hz should yield approximately 16000 samples
    assert abs(len(loaded) - 16000) <= 2


