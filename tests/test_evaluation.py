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

