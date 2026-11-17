import os
import numpy as np

from download_model import ensure_wakeword_backbone
from wakeword_features import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WakeWordFeatureExtractor,
    StreamingFeatureBuffer,
)
from wakeword_metrics import compute_binary_metrics, calibrate_threshold
from train_wakeword import train_pipeline
from evaluate_wakeword import evaluate_all_conditions, generate_markdown_report
from keyword_spotter import KeywordSpotter


def test_wakeword_metrics_and_calibration():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.92, 0.81, 0.74, 0.15, 0.25, 0.40])

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["frr"] == 0.0
    assert m["far"] == 0.0
    assert m["roc_auc"] == 1.0

    thr = calibrate_threshold(y_true, y_prob)
    assert 0.40 <= thr <= 0.74


def test_streaming_and_offline_feature_shapes():
    mel_path, emb_path = ensure_wakeword_backbone()
    extractor = WakeWordFeatureExtractor(mel_path, emb_path)

    rng = np.random.default_rng(7)
    wav = (rng.normal(0, 0.15, WINDOW_SAMPLES) * 32767.0).astype(np.float32)

    seq = extractor.extract_sequence(wav)
    assert seq.shape == (16, 96)

    stream = StreamingFeatureBuffer(extractor)
    windows = []
    for i in range(0, len(wav), 1600):  # 100ms chunks like SynchronizedRollingSink
        windows.extend(stream.push_audio(wav[i : i + 1600]))

    assert len(windows) >= 1
    assert windows[-1].shape == (16, 96)


def test_end_to_end_training_onnx_export_and_evaluation(tmp_path):
    """
    Smoke test of the full trainable wake-word pipeline on synthetic 2-speaker waveforms:
    feature extraction -> LOSO CV -> ONNX export -> KeywordSpotter streaming -> evaluate_wakeword.
    """
    rng = np.random.default_rng(123)
    t = np.linspace(0, 2.5, int(SAMPLE_RATE * 2.5), endpoint=False)

    records = []
    for spk in ["spk1", "spk2"]:
        for idx in range(4):
            # Positive pattern: modulated 3-syllable tone burst + harmonics
            env = np.exp(-((t - 0.8) ** 2) / 0.02) + np.exp(-((t - 1.2) ** 2) / 0.02) + np.exp(-((t - 1.6) ** 2) / 0.02)
            pos_wav = (env * (0.4 * np.sin(2 * np.pi * (320 + idx * 15) * t) + 0.2 * np.sin(2 * np.pi * 640 * t)) * 32767.0).astype(np.float32)
            records.append(
                {
                    "id": f"{spk}_pos_{idx}",
                    "speaker_id": spk,
                    "category": "positive_isolated",
                    "label": 1,
                    "waveform": pos_wav,
                }
            )

            # Negative pattern: low-level broadband noise + steady low tone
            neg_wav = ((0.03 * rng.normal(0, 1, len(t)) + 0.08 * np.sin(2 * np.pi * 120 * t)) * 32767.0).astype(np.float32)
            records.append(
                {
                    "id": f"{spk}_neg_{idx}",
                    "speaker_id": spk,
                    "category": "negative_speech_noise",
                    "label": 0,
                    "waveform": neg_wav,
                }
            )

    onnx_path = str(tmp_path / "bot_clip_that.onnx")
    cfg_path = str(tmp_path / "model_config.json")
    loso_path = str(tmp_path / "loso_results.json")

    config = train_pipeline(
        records=records,
        output_onnx_path=onnx_path,
        output_config_path=cfg_path,
        loso_output_path=loso_path,
        public_negatives_path=None,
        augment_factor=1,
        epochs=6,
        batch_size=8,
        run_loso=True,
    )

    assert os.path.exists(onnx_path)
    assert os.path.exists(cfg_path)
    assert os.path.exists(loso_path)
    assert config["onnx_max_diff"] < 1e-3

    # Verify live KeywordSpotter loads exported ONNX and processes 100ms chunks
    mel_path, emb_path = ensure_wakeword_backbone()
    spotter = KeywordSpotter(
        model_path=onnx_path,
        mel_model_path=mel_path,
        emb_model_path=emb_path,
        config_path=cfg_path,
        cooldown_seconds=1.0,
    )
    assert spotter.is_ready is True

    # Run evaluation suite
    results = evaluate_all_conditions(
        records,
        onnx_path=onnx_path,
        config_path=cfg_path,
        loso_path=loso_path,
        public_negatives_path=None,
    )
    assert "baseline" in results and "streaming" in results
    assert results["streaming"]["mean_chunk_latency_ms"] < 50.0

    md = generate_markdown_report(results)
    assert "Quantitative Wake-Word Evaluation Report" in md
    assert "Leave-One-Speaker-Out" in md
