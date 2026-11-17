"""
Comprehensive Quantitative Wake-Word Evaluation Script (COMP6822001 LO-2 & LO-3).

Evaluates the trained ONNX 'bot clip that' classifier across:
  - Baseline: Clean speech (full dataset / specified split)
  - E1: Speaker Independence (5-Fold LOSO results from models/wakeword/loso_results.json)
  - E2: Background Noise Robustness (20 dB, 10 dB, 5 dB SNR)
  - E3: Acoustic & Speed Variations (Quiet 0.35x, Loud 1.4x, Slow 0.88x, Fast 1.12x)
  - E4: Streaming Latency & Continuous Negative False-Alarm Stress Test (FP/hr, ms/chunk)
  - Threshold Sweep & Error Analysis (False Positives / False Negatives by category)

Exports Markdown report to evaluation_wakeword_report.md.
"""

import os
import json
import time
import argparse
import numpy as np
from scipy import signal

from download_model import (
    ensure_wakeword_backbone,
    MEL_MODEL_PATH,
    EMB_MODEL_PATH,
    PUBLIC_NEG_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_CONFIG_PATH,
    WAKEWORD_MODEL_DIR,
)
from wakeword_features import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    SEQUENCE_LENGTH,
    WakeWordFeatureExtractor,
    extract_speech_centered_window,
)
from wakeword_metrics import compute_binary_metrics
from train_wakeword import DEFAULT_MANIFEST, LOSO_REPORT_PATH, load_dataset_records, predict_onnx
from keyword_spotter import KeywordSpotter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPORT_PATH = os.path.join(ROOT_DIR, "evaluation_wakeword_report.md")


def add_noise_at_snr(wav: np.ndarray, noise_pool: list[np.ndarray], snr_db: float, rng: np.random.Generator) -> np.ndarray:
    centered = extract_speech_centered_window(wav, window_samples=WINDOW_SAMPLES)
    if noise_pool:
        src = noise_pool[int(rng.integers(0, len(noise_pool)))]
        noise = extract_speech_centered_window(src, window_samples=WINDOW_SAMPLES, jitter_samples=3200, rng=rng)
    else:
        noise = rng.normal(0.0, 1.0, size=WINDOW_SAMPLES).astype(np.float32)

    sig_rms = float(np.sqrt(np.mean(centered ** 2))) + 1e-6
    noise_rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-6
    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    mixed = centered + noise * (target_noise_rms / noise_rms)
    return np.clip(mixed, -32767.0, 32767.0).astype(np.float32)


def change_speed(wav: np.ndarray, factor: float) -> np.ndarray:
    target_len = max(int(SAMPLE_RATE * 0.5), int(round(len(wav) / factor)))
    return signal.resample(wav, target_len).astype(np.float32)


def score_waveforms(
    waveforms: list[np.ndarray],
    labels: np.ndarray,
    extractor: WakeWordFeatureExtractor,
    onnx_path: str,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    feats = np.stack([extractor.extract_sequence(w) for w in waveforms], axis=0).astype(np.float32)
    probs = predict_onnx(onnx_path, feats)
    metrics = compute_binary_metrics(labels, probs, threshold=threshold)
    return probs, metrics


def run_streaming_stress_test(
    negative_records: list[dict],
    onnx_path: str,
    mel_path: str,
    emb_path: str,
    config_path: str,
) -> dict:
    """
    Feeds all negative clips sequentially as 100ms 16-bit PCM chunks through
    KeywordSpotter to measure real streaming latency and False Positives per Hour.
    """
    triggers = []
    spotter = KeywordSpotter(
        model_path=onnx_path,
        mel_model_path=mel_path,
        emb_model_path=emb_path,
        config_path=config_path,
        cooldown_seconds=1.5,
        trigger_callback=lambda msg: triggers.append(msg),
    )

    chunk_samples = 1600  # 100 ms @ 16 kHz (matches SynchronizedRollingSink)
    latencies_ms = []
    total_samples = 0

    for rec in negative_records:
        wav_int16 = np.clip(rec["waveform"], -32767, 32767).astype(np.int16)
        total_samples += len(wav_int16)
        for i in range(0, len(wav_int16), chunk_samples):
            chunk = wav_int16[i : i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            pcm_bytes = chunk.tobytes()
            t0 = time.perf_counter()
            spotter.process_chunk(pcm_bytes)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    total_hours = (total_samples / SAMPLE_RATE) / 3600.0 if total_samples > 0 else 0.0
    fp_per_hour = (len(triggers) / total_hours) if total_hours > 0 else 0.0
    mean_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p95_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

    return {
        "negative_duration_sec": round(total_samples / SAMPLE_RATE, 2),
        "chunks_processed": len(latencies_ms),
        "false_triggers": len(triggers),
        "fp_per_hour": round(fp_per_hour, 2),
        "mean_chunk_latency_ms": round(mean_ms, 3),
        "p95_chunk_latency_ms": round(p95_ms, 3),
        "streaming_rtf": round((mean_ms / 100.0), 4),
    }


def run_open_set_fp_benchmark(
    npy_path: str,
    onnx_path: str,
    threshold: float,
    patience_frames: int = 3,
    cooldown_seconds: float = 8.0,
) -> dict:
    """
    E5: Runs the trained classifier over the strictly held-out 70%..100% split
    of public_negative_features.npy (~3.2 hours of unseen speech/music/noise)
    to measure open-set False Positives per Hour (FP/hr).
    """
    arr = np.load(npy_path, mmap_mode="r")
    start_f = int(len(arr) * 0.70)
    sub = arr[start_f:]
    n_win = len(sub) - SEQUENCE_LENGTH + 1
    if n_win <= 0:
        return {}

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # Batch inference across sequential 80ms sliding windows
    probs = np.empty(n_win, dtype=np.float32)
    batch_sz = 1024
    for i in range(0, n_win, batch_sz):
        end = min(n_win, i + batch_sz)
        batch = np.stack([sub[j : j + SEQUENCE_LENGTH] for j in range(i, end)], axis=0).astype(np.float32)
        probs[i:end] = sess.run(None, {in_name: batch})[0].reshape(-1)

    consec = 0
    cooldown_steps = int(round(cooldown_seconds / 0.08))
    cooldown_left = 0
    triggers = 0
    for p in probs:
        if cooldown_left > 0:
            cooldown_left -= 1
            consec = 0
            continue
        if p >= threshold:
            consec += 1
            if consec >= patience_frames:
                triggers += 1
                consec = 0
                cooldown_left = cooldown_steps
        else:
            consec = 0

    hours = (len(sub) * 0.08) / 3600.0
    return {
        "duration_hours": round(hours, 2),
        "windows_evaluated": int(n_win),
        "false_triggers": triggers,
        "fp_per_hour": round(triggers / hours, 2) if hours > 0 else 0.0,
        "window_far_pct": round(float(np.mean(probs >= threshold) * 100.0), 4),
    }


def evaluate_all_conditions(
    records: list[dict],
    onnx_path: str = CLASSIFIER_MODEL_PATH,
    config_path: str = CLASSIFIER_CONFIG_PATH,
    loso_path: str = LOSO_REPORT_PATH,
    public_negatives_path: str | None = PUBLIC_NEG_PATH,
    seed: int = 42,
) -> dict:
    mel_path, emb_path = ensure_wakeword_backbone()
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"Trained wake-word ONNX model not found at {onnx_path}. Run train_wakeword.py first."
        )

    threshold = 0.5
    patience_frames = 3
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            threshold = float(cfg.get("threshold", 0.5))
            patience_frames = int(cfg.get("patience_frames", 3))

    extractor = WakeWordFeatureExtractor(mel_path, emb_path)
    rng = np.random.default_rng(seed)

    labels = np.array([r["label"] for r in records], dtype=np.int32)
    clean_wavs = [r["waveform"] for r in records]
    noise_pool = [r["waveform"] for r in records if r["label"] == 0 and r["category"] == "negative_speech_noise"]

    # Baseline: Clean
    clean_probs, baseline_metrics = score_waveforms(clean_wavs, labels, extractor, onnx_path, threshold)

    # E2: Background Noise
    noise_conditions = {}
    for snr in [20, 10, 5]:
        noisy_wavs = [add_noise_at_snr(w, noise_pool, snr, rng) for w in clean_wavs]
        _, m = score_waveforms(noisy_wavs, labels, extractor, onnx_path, threshold)
        noise_conditions[f"SNR_{snr}dB"] = m

    # E3: Acoustic & Speed Variations
    variation_conditions = {}
    variation_map = {
        "Quiet_0.35x": [np.clip(w * 0.35, -32767, 32767).astype(np.float32) for w in clean_wavs],
        "Loud_1.40x": [np.clip(w * 1.40, -32767, 32767).astype(np.float32) for w in clean_wavs],
        "Slow_0.88x": [change_speed(w, 0.88) for w in clean_wavs],
        "Fast_1.12x": [change_speed(w, 1.12) for w in clean_wavs],
    }
    for name, wavs in variation_map.items():
        _, m = score_waveforms(wavs, labels, extractor, onnx_path, threshold)
        variation_conditions[name] = m

    # E4: Streaming Stress Test on all negative recordings
    neg_records = [r for r in records if r["label"] == 0]
    streaming_metrics = run_streaming_stress_test(neg_records, onnx_path, mel_path, emb_path, config_path)

    # Threshold sweep on baseline
    sweep = [compute_binary_metrics(labels, clean_probs, thr) for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]

    # Error analysis samples
    errors = []
    for rec, prob in zip(records, clean_probs):
        pred = 1 if prob >= threshold else 0
        if pred != rec["label"]:
            errors.append(
                {
                    "id": rec["id"],
                    "speaker_id": rec["speaker_id"],
                    "category": rec["category"],
                    "true_label": rec["label"],
                    "pred_prob": round(float(prob), 4),
                    "error_type": "False Positive" if pred == 1 else "False Negative",
                }
            )

    loso_data = {}
    if os.path.exists(loso_path):
        with open(loso_path, "r", encoding="utf-8") as f:
            loso_data = json.load(f)

    open_set = {}
    if public_negatives_path and os.path.exists(public_negatives_path):
        open_set = run_open_set_fp_benchmark(
            public_negatives_path, onnx_path, threshold=threshold, patience_frames=patience_frames
        )

    return {
        "threshold": threshold,
        "patience_frames": patience_frames,
        "baseline": baseline_metrics,
        "loso": loso_data,
        "noise": noise_conditions,
        "variations": variation_conditions,
        "streaming": streaming_metrics,
        "open_set": open_set,
        "threshold_sweep": sweep,
        "errors": errors,
    }


def generate_markdown_report(results: dict) -> str:
    b = results["baseline"]
    st = results["streaming"]
    thr = results["threshold"]

    lines = [
        "# Quantitative Wake-Word Evaluation Report (`bot clip that`)",
        "",
        f"- **Calibrated Decision Threshold:** `{thr:.3f}`",
        f"- **Total Evaluated Clips:** {b['samples']} ({b['positives']} positive, {b['negatives']} negative)",
        f"- **Streaming Mean Chunk Latency (100 ms input):** `{st['mean_chunk_latency_ms']:.3f} ms` (P95: `{st['p95_chunk_latency_ms']:.3f} ms`, RTF: `{st['streaming_rtf']:.4f}x`)",
        f"- **Continuous Negative Stream False Alarm Rate (Group Negatives):** `{st['false_triggers']}` triggers over `{st['negative_duration_sec']}s` (`{st['fp_per_hour']:.2f} FP/hr`)",
    ]
    if results.get("open_set"):
        oset = results["open_set"]
        lines.append(
            f"- **E5 Open-Set Held-Out False Alarm Rate (Unseen Speech/Music/Noise):** `{oset['false_triggers']}` triggers over `{oset['duration_hours']} hours` (**`{oset['fp_per_hour']:.2f} FP/hr`**, window FAR `{oset['window_far_pct']:.3f}%`)"
        )
    lines.extend([
        "",
        "## 1. Baseline & Experimental Conditions (PDF Section 12–13)",
        "",
        "| Experiment | Condition | Accuracy | Precision | Recall | F1 | FRR | FAR | ROC-AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    def row(exp: str, cond: str, m: dict) -> str:
        return (
            f"| {exp} | {cond} | {m['accuracy']*100:.1f}% | {m['precision']*100:.1f}% | "
            f"{m['recall']*100:.1f}% | {m['f1']:.3f} | {m['frr']*100:.1f}% | {m['far']*100:.1f}% | {m['roc_auc']:.3f} |"
        )

    lines.append(row("Baseline", "Clean Speech", b))
    for cond, m in results["noise"].items():
        lines.append(row("E2 (Noise)", cond, m))
    for cond, m in results["variations"].items():
        lines.append(row("E3 (Acoustic)", cond, m))

    lines.extend(
        [
            "",
            "### Baseline Confusion Matrix",
            "",
            "| | Predicted Positive | Predicted Negative |",
            "|---|---:|---:|",
            f"| **Actual Positive (`bot clip that`)** | {b['tp']} (TP) | {b['fn']} (FN) |",
            f"| **Actual Negative (Speech / Noise)** | {b['fp']} (FP) | {b['tn']} (TN) |",
        ]
    )

    loso = results.get("loso", {})
    if loso.get("folds"):
        lines.extend(
            [
                "",
                "## 2. E1 — Speaker-Independent Leave-One-Speaker-Out (LOSO) Results",
                "",
                f"- **Mean Unseen-Speaker Accuracy:** `{loso['mean_accuracy']*100:.1f}%` | **Mean F1:** `{loso['mean_f1']:.3f}` | **Mean FRR:** `{loso['mean_frr']*100:.1f}%` | **Mean FAR:** `{loso['mean_far']*100:.1f}%`",
                "",
                "| Fold | Held-Out Speaker | Samples | Accuracy | F1 | FRR | FAR | ROC-AUC |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for f in loso["folds"]:
            lines.append(
                f"| {f['fold']} | {f['held_out_speaker']} | {f['samples']} | {f['accuracy']*100:.1f}% | "
                f"{f['f1']:.3f} | {f['frr']*100:.1f}% | {f['far']*100:.1f}% | {f['roc_auc']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Threshold Sensitivity Sweep (Baseline)",
            "",
            "| Threshold | Precision | Recall | F1 | FRR | FAR |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m in results["threshold_sweep"]:
        lines.append(
            f"| {m['threshold']:.2f} | {m['precision']*100:.1f}% | {m['recall']*100:.1f}% | "
            f"{m['f1']:.3f} | {m['frr']*100:.1f}% | {m['far']*100:.1f}% |"
        )

    lines.extend(["", "## 4. Error Analysis (Baseline Misclassifications)", ""])
    if results["errors"]:
        lines.append("| Clip ID | Speaker | Category | True Label | Predicted Prob | Error Type |")
        lines.append("|---|---|---|---:|---:|---|")
        for err in results["errors"][:25]:
            lines.append(
                f"| {err['id']} | {err['speaker_id']} | {err['category']} | "
                f"{err['true_label']} | {err['pred_prob']:.4f} | {err['error_type']} |"
            )
    else:
        lines.append("Zero misclassifications on the baseline clean evaluation set.")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained 'bot clip that' wake-word model.")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=str, default=CLASSIFIER_MODEL_PATH)
    parser.add_argument("--config", type=str, default=CLASSIFIER_CONFIG_PATH)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    records = load_dataset_records(args.manifest)
    results = evaluate_all_conditions(records, onnx_path=args.model, config_path=args.config)
    md = generate_markdown_report(results)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write(md)

    summary_json = os.path.join(WAKEWORD_MODEL_DIR, "eval_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wake-word evaluation complete!\n  Report: {args.report}\n  JSON:   {summary_json}")
    print(
        f"  Baseline Acc={results['baseline']['accuracy']*100:.1f}% | "
        f"F1={results['baseline']['f1']:.3f} | "
        f"Mean Stream Latency={results['streaming']['mean_chunk_latency_ms']:.3f} ms"
    )


if __name__ == "__main__":
    main()
