"""
Trains a custom PyTorch BiGRU + Attention classifier head on top of frozen
speech embeddings (melspectrogram.onnx + embedding_model.onnx) using the
group's recorded dataset (data/wakeword/metadata.json).

Supports:
  - Local Windows CPU training (automatic CUDA if available)
  - Online waveform augmentation (time jitter, gain, speed, noise mixing)
  - 5-Fold Leave-One-Speaker-Out (LOSO) cross-validation for speaker independence
  - ONNX export (models/wakeword/bot_clip_that.onnx) + threshold calibration
"""

import os
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy import signal
import onnxruntime as ort

from download_model import (
    ensure_wakeword_backbone,
    PUBLIC_NEG_PATH,
    WAKEWORD_MODEL_DIR,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_CONFIG_PATH,
)
from wakeword_features import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    WakeWordFeatureExtractor,
    load_wav_int16_scale,
    extract_speech_centered_window,
)
from wakeword_metrics import compute_binary_metrics, calibrate_threshold

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(ROOT_DIR, "data", "wakeword", "metadata.json")
LOSO_REPORT_PATH = os.path.join(WAKEWORD_MODEL_DIR, "loso_results.json")


class WakeWordBiGRU(nn.Module):
    """
    Lightweight temporal wake-word classifier operating on (batch, 16, 96)
    frozen speech embedding sequences (~2.0 s receptive field).
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, hidden_dim: int = 32, dropout: float = 0.35):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        gru_out_dim = hidden_dim * 2
        self.attn = nn.Linear(gru_out_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(gru_out_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 96)
        h = self.proj(self.norm(x))
        gru_out, _ = self.gru(h)  # (batch, seq_len, hidden*2)
        attn_weights = torch.softmax(self.attn(gru_out), dim=1)  # (batch, seq_len, 1)
        attn_pool = torch.sum(gru_out * attn_weights, dim=1)     # (batch, hidden*2)
        mean_pool = torch.mean(gru_out, dim=1)                   # (batch, hidden*2)
        combined = torch.cat([attn_pool, mean_pool], dim=-1)     # (batch, hidden*4)
        return self.head(combined).squeeze(-1)                   # (batch,) logits


class WakeWordONNXWrapper(nn.Module):
    """Wraps WakeWordBiGRU with Sigmoid so ONNX outputs calibrated probabilities [0, 1]."""

    def __init__(self, base_model: WakeWordBiGRU):
        super().__init__()
        self.base_model = base_model

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(embeddings)
        return torch.sigmoid(logits).unsqueeze(-1)


def augment_waveform(
    wav: np.ndarray,
    noise_pool: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Applies random speed, gain, time-jitter, and additive noise to int16-scale audio."""
    out = wav.astype(np.float32, copy=True)

    # 1. Speed perturbation (0.90x - 1.10x)
    if rng.random() < 0.7:
        speed = float(rng.uniform(0.90, 1.10))
        target_len = max(int(SAMPLE_RATE * 0.5), int(round(len(out) / speed)))
        out = signal.resample(out, target_len).astype(np.float32)

    # 2. Extract 2.0s speech-centered window with random time jitter (+-200ms)
    out = extract_speech_centered_window(
        out,
        window_samples=WINDOW_SAMPLES,
        jitter_samples=int(SAMPLE_RATE * 0.20),
        rng=rng,
    )

    # 3. Gain scaling
    gain = float(rng.uniform(0.45, 1.35))
    out = out * gain

    # 4. Additive noise (either recorded ambient/negative clip or Gaussian noise)
    if rng.random() < 0.75:
        if noise_pool and rng.random() < 0.65:
            noise_src = noise_pool[int(rng.integers(0, len(noise_pool)))]
            noise = extract_speech_centered_window(noise_src, window_samples=WINDOW_SAMPLES, jitter_samples=4000, rng=rng)
        else:
            noise = rng.normal(0.0, 1.0, size=WINDOW_SAMPLES).astype(np.float32)

        sig_rms = float(np.sqrt(np.mean(out ** 2))) + 1e-6
        noise_rms = float(np.sqrt(np.mean(noise ** 2))) + 1e-6
        snr_db = float(rng.uniform(6.0, 24.0))
        target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
        out = out + noise * (target_noise_rms / noise_rms)

    return np.clip(out, -32767.0, 32767.0).astype(np.float32)


def load_dataset_records(manifest_path: str = DEFAULT_MANIFEST) -> list[dict]:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Dataset manifest not found at {manifest_path}. "
            "Record your dataset first using: python tools/record_dataset.py"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        raise ValueError(f"Dataset manifest {manifest_path} is empty.")

    loaded = []
    for rec in records:
        wav_path = rec["path"]
        if not os.path.isabs(wav_path):
            wav_path = os.path.join(ROOT_DIR, wav_path)
        if not os.path.exists(wav_path):
            continue
        wav = load_wav_int16_scale(wav_path)
        loaded.append(
            {
                "id": rec.get("id", os.path.basename(wav_path)),
                "speaker_id": rec["speaker_id"],
                "category": rec.get("category", "unknown"),
                "label": int(rec["label"]),
                "waveform": wav,
                "path": wav_path,
            }
        )
    return loaded


def _extract_sliding_negative_windows(
    extractor: WakeWordFeatureExtractor,
    wav: np.ndarray,
    stride: int = 2,
) -> list[np.ndarray]:
    """
    Extracts sliding (16, 96) windows across a negative clip plus silence-onset
    transitions so training matches live StreamingFeatureBuffer behavior.
    """
    windows = []
    embs = extractor.extract_all_embeddings(wav)
    for i in range(0, max(0, len(embs) - SEQUENCE_LENGTH + 1), stride):
        windows.append(embs[i : i + SEQUENCE_LENGTH])

    # Explicitly include silence -> speech onset windows (first 1.2s after 1.0s silence)
    onset_wav = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), wav[:SAMPLE_RATE]])
    onset_embs = extractor.extract_all_embeddings(onset_wav)
    for i in range(0, max(0, len(onset_embs) - SEQUENCE_LENGTH + 1), 3):
        windows.append(onset_embs[i : i + SEQUENCE_LENGTH])

    return windows


def build_feature_bank(
    records: list[dict],
    extractor: WakeWordFeatureExtractor,
    augment_factor: int = 4,
    seed: int = 42,
) -> list[dict]:
    """
    Precomputes clean, sliding-negative, and augmented (16, 96) embedding sequences
    so training matches continuous streaming and LOSO folds run in seconds on CPU.
    """
    rng = np.random.default_rng(seed)
    noise_pool = [
        r["waveform"]
        for r in records
        if r["label"] == 0 and r["category"] == "negative_speech_noise"
    ]

    bank = []
    for rec in records:
        clean_emb = extractor.extract_sequence(rec["waveform"])
        aug_embs = []
        sliding_neg_embs = []

        if rec["label"] == 0:
            sliding_neg_embs = _extract_sliding_negative_windows(extractor, rec["waveform"], stride=2)
            aug_embs.extend(sliding_neg_embs)
            # Include pure silence / ambient noise floor variants as explicit negatives
            if rng.random() < 0.3:
                aug_embs.append(extractor.extract_sequence(np.zeros(WINDOW_SAMPLES, dtype=np.float32)))

        for _ in range(augment_factor):
            aug_wav = augment_waveform(rec["waveform"], noise_pool, rng)
            aug_embs.append(extractor.extract_sequence(aug_wav))

        bank.append(
            {
                "id": rec["id"],
                "speaker_id": rec["speaker_id"],
                "category": rec["category"],
                "label": rec["label"],
                "clean_emb": clean_emb,
                "aug_embs": aug_embs,
                "sliding_neg_embs": sliding_neg_embs,
            }
        )
    return bank


def _pack_features(
    bank_items: list[dict],
    include_augmented: bool,
    include_sliding_negs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for item in bank_items:
        xs.append(item["clean_emb"])
        ys.append(item["label"])
        if include_augmented:
            for aug in item["aug_embs"]:
                xs.append(aug)
                ys.append(item["label"])
        elif include_sliding_negs and item["label"] == 0:
            for sw in item.get("sliding_neg_embs", []):
                xs.append(sw)
                ys.append(0)
    return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.float32)


def _stratified_split(items: list[dict], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val_pos = max(1, int(round(len(pos) * val_ratio))) if len(pos) > 1 else 0
    n_val_neg = max(1, int(round(len(neg) * val_ratio))) if len(neg) > 1 else 0

    val_items = pos[:n_val_pos] + neg[:n_val_neg]
    train_items = pos[n_val_pos:] + neg[n_val_neg:]
    if not val_items:
        val_items = train_items[:]
    return train_items, val_items


def load_public_negative_splits(
    npy_path: str = PUBLIC_NEG_PATH,
    train_count: int = 12000,
    val_count: int = 2000,
    mine_count: int = 30000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Samples disjoint (16, 96) negative windows from public_negative_features.npy:
      - 0%..60%: train_negs + mine_pool (for Stage 2 hard-negative mining)
      - 60%..70%: val_negs (for validation & threshold calibration)
      - 70%..100%: untouched (reserved strictly for evaluate_wakeword.py E5)
    """
    if not npy_path or not os.path.exists(npy_path):
        return None, None, None

    arr = np.load(npy_path, mmap_mode="r")
    n_frames = len(arr)
    if n_frames < SEQUENCE_LENGTH * 10:
        return None, None, None

    train_end = int(n_frames * 0.60)
    val_end = int(n_frames * 0.70)
    rng = np.random.default_rng(seed)

    def _sample_windows(start_f: int, end_f: int, count: int) -> np.ndarray:
        max_start = end_f - SEQUENCE_LENGTH
        if max_start <= start_f:
            return np.zeros((0, SEQUENCE_LENGTH, EMBEDDING_DIM), dtype=np.float32)
        starts = rng.integers(start_f, max_start, size=count)
        return np.stack([arr[s : s + SEQUENCE_LENGTH] for s in starts], axis=0).astype(np.float32)

    train_negs = _sample_windows(0, train_end, train_count)
    mine_negs = _sample_windows(0, train_end, mine_count)
    val_negs = _sample_windows(train_end, val_end, val_count)
    return train_negs, val_negs, mine_negs


def _run_epochs(
    model: WakeWordBiGRU,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    y_val_np: np.ndarray,
    criterion: nn.Module,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, len(X_train)),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for _ in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(bx)

        scheduler.step()
        train_loss = running_loss / max(1, len(X_train))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        val_metrics = compute_binary_metrics(y_val_np, val_probs, threshold=0.5)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(val_metrics["f1"])

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    model.load_state_dict(best_state)
    return history


def train_classifier(
    train_items: list[dict],
    val_items: list[dict],
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
    extra_train_negs: np.ndarray | None = None,
    extra_val_negs: np.ndarray | None = None,
    mine_pool_negs: np.ndarray | None = None,
) -> tuple[WakeWordBiGRU, float, dict]:
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = _pack_features(train_items, include_augmented=True)
    X_val, y_val = _pack_features(val_items, include_augmented=False, include_sliding_negs=True)

    if extra_train_negs is not None and len(extra_train_negs) > 0:
        X_train = np.concatenate([X_train, extra_train_negs], axis=0)
        y_train = np.concatenate([y_train, np.zeros(len(extra_train_negs), dtype=np.float32)], axis=0)

    if extra_val_negs is not None and len(extra_val_negs) > 0:
        X_val = np.concatenate([X_val, extra_val_negs], axis=0)
        y_val = np.concatenate([y_val, np.zeros(len(extra_val_negs), dtype=np.float32)], axis=0)

    pos_weight = torch.tensor([1.5], dtype=torch.float32, device=device)
    model = WakeWordBiGRU().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    history = _run_epochs(
        model, X_train, y_train, X_val_t, y_val_t, y_val, criterion, epochs, batch_size, lr, device
    )

    # Stage 2: Hard-Negative Mining on unseen general speech/music/noise pool
    mined_count = 0
    if mine_pool_negs is not None and len(mine_pool_negs) > 0:
        model.eval()
        pool_probs = []
        with torch.no_grad():
            for i in range(0, len(mine_pool_negs), 512):
                bx = torch.from_numpy(mine_pool_negs[i : i + 512]).to(device)
                pool_probs.append(torch.sigmoid(model(bx)).cpu().numpy())
        all_p = np.concatenate(pool_probs)
        hard_idx = np.where(all_p >= 0.10)[0]
        if len(hard_idx) > 0:
            # Take up to 2,500 hardest negatives sorted by confidence descending
            hard_idx = hard_idx[np.argsort(-all_p[hard_idx])[:2500]]
            hard_X = mine_pool_negs[hard_idx]
            mined_count = len(hard_X)
            X_train = np.concatenate([X_train, hard_X], axis=0)
            y_train = np.concatenate([y_train, np.zeros(mined_count, dtype=np.float32)], axis=0)
            history = _run_epochs(
                model, X_train, y_train, X_val_t, y_val_t, y_val, criterion, 15, batch_size, lr * 0.5, device
            )

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()

    calibrated_thr = calibrate_threshold(y_val, val_probs)
    val_summary = compute_binary_metrics(y_val, val_probs, threshold=calibrated_thr)
    return model.cpu(), calibrated_thr, {
        "val_metrics": val_summary,
        "epochs_run": len(history["train_loss"]),
        "mined_hard_negatives": mined_count,
        "history": history,
    }


def predict_onnx(onnx_path: str, X: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: X.astype(np.float32)})[0]
    return out.reshape(-1)


def export_to_onnx(model: WakeWordBiGRU, onnx_path: str, sample_X: np.ndarray) -> float:
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    wrapper = WakeWordONNXWrapper(model).eval()
    dummy_in = torch.zeros(1, SEQUENCE_LENGTH, EMBEDDING_DIM, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_in,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["embeddings"],
        output_names=["probability"],
        dynamic_axes={"embeddings": {0: "batch_size"}, "probability": {0: "batch_size"}},
        dynamo=False,
    )

    # Verify ONNX output matches PyTorch output
    with torch.no_grad():
        pt_probs = wrapper(torch.from_numpy(sample_X[:16].astype(np.float32))).numpy().reshape(-1)
    ort_probs = predict_onnx(onnx_path, sample_X[:16])
    max_diff = float(np.max(np.abs(pt_probs - ort_probs)))
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX verification failed: max diff {max_diff} > 1e-3")
    return max_diff


def run_loso_cv(
    bank: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    extra_train_negs: np.ndarray | None = None,
    extra_val_negs: np.ndarray | None = None,
    mine_pool_negs: np.ndarray | None = None,
) -> dict:
    speakers = sorted({item["speaker_id"] for item in bank})
    if len(speakers) < 2:
        return {"folds": [], "note": "Need >= 2 speakers for Leave-One-Speaker-Out evaluation."}

    fold_results = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, held_out in enumerate(speakers, start=1):
        test_items = [x for x in bank if x["speaker_id"] == held_out]
        remain_items = [x for x in bank if x["speaker_id"] != held_out]
        train_items, val_items = _stratified_split(remain_items, val_ratio=0.15, seed=seed + fold_idx)

        model, thr, _ = train_classifier(
            train_items,
            val_items,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + fold_idx,
            device=device,
            extra_train_negs=extra_train_negs,
            extra_val_negs=extra_val_negs,
            mine_pool_negs=mine_pool_negs,
        )
        X_test, y_test = _pack_features(test_items, include_augmented=False)
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(X_test))).numpy()

        metrics = compute_binary_metrics(y_test, probs, threshold=thr)
        metrics["held_out_speaker"] = held_out
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())
        print(
            f"[LOSO Fold {fold_idx}/{len(speakers)} | Unseen: {held_out}] "
            f"Acc={metrics['accuracy']*100:.1f}% F1={metrics['f1']:.3f} "
            f"FRR={metrics['frr']*100:.1f}% FAR={metrics['far']*100:.1f}% (thr={thr:.2f})"
        )

    pooled = compute_binary_metrics(np.array(all_y_true), np.array(all_y_prob), threshold=0.5)
    summary = {
        "num_folds": len(fold_results),
        "mean_accuracy": round(float(np.mean([f["accuracy"] for f in fold_results])), 4),
        "mean_f1": round(float(np.mean([f["f1"] for f in fold_results])), 4),
        "mean_frr": round(float(np.mean([f["frr"] for f in fold_results])), 4),
        "mean_far": round(float(np.mean([f["far"] for f in fold_results])), 4),
        "mean_roc_auc": round(float(np.mean([f["roc_auc"] for f in fold_results])), 4),
        "pooled_at_0_5": pooled,
        "folds": fold_results,
    }
    return summary


def train_pipeline(
    records: list[dict],
    output_onnx_path: str = CLASSIFIER_MODEL_PATH,
    output_config_path: str = CLASSIFIER_CONFIG_PATH,
    loso_output_path: str = LOSO_REPORT_PATH,
    public_negatives_path: str | None = PUBLIC_NEG_PATH,
    augment_factor: int = 8,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    run_loso: bool = True,
) -> dict:
    t0 = time.perf_counter()
    mel_path, emb_path = ensure_wakeword_backbone()
    extractor = WakeWordFeatureExtractor(mel_path, emb_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device} | Total raw clips: {len(records)}")

    extra_train_negs, extra_val_negs, mine_pool_negs = load_public_negative_splits(
        public_negatives_path, seed=seed
    )
    if extra_train_negs is not None:
        print(
            f"Loaded public negative background pool: {len(extra_train_negs)} train windows, "
            f"{len(extra_val_negs)} val windows, {len(mine_pool_negs)} mining pool"
        )

    bank = build_feature_bank(records, extractor, augment_factor=augment_factor, seed=seed)
    speakers = sorted({r["speaker_id"] for r in bank})
    print(f"Speakers ({len(speakers)}): {speakers}")

    loso_summary = {}
    if run_loso and len(speakers) >= 2:
        print("\nRunning Leave-One-Speaker-Out (LOSO) cross-validation...")
        loso_summary = run_loso_cv(
            bank,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
            extra_train_negs=extra_train_negs,
            extra_val_negs=extra_val_negs,
            mine_pool_negs=mine_pool_negs,
        )
        os.makedirs(os.path.dirname(loso_output_path), exist_ok=True)
        with open(loso_output_path, "w", encoding="utf-8") as f:
            json.dump(loso_summary, f, indent=2)

    print("\nTraining final production model on full group dataset...")
    train_items, val_items = _stratified_split(bank, val_ratio=0.15, seed=seed)
    final_model, best_thr, train_info = train_classifier(
        train_items,
        val_items,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        extra_train_negs=extra_train_negs,
        extra_val_negs=extra_val_negs,
        mine_pool_negs=mine_pool_negs,
    )

    sample_X, _ = _pack_features(val_items, include_augmented=False)
    max_diff = export_to_onnx(final_model, output_onnx_path, sample_X)

    config = {
        "wake_word": "bot clip that",
        "architecture": "Frozen openWakeWord ONNX Embedding + PyTorch BiGRU-Attention",
        "sequence_length": SEQUENCE_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "sample_rate": SAMPLE_RATE,
        "threshold": best_thr,
        "patience_frames": 3,
        "cooldown_seconds": 8.0,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "total_clips": len(records),
        "augment_factor": augment_factor,
        "onnx_max_diff": round(max_diff, 6),
        "val_metrics": train_info["val_metrics"],
        "loso_summary": {k: v for k, v in loso_summary.items() if k != "folds"} if loso_summary else {},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.perf_counter() - t0, 2),
    }

    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nExported ONNX classifier -> {output_onnx_path} (PyTorch/ONNX max diff={max_diff:.6f})")
    print(f"Saved config -> {output_config_path} (calibrated threshold={best_thr})")
    print(f"Validation F1={train_info['val_metrics']['f1']:.3f}, Accuracy={train_info['val_metrics']['accuracy']*100:.1f}%")
    return config


def main():
    parser = argparse.ArgumentParser(description="Train custom 'bot clip that' wake-word classifier.")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Path to data/wakeword/metadata.json")
    parser.add_argument("--augment-factor", type=int, default=8, help="Augmented variants per training clip.")
    parser.add_argument("--epochs", type=int, default=45, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip-loso", action="store_true", help="Skip Leave-One-Speaker-Out CV.")
    parser.add_argument("--no-public-negatives", action="store_true", help="Disable public negative speech background pool.")
    args = parser.parse_args()

    records = load_dataset_records(args.manifest)
    train_pipeline(
        records=records,
        public_negatives_path=None if args.no_public_negatives else PUBLIC_NEG_PATH,
        augment_factor=args.augment_factor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        run_loso=not args.skip_loso,
    )


if __name__ == "__main__":
    main()
