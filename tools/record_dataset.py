"""
Interactive dataset recorder for the "bot clip that" wake-word model.
Records 16 kHz, 16-bit mono WAV clips organized by speaker_id and category,
and maintains data/wakeword/metadata.json for training and 5-fold LOSO evaluation.
"""

import os
import sys
import json
import time
import wave
import argparse
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from wakeword_features import SAMPLE_RATE, load_wav_int16_scale

DATASET_ROOT = os.path.join(ROOT_DIR, "data", "wakeword")
MANIFEST_PATH = os.path.join(DATASET_ROOT, "metadata.json")

CATEGORIES = {
    "positive_isolated": {
        "label": 1,
        "default_duration": 3.0,
        "description": "Isolated wake word ('bot clip that') with varied speed, tone, and mic distance.",
        "prompts": [
            "Say clearly at normal pace: 'bot clip that'",
            "Say quickly: 'bot clip that'",
            "Say slowly and clearly: 'bot clip that'",
            "Say softly / quiet voice: 'bot clip that'",
            "Say loudly / excited (like a gaming moment): 'bot clip that!'",
            "Say with rising intonation: 'bot clip that?'",
            "Say flatly like a voice command: 'bot clip that'",
            "Step ~40cm back from mic and say: 'bot clip that'",
            "Say casually as if wrapping up a thought: 'bot clip that'",
            "Say firmly with slight pause between words: 'bot... clip that'",
        ],
    },
    "positive_embedded": {
        "label": 1,
        "default_duration": 4.0,
        "description": "Wake word embedded naturally inside Discord meeting sentences.",
        "prompts": [
            "Say: 'Wait, bot clip that.'",
            "Say: 'Hold on, bot clip that right now.'",
            "Say: 'Okay that is important, bot clip that please.'",
            "Say: 'Hey bot clip that part.'",
            "Say: 'Nice point, bot clip that and let us move on.'",
            "Say: 'Wait wait, bot clip that before we forget.'",
            "Say: 'Eh bentar, bot clip that dong.'",
            "Say: 'Oke sip, bot clip that ya.'",
            "Say: 'That deadline changed, bot clip that.'",
            "Say: 'Actually yes, bot clip that.'",
        ],
    },
    "hard_negative": {
        "label": 0,
        "default_duration": 4.0,
        "description": "Confusable phrases containing 'bot', 'clip', or 'that' (must NOT trigger).",
        "prompts": [
            "Say: 'Pass me that paper clip please.'",
            "Say: 'Where did I put my hair clip?'",
            "Say: 'Use a binder clip for those pages.'",
            "Say: 'Can you flip that slide to the next one?'",
            "Say: 'Let us skip that section for now.'",
            "Say: 'Check that out in the repository.'",
            "Say: 'Got that, thank you very much.'",
            "Say: 'The Discord bot is still listening in voice.'",
            "Say: 'I bought a new microphone yesterday.'",
            "Say: 'Clip the video segment after the call.'",
            "Say: 'Put that link inside the text channel.'",
            "Say: 'Not that one, the other button.'",
        ],
    },
    "negative_speech_noise": {
        "label": 0,
        "default_duration": 5.0,
        "description": "Normal meeting speech (EN/ID) plus ambient mic noise, typing, and silence.",
        "prompts": [
            "Read: 'The weekly engineering coordination meeting starts right now.'",
            "Read: 'We encountered several technical challenges during server load testing.'",
            "Read: 'Please ensure all team members document the agreed discussion points.'",
            "Read: 'Are there any additional questions before we conclude today?'",
            "Speak naturally in Indonesian/English about today's progress for 3 seconds.",
            "Speak casually as if chatting with teammates on Discord.",
            "Type on your keyboard and click your mouse without speaking.",
            "Stay silent for ambient room / fan / AC background noise.",
            "Clear your throat or breathe normally near the mic.",
            "Hum or play faint background noise without saying the wake word.",
        ],
    },
}


def load_manifest(manifest_path: str = MANIFEST_PATH) -> list[dict]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(entries: list[dict], manifest_path: str = MANIFEST_PATH) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, manifest_path)


def save_wav_16k_mono(file_path: str, audio_int16: np.ndarray) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.astype(np.int16).tobytes())


def rebuild_manifest(dataset_root: str = DATASET_ROOT, manifest_path: str = MANIFEST_PATH) -> list[dict]:
    """
    Scans data/wakeword/<speaker_id>/<category>/*.wav and rebuilds metadata.json.
    Useful when merging recordings from multiple group members' machines.
    """
    entries = []
    if not os.path.exists(dataset_root):
        save_manifest(entries, manifest_path)
        return entries

    existing_by_path = {
        e["path"].replace("\\", "/"): e for e in load_manifest(manifest_path) if "path" in e
    }

    for speaker_id in sorted(os.listdir(dataset_root)):
        speaker_dir = os.path.join(dataset_root, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        for category in sorted(os.listdir(speaker_dir)):
            cat_dir = os.path.join(speaker_dir, category)
            if not os.path.isdir(cat_dir) or category not in CATEGORIES:
                continue
            label = CATEGORIES[category]["label"]
            for fname in sorted(os.listdir(cat_dir)):
                if not fname.lower().endswith(".wav"):
                    continue
                full_path = os.path.join(cat_dir, fname)
                rel_path = os.path.relpath(full_path, ROOT_DIR).replace("\\", "/")
                prev = existing_by_path.get(rel_path, {})
                wav = load_wav_int16_scale(full_path)
                norm = wav / 32767.0
                rms = float(np.sqrt(np.mean(norm ** 2))) if len(norm) > 0 else 0.0
                entries.append(
                    {
                        "id": os.path.splitext(fname)[0],
                        "path": rel_path,
                        "speaker_id": speaker_id,
                        "category": category,
                        "label": label,
                        "prompt": prev.get("prompt", ""),
                        "duration_sec": round(len(wav) / SAMPLE_RATE, 3),
                        "sample_rate": SAMPLE_RATE,
                        "rms": round(rms, 5),
                    }
                )

    save_manifest(entries, manifest_path)
    return entries


def record_clip(duration_sec: float, device: int | None = None) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice is not installed. Run: pip install sounddevice")
    num_samples = int(round(duration_sec * SAMPLE_RATE))
    recording = sd.rec(
        num_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return recording.reshape(-1)


def print_dataset_summary(entries: list[dict]) -> None:
    if not entries:
        print("Dataset is currently empty.")
        return
    by_speaker = {}
    for e in entries:
        spk = e["speaker_id"]
        cat = e["category"]
        by_speaker.setdefault(spk, {}).setdefault(cat, 0)
        by_speaker[spk][cat] += 1

    print(f"\n=== Dataset Summary ({len(entries)} total clips) ===")
    for spk, cats in sorted(by_speaker.items()):
        pos = cats.get("positive_isolated", 0) + cats.get("positive_embedded", 0)
        neg = cats.get("hard_negative", 0) + cats.get("negative_speech_noise", 0)
        print(f"  Speaker '{spk}': {pos} positive, {neg} negative -> {dict(cats)}")


def _count_existing_takes(entries: list[dict], speaker_id: str) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for e in entries:
        if e.get("speaker_id") != speaker_id:
            continue
        rel_path = e.get("path", "")
        full_path = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT_DIR, rel_path)
        if os.path.exists(full_path):
            key = (e.get("category", ""), e.get("prompt", ""))
            counts[key] = counts.get(key, 0) + 1
    return counts


def interactive_session(
    speaker_id: str,
    selected_categories: list[str],
    takes_per_prompt: int,
    duration_override: float | None,
    device: int | None,
    resume: bool = True,
) -> None:
    entries = load_manifest()
    existing_counts = _count_existing_takes(entries, speaker_id) if resume else {}
    print(f"\nRecording session for speaker: '{speaker_id}'")
    print("Controls after each take: [Enter]=Keep  p=Play back  r=Redo  s=Skip prompt  q=Quit\n")

    for cat in selected_categories:
        cfg = CATEGORIES[cat]
        dur = duration_override or cfg["default_duration"]
        print(f"\n--- Category: {cat} (label={cfg['label']}, {dur:.1f}s/clip) ---")
        print(f"Goal: {cfg['description']}")

        for p_idx, prompt in enumerate(cfg["prompts"], start=1):
            done_takes = existing_counts.get((cat, prompt), 0) if resume else 0
            if done_takes >= takes_per_prompt:
                print(f"  [Skipping completed prompt {p_idx}/{len(cfg['prompts'])}: {done_takes}/{takes_per_prompt}]")
                continue

            for take in range(done_takes + 1, takes_per_prompt + 1):
                while True:
                    input(
                        f"\n[{cat} | prompt {p_idx}/{len(cfg['prompts'])} | take {take}/{takes_per_prompt}]\n"
                        f"  >> {prompt}\n"
                        f"  Press [Enter] to record ({dur:.1f}s)..."
                    )
                    print("  Recording NOW...", flush=True)
                    audio_int16 = record_clip(dur, device=device)
                    norm = audio_int16.astype(np.float32) / 32767.0
                    rms = float(np.sqrt(np.mean(norm ** 2)))
                    peak = float(np.max(np.abs(norm)))

                    warn = ""
                    if peak > 0.98:
                        warn = " [WARNING: CLIPPING - move slightly away from mic]"
                    elif rms < 0.005 and cat != "negative_speech_noise":
                        warn = " [WARNING: VERY QUIET - speak louder or check mic]"

                    print(f"  Done! RMS={rms:.4f}, Peak={peak:.3f}{warn}")
                    while True:
                        choice = input("  [Enter]=Keep / p=Play / r=Redo / s=Skip / q=Quit: ").strip().lower()
                        if choice == "p":
                            print("  Playing back...", flush=True)
                            sd.play(audio_int16, SAMPLE_RATE)
                            sd.wait()
                            continue
                        break

                    if choice == "r":
                        continue
                    if choice == "s":
                        break
                    if choice == "q":
                        save_manifest(entries)
                        print_dataset_summary(entries)
                        return

                    ts = int(time.time() * 1000)
                    fname = f"{speaker_id}_{cat}_{ts}_{p_idx:02d}_{take:02d}.wav"
                    out_path = os.path.join(DATASET_ROOT, speaker_id, cat, fname)
                    save_wav_16k_mono(out_path, audio_int16)
                    print(f"  Saved: {os.path.relpath(out_path, ROOT_DIR)}")

                    rel_path = os.path.relpath(out_path, ROOT_DIR).replace("\\", "/")
                    entries.append(
                        {
                            "id": os.path.splitext(fname)[0],
                            "path": rel_path,
                            "speaker_id": speaker_id,
                            "category": cat,
                            "label": cfg["label"],
                            "prompt": prompt,
                            "duration_sec": round(len(audio_int16) / SAMPLE_RATE, 3),
                            "sample_rate": SAMPLE_RATE,
                            "rms": round(rms, 5),
                        }
                    )
                    save_manifest(entries)
                    break
                if choice == "s":
                    break

    print("\nSession complete!")
    print_dataset_summary(entries)


def main():
    parser = argparse.ArgumentParser(description="Interactive 16kHz mono wake-word dataset recorder.")
    parser.add_argument("--speaker", type=str, help="Speaker identifier (e.g. member1, marco).")
    parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORIES.keys()) + ["all"],
        default="all",
        help="Which prompt category to record.",
    )
    parser.add_argument("--takes", type=int, default=2, help="Number of takes per prompt (default: 2).")
    parser.add_argument("--duration", type=float, default=None, help="Override recording duration in seconds.")
    parser.add_argument("--device", type=int, default=None, help="Input audio device index.")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit.")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Rebuild metadata.json from WAV files on disk.")
    parser.add_argument("--summary", action="store_true", help="Print current dataset summary and exit.")
    parser.add_argument("--no-resume", action="store_true", help="Record all prompts from the beginning instead of skipping completed ones.")
    args = parser.parse_args()

    if args.list_devices:
        if sd is None:
            print("sounddevice not installed.")
            return
        print(sd.query_devices())
        return

    if args.rebuild_manifest:
        entries = rebuild_manifest()
        print(f"Rebuilt manifest at {MANIFEST_PATH}")
        print_dataset_summary(entries)
        return

    if args.summary:
        print_dataset_summary(load_manifest())
        return

    speaker_id = args.speaker or input("Enter speaker_id (e.g. member1): ").strip()
    if not speaker_id:
        print("speaker_id is required.")
        return
    speaker_id = "".join(c if (c.isalnum() or c in "_-") else "_" for c in speaker_id.lower())

    cats = list(CATEGORIES.keys()) if args.category == "all" else [args.category]
    interactive_session(
        speaker_id=speaker_id,
        selected_categories=cats,
        takes_per_prompt=max(1, args.takes),
        duration_override=args.duration,
        device=args.device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
