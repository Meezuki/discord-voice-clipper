import os
import time
import json
import wave
import numpy as np
import jiwer
from asr_engine import LocalASREngine

class WERBenchmarkEvaluator:
    """
    Quantitative evaluation module for academic assessment (COMP6822001 LO-2):
    Calculates Word Error Rate (WER), error breakdown (Substitutions, Deletions, Insertions),
    latency metrics, and Real-Time Factor (RTF).
    """
    def __init__(self, asr_engine=None):
        self.asr_engine = asr_engine or LocalASREngine(model_size="base", device="cpu", compute_type="int8")

    def evaluate_sample(self, audio_data: np.ndarray, reference_text: str, language: str = "en"):
        """
        Evaluates a single audio sample against reference ground truth text.
        """
        duration_sec = len(audio_data) / 16000.0 if len(audio_data) > 0 else 0.0
        hypothesis_text, latency = self.asr_engine.transcribe(audio_data, language=language)

        # Text normalization for fair comparison (lowercase, remove punctuation)
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip()
        ])

        ref_clean = transformation(reference_text)
        hyp_clean = transformation(hypothesis_text)

        output = jiwer.process_words(ref_clean, hyp_clean)
        wer = output.wer
        substitutions = output.substitutions
        deletions = output.deletions
        insertions = output.insertions
        hits = output.hits
        total_words = len(ref_clean.split()) if ref_clean.strip() else 0

        rtf = latency / duration_sec if duration_sec > 0 else 0.0

        return {
            "reference": reference_text,
            "hypothesis": hypothesis_text,
            "duration_sec": duration_sec,
            "latency_sec": latency,
            "rtf": rtf,
            "wer": wer,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "hits": hits,
            "total_words": total_words
        }

    def generate_markdown_report(self, results: list) -> str:
        """Generates a Markdown summary table for academic report documentation."""
        if not results:
            return "No evaluation data."

        total_words = sum(r["total_words"] for r in results)
        total_s = sum(r["substitutions"] for r in results)
        total_d = sum(r["deletions"] for r in results)
        total_i = sum(r["insertions"] for r in results)
        avg_wer = (total_s + total_d + total_i) / total_words if total_words > 0 else 0.0
        avg_latency = sum(r["latency_sec"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)

        md = "# Quantitative ASR Evaluation Report (Word Error Rate & Latency)\n\n"
        md += f"- **Total Test Samples:** {len(results)}\n"
        md += f"- **Average Word Error Rate (WER):** {avg_wer * 100:.2f}%\n"
        md += f"- **Average Inference Latency:** {avg_latency:.2f} seconds\n"
        md += f"- **Real-Time Factor (RTF):** {avg_rtf:.3f}x\n\n"
        md += "### Results Per Sample\n\n"
        md += "| No | Reference (Ground Truth) | Hypothesis (ASR Output) | Duration (s) | Latency (s) | WER (%) | S | D | I |\n"
        md += "|---|---|---|---|---|---|---|---|---|\n"

        for idx, r in enumerate(results, start=1):
            ref_short = (r["reference"][:30] + "...") if len(r["reference"]) > 30 else r["reference"]
            hyp_short = (r["hypothesis"][:30] + "...") if len(r["hypothesis"]) > 30 else r["hypothesis"]
            md += f"| {idx} | {ref_short} | {hyp_short} | {r['duration_sec']:.1f} | {r['latency_sec']:.2f} | {r['wer'] * 100:.1f}% | {r['substitutions']} | {r['deletions']} | {r['insertions']} |\n"

        return md

from scipy import signal

def load_wav_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Reads a WAV file, downmixes stereo to mono if needed,
    and automatically resamples to target_sr (default 16kHz) float32 normalized to [-1.0, 1.0].
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio_int16 = np.frombuffer(frames, dtype=np.int16)
    if n_channels == 2:
        audio_int16 = audio_int16.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample to 16 kHz if original sample rate differs (e.g. 44.1 kHz or 48 kHz)
    if framerate != target_sr and len(audio_int16) > 0:
        num_target_samples = int(round(len(audio_int16) * float(target_sr) / framerate))
        audio_resampled = signal.resample(audio_int16, num_target_samples)
        return audio_resampled.astype(np.float32) / 32768.0

    return audio_int16.astype(np.float32) / 32768.0

if __name__ == "__main__":
    from dataset_helper import DATASET_DIR, METADATA_FILE, init_dataset_structure, create_synthetic_audio_benchmarks

    if not os.path.exists(METADATA_FILE):
        print("Dataset metadata not found, generating initial baseline dataset...")
        create_synthetic_audio_benchmarks()

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    evaluator = WERBenchmarkEvaluator()
    all_results = []

    print(f"Beginning quantitative evaluation on {len(metadata)} samples...")
    for item in metadata:
        wav_path = os.path.join(DATASET_DIR, f"{item['id']}.wav")
        if os.path.exists(wav_path):
            audio = load_wav_file(wav_path)
            target_lang = item.get("language", "en")
            res = evaluator.evaluate_sample(audio, item["text"], language=target_lang)
            all_results.append(res)
            print(f"[{item['id']}] Latency: {res['latency_sec']:.2f}s | WER: {res['wer'] * 100:.1f}%")

    report_md = evaluator.generate_markdown_report(all_results)
    report_file = os.path.join(os.path.dirname(__file__), "evaluation_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nEvaluation complete! Markdown report exported to: {report_file}")

