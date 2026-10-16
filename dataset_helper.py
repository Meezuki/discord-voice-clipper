import os
import json
import wave
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "test_dataset")
METADATA_FILE = os.path.join(DATASET_DIR, "dataset_metadata.json")

# 10 English meeting benchmark sentences for LO-2 evaluation
SAMPLE_MEETING_TRANSCRIPTS = [
    {"id": "sample_01", "text": "the weekly engineering coordination meeting starts right now"},
    {"id": "sample_02", "text": "the project budget for this quarter has been approved by management"},
    {"id": "sample_03", "text": "we encountered several technical challenges during server load testing"},
    {"id": "sample_04", "text": "the product release schedule will be postponed for three business days"},
    {"id": "sample_05", "text": "the user interface design review has been completed with the client"},
    {"id": "sample_06", "text": "please ensure all team members document the agreed discussion points"},
    {"id": "sample_07", "text": "the artificial intelligence system runs completely offline on local machines"},
    {"id": "sample_08", "text": "the thirty second rolling audio buffer was frozen and saved successfully"},
    {"id": "sample_09", "text": "are there any additional questions before we conclude today's session"},
    {"id": "sample_10", "text": "thank you everyone for your attendance and participation in this meeting"}
]

def init_dataset_structure():
    """
    Initializes dataset directory and ground truth metadata file.
    Users can place real benchmark .wav files into data/test_dataset/.
    """
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_MEETING_TRANSCRIPTS, f, indent=2, ensure_ascii=False)
    print(f"Dataset directory and ground truth reference initialized at: {DATASET_DIR}")
    print(f"Metadata file: {METADATA_FILE}")

def create_synthetic_audio_benchmarks():
    """
    Creates synthetic/dummy WAV audio files (16-bit 16kHz mono)
    as placeholder validation files so evaluate_wer.py can be tested immediately.
    """
    init_dataset_structure()
    for item in SAMPLE_MEETING_TRANSCRIPTS:
        wav_path = os.path.join(DATASET_DIR, f"{item['id']}.wav")
        if not os.path.exists(wav_path):
            # 3.0-second sine wave + subtle white noise simulating mic audio
            duration = 3.0
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            sine = np.sin(2 * np.pi * 440 * t) * 0.1
            noise = np.random.normal(0, 0.005, len(t))
            audio_float = sine + noise
            audio_int16 = (audio_float * 32767).astype(np.int16)

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
            print(f"Created synthetic benchmark sample: {wav_path}")

if __name__ == "__main__":
    create_synthetic_audio_benchmarks()

