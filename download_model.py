import os
import zipfile
import requests

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
WAKEWORD_MODEL_DIR = os.path.join(MODEL_DIR, "wakeword")

MEL_MODEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx"
EMB_MODEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
PUBLIC_NEG_URL = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"

MEL_MODEL_PATH = os.path.join(WAKEWORD_MODEL_DIR, "melspectrogram.onnx")
EMB_MODEL_PATH = os.path.join(WAKEWORD_MODEL_DIR, "embedding_model.onnx")
PUBLIC_NEG_PATH = os.path.join(WAKEWORD_MODEL_DIR, "public_negative_features.npy")
CLASSIFIER_MODEL_PATH = os.path.join(WAKEWORD_MODEL_DIR, "bot_clip_that.onnx")
CLASSIFIER_CONFIG_PATH = os.path.join(WAKEWORD_MODEL_DIR, "model_config.json")

# Legacy Vosk constants kept only for backward reference
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
EXTRACTED_PATH = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")
ZIP_PATH = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15.zip")


def _download_file(url: str, dest_path: str) -> None:
    print(f"Downloading {os.path.basename(dest_path)} from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, dest_path)
    print(f"Saved to: {dest_path}")


def ensure_wakeword_backbone() -> tuple[str, str]:
    """
    Ensures the frozen openWakeWord ONNX speech-embedding backbone
    (melspectrogram.onnx and embedding_model.onnx) is available locally.
    """
    os.makedirs(WAKEWORD_MODEL_DIR, exist_ok=True)
    if not os.path.exists(MEL_MODEL_PATH):
        _download_file(MEL_MODEL_URL, MEL_MODEL_PATH)
    if not os.path.exists(EMB_MODEL_PATH):
        _download_file(EMB_MODEL_URL, EMB_MODEL_PATH)
    return MEL_MODEL_PATH, EMB_MODEL_PATH


def ensure_public_negative_features() -> str:
    """Downloads precomputed open-set negative speech/music/noise embeddings (~184 MB)."""
    os.makedirs(WAKEWORD_MODEL_DIR, exist_ok=True)
    if not os.path.exists(PUBLIC_NEG_PATH):
        _download_file(PUBLIC_NEG_URL, PUBLIC_NEG_PATH)
    return PUBLIC_NEG_PATH


def ensure_vosk_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(EXTRACTED_PATH):
        print(f"Vosk model is already available at: {EXTRACTED_PATH}")
        return EXTRACTED_PATH

    print(f"Downloading Vosk English model from {MODEL_URL} ...")
    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Extracting model archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"Model successfully extracted to: {EXTRACTED_PATH}")
    return EXTRACTED_PATH


if __name__ == "__main__":
    mel_path, emb_path = ensure_wakeword_backbone()
    print(f"Wake-word backbone ready:\n  - {mel_path}\n  - {emb_path}")
