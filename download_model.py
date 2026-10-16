import os
import zipfile
import requests

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
EXTRACTED_PATH = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")
ZIP_PATH = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15.zip")

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
    ensure_vosk_model()

