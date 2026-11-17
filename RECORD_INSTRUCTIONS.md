# Wake-Word Dataset Recording Instructions (`"bot clip that"`)

Follow these steps to record your voice samples for the 5-person group dataset (`16 kHz` mono WAV) so we can train a speaker-independent wake-word model and run 5-Fold Leave-One-Speaker-Out (LOSO) evaluation.

---

## 1. Clone & Set Up Environment (Windows PowerShell)

```powershell
git clone https://github.com/Meezuki/discord-voice-clipper.git
cd discord-voice-clipper
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the lightweight dependencies needed for recording (~15 seconds):

```powershell
pip install numpy scipy onnxruntime sounddevice
```

*(Optional: run `pip install -r requirements.txt` instead if you also plan to train the model or run the Discord bot locally.)*

---

## 2. Check Your Microphone (Optional)

If you have multiple microphones (e.g., laptop mic vs. Discord headset), list available input devices first:

```powershell
python tools\record_dataset.py --list-devices
```

Note the device index number of your main Discord mic if Windows is not using it by default.

---

## 3. Record Your Voice Samples

Replace `<your_name>` with your first name in lowercase (for example, `alex`):

```powershell
python tools\record_dataset.py --speaker <your_name> --takes 2
```

* Pass `--device <id>` if you need to select a specific microphone from Step 2.
* Pass `--takes 1` if you only have ~3 minutes (records 1 take per prompt instead of 2).

### Controls During Recording
1. Press **`[Enter]`** when ready $\rightarrow$ as soon as `Recording NOW...` prints, speak the prompt once within the `3–5s` window.
2. After `Done! RMS=..., Peak=...` prints, review your take:
   * **`p`** + `[Enter]` — **Play back** the take you just recorded
   * **`r`** + `[Enter]` — **Redo** the take (use this if you stumbled, spoke too late, or saw a `CLIPPING` / `VERY QUIET` warning)
   * **`[Enter]`** — **Keep & Save** to `data/wakeword/<your_name>/<category>/`
   * **`s`** + `[Enter]` — **Skip** the current prompt
   * **`q`** + `[Enter]` — **Quit** safely (all kept takes remain saved; re-running the same command later automatically resumes where you left off)

---

## 4. How to Delete an Accidental Save (Pressed `[Enter]` Too Early)

If you accidentally pressed `[Enter]` to keep an empty or bad take:

1. Check the `Saved: data\wakeword\<your_name>\...wav` path printed right above (or open that category folder in File Explorer and sort by **Date modified** to find the latest `.wav`), and **delete that `.wav` file**.
2. Rebuild the dataset manifest:
   ```powershell
   python tools\record_dataset.py --rebuild-manifest
   ```
3. Re-run the recorder:
   ```powershell
   python tools\record_dataset.py --speaker <your_name> --takes 2
   ```
   It will automatically skip completed prompts and jump back to the deleted take so you can re-record it.

---

## 5. Send Your Recordings to Marco

1. Verify your recorded clip counts:
   ```powershell
   python tools\record_dataset.py --summary
   ```
2. Because `*.wav` files are excluded in `.gitignore`, **compress your `data\wakeword\<your_name>\` folder into a `.zip` file** and send it to Marco (via Discord or Google Drive).

---

## 6. Merging, Training & Evaluation (After Collecting All 5 Members)

Once all teammates' folders are extracted into `data\wakeword\<name>\`:

```powershell
.\venv\Scripts\python.exe tools\record_dataset.py --rebuild-manifest
.\venv\Scripts\python.exe train_wakeword.py
.\venv\Scripts\python.exe evaluate_wakeword.py
```

This rebuilds the unified manifest, runs 5-Fold Leave-One-Speaker-Out (LOSO) cross-validation (`models/wakeword/loso_results.json`), exports `models/wakeword/bot_clip_that.onnx`, and regenerates `evaluation_wakeword_report.md`.
