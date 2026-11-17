"""
Desktop Launcher GUI for Discord Voice Clipper (built-in tkinter + ttk).
Provides non-destructive .env credential/settings management, clean Start/Stop
process-tree lifecycle control, live status tracking, and clip/transcription logs.
"""

import os
import re
import sys
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
MAIN_SCRIPT = os.path.join(ROOT_DIR, "main.py")

# Dark-slate palette
BG_MAIN = "#1E1F22"
BG_PANEL = "#2B2D31"
BG_INPUT = "#18191C"
BORDER = "#383A40"
ACCENT = "#5865F2"
ACCENT_HOVER = "#4752C4"
DANGER = "#DA373C"
DANGER_HOVER = "#B82B30"
TEXT_PRIMARY = "#F2F3F5"
TEXT_SECONDARY = "#B5BAC1"
TEXT_MUTED = "#80848E"

STATUS_STYLES = {
    "Stopped": ("#4E5058", "#F2F3F5"),
    "Connecting...": ("#F0B232", "#18191C"),
    "Online (Idle)": ("#5865F2", "#F2F3F5"),
    "Listening in VC": ("#23A55A", "#F2F3F5"),
    "Transcribing...": ("#F0B232", "#18191C"),
    "Stopping...": ("#DA373C", "#F2F3F5"),
}

MIN_DURATION_SEC = 10
MAX_DURATION_SEC = 3600

CLIP_EVENT_RE = re.compile(
    r"\[CLIP_EVENT\]\s+time=(?P<time>[^|]+?)\s*\|\s*trigger=(?P<trigger>[^|]+?)\s*\|\s*"
    r"latency=(?P<latency>[^|]+?)\s*\|\s*dest=(?P<dest>[^|]+?)\s*\|\s*text=(?P<text>.*)$"
)


def parse_duration_choice(raw: str) -> int:
    """Parses duration strings ('30s', '1m', '5m', '30m', '120') into bounded seconds."""
    val = (raw or "").strip().lower()
    if not val:
        raise ValueError("Duration cannot be empty.")
    multiplier = 1
    if val.endswith("m"):
        multiplier = 60
        val = val[:-1].strip()
    elif val.endswith("s"):
        val = val[:-1].strip()
    if not val.isdigit():
        raise ValueError(f"Invalid duration '{raw}'. Use e.g. 30s, 5m, 30m, or seconds.")
    sec = int(val) * multiplier
    if not (MIN_DURATION_SEC <= sec <= MAX_DURATION_SEC):
        raise ValueError(f"Duration must be between {MIN_DURATION_SEC}s and {MAX_DURATION_SEC}s (60m).")
    return sec


def format_duration_choice(seconds: int) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def load_env_settings(env_path: str = ENV_PATH) -> dict:
    """Reads GUI-managed keys from .env with safe defaults."""
    settings = {
        "DISCORD_TOKEN": "",
        "DEFAULT_BUFFER_SECONDS": "30",
        "DEFAULT_LANGUAGE": "en",
        "SAVE_LOCALLY": "false",
    }
    if not os.path.exists(env_path):
        return settings

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in settings:
                settings[key] = val
    return settings


def save_env_settings(env_path: str, updates: dict[str, str]) -> None:
    """
    Updates specified keys in .env while preserving all unrelated keys,
    comments, and blank lines.
    """
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                new_lines.append(f"{key}={remaining.pop(key)}")
                continue
        new_lines.append(line)

    for key, val in remaining.items():
        new_lines.append(f"{key}={val}")

    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")


def parse_bot_log_line(line: str) -> dict:
    """
    Extracts live bot status transitions and structured [CLIP_EVENT] payloads
    from main.py stdout/stderr lines.
    """
    result = {"status": None, "clip": None}
    match = CLIP_EVENT_RE.search(line)
    if match:
        result["clip"] = {
            "time": match.group("time").strip(),
            "trigger": match.group("trigger").strip(),
            "latency": match.group("latency").strip(),
            "dest": match.group("dest").strip(),
            "text": match.group("text").strip(),
        }
        result["status"] = "Listening in VC"
        return result

    if "Transcribing " in line and "clip (" in line:
        result["status"] = "Transcribing..."
    elif "Voice Channel:" in line and "DAVE Version:" in line:
        result["status"] = "Listening in VC"
    elif "Disconnected from voice channel" in line or "Bot connected as " in line:
        result["status"] = "Online (Idle)"
    elif "Voice clip dispatched to " in line:
        result["status"] = "Listening in VC"

    return result


def terminate_process_tree(proc: subprocess.Popen | None, graceful_timeout: float = 3.0) -> None:
    """
    Requests graceful Discord shutdown via stdin ('STOP'), waits briefly so
    the bot leaves VC/gateway cleanly, then force-kills any remaining process tree.
    """
    if proc is None or proc.poll() is not None:
        return

    # 1. Try graceful shutdown so Discord VC/gateway disconnects immediately
    if proc.stdin:
        try:
            proc.stdin.write("STOP\n")
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=graceful_timeout)
            return
        except subprocess.TimeoutExpired:
            pass

    # 2. Fallback: force-terminate entire process tree on Windows
    if sys.platform.startswith("win"):
        try:
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
                timeout=5,
                check=False,
            )
            return
        except Exception:
            pass
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _resolve_python_executable() -> str:
    venv_py = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


class DiscordClipperGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Discord Voice Clipper — Local KWS & ASR Launcher")
        self.root.geometry("980x660")
        self.root.minsize(920, 620)
        self.root.configure(bg=BG_MAIN)

        self.proc: subprocess.Popen | None = None
        self.is_stopping = False
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.clip_history: list[dict] = []
        self.token_visible = False

        self._setup_styles()
        self._build_ui()
        self._load_initial_settings()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_queue)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("Main.TFrame", background=BG_MAIN)
        style.configure("Card.TFrame", background=BG_PANEL)
        style.configure("Header.TLabel", background=BG_MAIN, foreground=TEXT_PRIMARY, font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", background=BG_MAIN, foreground=TEXT_MUTED, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=BG_PANEL, foreground=TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", background=BG_PANEL, foreground=TEXT_SECONDARY, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=BG_PANEL, foreground=TEXT_MUTED, font=("Segoe UI", 8))
        style.configure("Mono.TLabel", background=BG_PANEL, foreground=TEXT_SECONDARY, font=("Consolas", 9))

        style.configure(
            "Dark.TEntry",
            fieldbackground=BG_INPUT,
            foreground=TEXT_PRIMARY,
            insertcolor=TEXT_PRIMARY,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=5,
        )
        style.configure(
            "Dark.TCombobox",
            fieldbackground=BG_INPUT,
            background=BG_PANEL,
            foreground=TEXT_PRIMARY,
            arrowcolor=TEXT_SECONDARY,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=4,
        )
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", BG_INPUT)],
            foreground=[("readonly", TEXT_PRIMARY)],
            selectbackground=[("readonly", ACCENT)],
            selectforeground=[("readonly", TEXT_PRIMARY)],
        )
        style.configure(
            "Dark.TCheckbutton",
            background=BG_PANEL,
            foreground=TEXT_SECONDARY,
            focuscolor=BG_PANEL,
            font=("Segoe UI", 9),
        )
        style.map("Dark.TCheckbutton", background=[("active", BG_PANEL)], foreground=[("active", TEXT_PRIMARY)])

        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground=TEXT_PRIMARY,
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 9, "bold"),
            padding=(12, 7),
        )
        style.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#3B3E45")], foreground=[("disabled", TEXT_MUTED)])

        style.configure(
            "Danger.TButton",
            background=DANGER,
            foreground=TEXT_PRIMARY,
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 9, "bold"),
            padding=(12, 7),
        )
        style.map("Danger.TButton", background=[("active", DANGER_HOVER), ("disabled", "#3B3E45")], foreground=[("disabled", TEXT_MUTED)])

        style.configure(
            "Secondary.TButton",
            background="#383A40",
            foreground=TEXT_PRIMARY,
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 8, "bold"),
            padding=(8, 5),
        )
        style.map("Secondary.TButton", background=[("active", "#474A52")])

    def _make_card(self, parent, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        inner = tk.Frame(outer, bg=BG_PANEL, padx=12, pady=10)
        inner.pack(fill="both", expand=True)
        ttk.Label(inner, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
        return outer, inner

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg=BG_MAIN, padx=16, pady=16)
        container.pack(fill="both", expand=True)

        # 1. Top Header Bar
        header = tk.Frame(container, bg=BG_MAIN)
        header.pack(fill="x", pady=(0, 12))

        title_col = tk.Frame(header, bg=BG_MAIN)
        title_col.pack(side="left")
        ttk.Label(title_col, text="Discord Voice Clipper", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            title_col,
            text="100% Local Wake-Word ('bot clip that') + Faster-Whisper ASR",
            style="SubHeader.TLabel",
        ).pack(anchor="w")

        controls = tk.Frame(header, bg=BG_MAIN)
        controls.pack(side="right")

        self.status_pill = tk.Label(
            controls,
            text="Stopped",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=5,
            bg=STATUS_STYLES["Stopped"][0],
            fg=STATUS_STYLES["Stopped"][1],
        )
        self.status_pill.pack(side="left", padx=(0, 10))

        self.start_btn = ttk.Button(controls, text="Start Bot", style="Accent.TButton", command=self.start_bot)
        self.start_btn.pack(side="left", padx=(0, 6))

        self.stop_btn = ttk.Button(
            controls, text="Stop Bot", style="Danger.TButton", command=self.stop_bot, state="disabled"
        )
        self.stop_btn.pack(side="left")

        # 2. Main Body (Fixed 320px Left Sidebar + Expanding Right Workspace)
        body = tk.Frame(container, bg=BG_MAIN)
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, bg=BG_MAIN, width=320)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)

        workspace = tk.Frame(body, bg=BG_MAIN)
        workspace.pack(side="left", fill="both", expand=True)

        # --- Left Card 1: Bot Credentials ---
        cred_outer, cred = self._make_card(sidebar, "Bot Credentials")
        cred_outer.pack(fill="x", pady=(0, 12))

        ttk.Label(cred, text="DISCORD_TOKEN", style="Field.TLabel").pack(anchor="w", pady=(0, 4))
        token_row = tk.Frame(cred, bg=BG_PANEL)
        token_row.pack(fill="x")

        self.token_var = tk.StringVar()
        self.token_entry = ttk.Entry(token_row, textvariable=self.token_var, show="*", style="Dark.TEntry")
        self.token_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.toggle_token_btn = ttk.Button(
            token_row, text="Show", style="Secondary.TButton", width=6, command=self._toggle_token_visibility
        )
        self.toggle_token_btn.pack(side="right")

        # --- Left Card 2: Capture Settings ---
        cfg_outer, cfg = self._make_card(sidebar, "Capture Settings")
        cfg_outer.pack(fill="x", pady=(0, 12))

        ttk.Label(cfg, text="Default Buffer Duration", style="Field.TLabel").pack(anchor="w", pady=(0, 4))
        self.duration_var = tk.StringVar(value="30s")
        self.duration_combo = ttk.Combobox(
            cfg,
            textvariable=self.duration_var,
            values=["30s", "1m", "5m", "30m"],
            style="Dark.TCombobox",
        )
        self.duration_combo.pack(fill="x")
        ttk.Label(
            cfg,
            text=">30s saves locally only (skips Discord WAV upload)",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor="w", pady=(3, 10))

        ttk.Label(cfg, text="Default Transcription Language", style="Field.TLabel").pack(anchor="w", pady=(0, 4))
        self.lang_var = tk.StringVar(value="en")
        self.lang_combo = ttk.Combobox(
            cfg,
            textvariable=self.lang_var,
            values=["en", "id", "auto"],
            state="readonly",
            style="Dark.TCombobox",
        )
        self.lang_combo.pack(fill="x", pady=(0, 10))

        self.save_local_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cfg,
            text="Also save <=30s clips locally (recordings/)",
            variable=self.save_local_var,
            style="Dark.TCheckbutton",
        ).pack(anchor="w", pady=(0, 12))

        self.save_btn = ttk.Button(cfg, text="Save Settings", style="Accent.TButton", command=self.save_settings)
        self.save_btn.pack(fill="x")

        # --- Left Card 3: Runtime Engine ---
        eng_outer, eng = self._make_card(sidebar, "Runtime Engine")
        eng_outer.pack(fill="x")
        ttk.Label(eng, text="KWS: WakeWordBiGRU (ONNX)", style="Mono.TLabel").pack(anchor="w", pady=(0, 3))
        ttk.Label(eng, text="ASR: Faster-Whisper (int8)", style="Mono.TLabel").pack(anchor="w", pady=(0, 3))
        ttk.Label(eng, text="Wake Word: 'bot clip that'", style="Muted.TLabel").pack(anchor="w")

        # --- Right Card 1: Latest Captured Clip ---
        clip_outer, clip_card = self._make_card(workspace, "Latest Captured Clip")
        clip_outer.pack(fill="x", pady=(0, 12))

        meta_row = tk.Frame(clip_card, bg=BG_PANEL)
        meta_row.pack(fill="x", pady=(0, 6))

        self.clip_trigger_lbl = tk.Label(
            meta_row, text="Trigger: Waiting for clip...", bg=BG_PANEL, fg=TEXT_SECONDARY, font=("Segoe UI", 9, "bold")
        )
        self.clip_trigger_lbl.pack(side="left")

        self.clip_time_lbl = tk.Label(
            meta_row, text="", bg=BG_PANEL, fg=TEXT_MUTED, font=("Consolas", 8)
        )
        self.clip_time_lbl.pack(side="right")

        self.clip_latency_badge = tk.Label(
            meta_row, text="Latency: --", bg=BG_INPUT, fg=TEXT_SECONDARY, font=("Consolas", 8, "bold"), padx=6, pady=2
        )
        self.clip_latency_badge.pack(side="right", padx=(0, 8))

        self.clip_text_box = tk.Text(
            clip_card,
            height=4,
            wrap="word",
            bg=BG_INPUT,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10),
            relief="flat",
            padx=8,
            pady=6,
            state="disabled",
        )
        self.clip_text_box.pack(fill="x", pady=(0, 6))

        self.clip_dest_lbl = tk.Label(
            clip_card, text="Output: --", bg=BG_PANEL, fg=TEXT_MUTED, font=("Consolas", 8), anchor="w"
        )
        self.clip_dest_lbl.pack(fill="x")

        # --- Right Card 2: Session Clip History ---
        hist_outer, hist_card = self._make_card(workspace, "Session Clip History")
        hist_outer.pack(fill="x", pady=(0, 12))

        hist_wrap = tk.Frame(hist_card, bg=BG_INPUT)
        hist_wrap.pack(fill="x")

        self.history_list = tk.Listbox(
            hist_wrap,
            height=4,
            bg=BG_INPUT,
            fg=TEXT_SECONDARY,
            selectbackground=ACCENT,
            selectforeground=TEXT_PRIMARY,
            font=("Consolas", 9),
            relief="flat",
            highlightthickness=0,
            activestyle="none",
        )
        hist_scroll = ttk.Scrollbar(hist_wrap, orient="vertical", command=self.history_list.yview)
        self.history_list.configure(yscrollcommand=hist_scroll.set)
        self.history_list.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        hist_scroll.pack(side="right", fill="y")
        self.history_list.bind("<<ListboxSelect>>", self._on_history_select)

        # --- Right Card 3: System Log ---
        log_outer = tk.Frame(workspace, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        log_outer.pack(fill="both", expand=True)
        log_inner = tk.Frame(log_outer, bg=BG_PANEL, padx=12, pady=10)
        log_inner.pack(fill="both", expand=True)

        log_hdr = tk.Frame(log_inner, bg=BG_PANEL)
        log_hdr.pack(fill="x", pady=(0, 6))
        ttk.Label(log_hdr, text="System Log", style="CardTitle.TLabel").pack(side="left")
        ttk.Button(log_hdr, text="Clear", style="Secondary.TButton", command=self._clear_logs).pack(side="right")

        log_wrap = tk.Frame(log_inner, bg=BG_INPUT)
        log_wrap.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_wrap,
            wrap="word",
            bg=BG_INPUT,
            fg=TEXT_SECONDARY,
            font=("Consolas", 8),
            relief="flat",
            padx=8,
            pady=6,
            state="disabled",
        )
        log_scroll = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    def _load_initial_settings(self) -> None:
        cfg = load_env_settings(ENV_PATH)
        self.token_var.set(cfg["DISCORD_TOKEN"])
        try:
            sec = parse_duration_choice(cfg["DEFAULT_BUFFER_SECONDS"])
            self.duration_var.set(format_duration_choice(sec))
        except ValueError:
            self.duration_var.set("30s")
        self.lang_var.set(cfg["DEFAULT_LANGUAGE"] if cfg["DEFAULT_LANGUAGE"] in {"en", "id", "auto"} else "en")
        self.save_local_var.set(cfg["SAVE_LOCALLY"].lower() in {"1", "true", "yes", "on"})

    def _toggle_token_visibility(self) -> None:
        self.token_visible = not self.token_visible
        self.token_entry.configure(show="" if self.token_visible else "*")
        self.toggle_token_btn.configure(text="Hide" if self.token_visible else "Show")

    def _set_status(self, status: str) -> None:
        bg, fg = STATUS_STYLES.get(status, STATUS_STYLES["Stopped"])
        self.status_pill.configure(text=status, bg=bg, fg=fg)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_logs(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _show_clip(self, clip: dict) -> None:
        self.clip_trigger_lbl.configure(text=f"Trigger: {clip['trigger']}")
        self.clip_latency_badge.configure(text=f"Latency: {clip['latency']}")
        self.clip_time_lbl.configure(text=clip["time"])
        self.clip_dest_lbl.configure(text=f"Output: {clip['dest']}")

        self.clip_text_box.configure(state="normal")
        self.clip_text_box.delete("1.0", "end")
        self.clip_text_box.insert("1.0", clip["text"])
        self.clip_text_box.configure(state="disabled")

    def _on_history_select(self, _event=None) -> None:
        sel = self.history_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.clip_history):
            self._show_clip(self.clip_history[idx])

    def save_settings(self, silent: bool = False) -> bool:
        token = self.token_var.get().strip()
        try:
            dur_sec = parse_duration_choice(self.duration_var.get())
        except ValueError as e:
            messagebox.showerror("Invalid Duration", str(e))
            return False

        lang = self.lang_var.get().strip().lower()
        if lang not in {"en", "id", "auto"}:
            lang = "en"

        updates = {
            "DISCORD_TOKEN": token,
            "DEFAULT_BUFFER_SECONDS": str(dur_sec),
            "DEFAULT_LANGUAGE": lang,
            "SAVE_LOCALLY": "true" if self.save_local_var.get() else "false",
        }
        save_env_settings(ENV_PATH, updates)
        self.duration_var.set(format_duration_choice(dur_sec))
        self._append_log(f"[GUI] Saved .env settings (duration={dur_sec}s, lang={lang}, save_local={updates['SAVE_LOCALLY']}).")
        if not silent:
            messagebox.showinfo("Settings Saved", "Configuration saved to .env.")
        return True

    def start_bot(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        if not self.save_settings(silent=True):
            return
        if not self.token_var.get().strip():
            messagebox.showwarning("Missing Token", "Please enter your DISCORD_TOKEN before starting the bot.")
            return

        py_exe = _resolve_python_executable()
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0
        try:
            self.proc = subprocess.Popen(
                [py_exe, "-u", MAIN_SCRIPT, "--gui-managed"],
                cwd=ROOT_DIR,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except Exception as e:
            messagebox.showerror("Launch Failed", f"Could not start main.py:\n{e}")
            return

        self.is_stopping = False
        self._set_status("Connecting...")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._append_log(f"[GUI] Started bot process (PID {self.proc.pid})...")

        reader = threading.Thread(target=self._read_process_output, args=(self.proc,), daemon=True)
        reader.start()

    def _read_process_output(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stdout:
                for line in proc.stdout:
                    self.log_queue.put(("log", line))
        finally:
            code = proc.wait()
            self.log_queue.put(("exit", str(code)))

    def stop_bot(self, wait: bool = False) -> None:
        proc = self.proc
        if proc is None:
            self._set_status("Stopped")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            return

        self.is_stopping = True
        self._set_status("Stopping...")
        self.stop_btn.configure(state="disabled")
        self._append_log(f"[GUI] Stopping bot gracefully (PID {proc.pid})...")

        if wait:
            terminate_process_tree(proc, graceful_timeout=2.0)
            self.proc = None
            self.is_stopping = False
            self._set_status("Stopped")
            self.start_btn.configure(state="normal")
            return

        def _stop_worker():
            terminate_process_tree(proc, graceful_timeout=3.0)
            self.log_queue.put(("stopped", ""))

        threading.Thread(target=_stop_worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                    parsed = parse_bot_log_line(payload)
                    if parsed["status"] and not self.is_stopping:
                        self._set_status(parsed["status"])
                    if parsed["clip"]:
                        clip = parsed["clip"]
                        self.clip_history.insert(0, clip)
                        snippet = clip["text"][:55] + ("..." if len(clip["text"]) > 55 else "")
                        self.history_list.insert(0, f"[{clip['time']}] {clip['trigger']} ({clip['latency']}) — {snippet}")
                        self._show_clip(clip)
                elif kind in ("exit", "stopped"):
                    if self.proc is not None:
                        self.proc = None
                        self.is_stopping = False
                        self._set_status("Stopped")
                        self.start_btn.configure(state="normal")
                        self.stop_btn.configure(state="disabled")
                        if kind == "exit":
                            self._append_log(f"[GUI] Bot process exited (code {payload}).")
                        else:
                            self._append_log("[GUI] Bot stopped cleanly.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_close(self) -> None:
        self.stop_bot(wait=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    DiscordClipperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
