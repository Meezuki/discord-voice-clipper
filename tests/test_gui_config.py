import tkinter as tk
import pytest

from gui import (
    load_env_settings,
    save_env_settings,
    parse_duration_choice,
    format_duration_choice,
    parse_bot_log_line,
    DiscordClipperGUI,
)


def test_env_save_preserves_unrelated_keys_and_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Discord Config\n"
        "DISCORD_TOKEN=old_token\n"
        "CUSTOM_UNRELATED_KEY=keep_me\n"
        "\n"
        "RECORDINGS_DIR=D:/custom_dir\n",
        encoding="utf-8",
    )

    save_env_settings(
        str(env_file),
        {
            "DISCORD_TOKEN": "new_secret_token",
            "DEFAULT_BUFFER_SECONDS": "1800",
            "DEFAULT_LANGUAGE": "auto",
            "SAVE_LOCALLY": "true",
        },
    )

    content = env_file.read_text(encoding="utf-8")
    assert "# Discord Config" in content
    assert "CUSTOM_UNRELATED_KEY=keep_me" in content
    assert "RECORDINGS_DIR=D:/custom_dir" in content

    loaded = load_env_settings(str(env_file))
    assert loaded["DISCORD_TOKEN"] == "new_secret_token"
    assert loaded["DEFAULT_BUFFER_SECONDS"] == "1800"
    assert loaded["DEFAULT_LANGUAGE"] == "auto"
    assert loaded["SAVE_LOCALLY"] == "true"


def test_parse_and_format_duration_choice():
    assert parse_duration_choice("30s") == 30
    assert parse_duration_choice("1m") == 60
    assert parse_duration_choice("5m") == 300
    assert parse_duration_choice("30m") == 1800
    assert parse_duration_choice("90") == 90

    assert format_duration_choice(30) == "30s"
    assert format_duration_choice(300) == "5m"
    assert format_duration_choice(90) == "90s"

    with pytest.raises(ValueError):
        parse_duration_choice("5s")
    with pytest.raises(ValueError):
        parse_duration_choice("120m")
    with pytest.raises(ValueError):
        parse_duration_choice("invalid")


def test_parse_bot_log_line_status_and_clip_event():
    assert parse_bot_log_line("2026-11-17 [INFO] MeetingAssistantBot: Bot connected as ClipBot#0001")["status"] == "Online (Idle)"
    assert parse_bot_log_line("2026-11-17 [INFO] MeetingAssistantBot: Voice Channel: 'General' | DAVE Version: 1")["status"] == "Listening in VC"
    assert parse_bot_log_line("2026-11-17 [INFO] MeetingAssistantBot: Transcribing 30 seconds clip (Voice: 'bot clip that')...")["status"] == "Transcribing..."
    assert parse_bot_log_line("2026-11-17 [INFO] MeetingAssistantBot: Disconnected from voice channel in guild 1.")["status"] == "Online (Idle)"

    clip_line = (
        "2026-11-17 [INFO] MeetingAssistantBot: [CLIP_EVENT] "
        "time=2026-11-17 19:10:00 UTC | trigger=Voice: 'bot clip that (conf=0.98)' | "
        "latency=0.42s | dest=Discord #general | text=Deploy the hotfix after lunch."
    )
    parsed = parse_bot_log_line(clip_line)
    assert parsed["status"] == "Listening in VC"
    assert parsed["clip"] == {
        "time": "2026-11-17 19:10:00 UTC",
        "trigger": "Voice: 'bot clip that (conf=0.98)'",
        "latency": "0.42s",
        "dest": "Discord #general",
        "text": "Deploy the hotfix after lunch.",
    }


def test_gui_initializes_and_destroys_cleanly():
    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError as e:
        pytest.skip(f"No GUI display available: {e}")

    app = DiscordClipperGUI(root)
    assert app.status_pill.cget("text") == "Stopped"
    root.destroy()


def test_terminate_process_tree_graceful_stdin_stop():
    import subprocess, sys
    from gui import terminate_process_tree

    # Spawn a tiny Python child that exits cleanly with code 0 when 'STOP' arrives on stdin
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0 if sys.stdin.readline().strip() == 'STOP' else 2)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    terminate_process_tree(proc, graceful_timeout=3.0)
    assert proc.poll() == 0


@pytest.mark.asyncio
async def test_main_graceful_shutdown_cleans_voice_and_state(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
    import main

    mock_vc = MagicMock()
    mock_vc.disconnect = AsyncMock()
    mock_sink = MagicMock()
    main.active_sinks[111] = mock_sink
    main.active_durations[111] = 30

    with patch.object(type(main.bot), "voice_clients", new_callable=PropertyMock, return_value=[mock_vc]), \
         patch.object(main.bot, "is_closed", return_value=False), \
         patch.object(main.bot, "change_presence", new_callable=AsyncMock) as mock_presence, \
         patch.object(main.bot, "close", new_callable=AsyncMock) as mock_close:
        await main._graceful_shutdown()

    mock_vc.disconnect.assert_awaited_once_with(force=True)
    mock_sink.cleanup.assert_called_once()
    assert main.active_sinks == {}
    mock_presence.assert_awaited_once()
    mock_close.assert_awaited_once()
