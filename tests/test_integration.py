import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import main

@pytest.mark.asyncio
async def test_end_to_end_pipeline_simulation():
    """Verify end-to-end pipeline simulation from buffer snapshot to Discord embed creation."""
    guild_id = 12345
    
    # Mocking sink
    mock_sink = MagicMock()
    # 1.0 second of dummy audio
    mock_audio = np.zeros(16000, dtype=np.float32)
    mock_sink.get_audio_clip.return_value = mock_audio
    mock_sink.get_wav_bytes.return_value = b"RIFF....WAVEfmt...."
    
    # Mocking text channel
    mock_text_channel = AsyncMock()
    mock_msg = AsyncMock()
    mock_text_channel.send.return_value = mock_msg
    mock_text_channel.name = "general-meeting"

    main.active_sinks[guild_id] = mock_sink
    main.active_text_channels[guild_id] = mock_text_channel

    # Mocking ASR engine
    with patch.object(main.asr_engine, "transcribe", return_value=("The meeting agenda was approved.", 0.45)):
        await main.process_and_send_notula(guild_id, trigger_source="Test Simulation")

    # Verify sending embed and WAV attachment
    assert mock_text_channel.send.call_count == 2  # 1 progress msg, 1 final minutes
    final_call_kwargs = mock_text_channel.send.call_args_list[1].kwargs
    
    embed = final_call_kwargs.get("embed")
    file_att = final_call_kwargs.get("file")

    assert embed is not None
    assert "Instant Voice Clip" in embed.title
    assert file_att is not None
    assert file_att.filename.startswith("clip_") and file_att.filename.endswith(".wav")


def test_parse_duration_seconds():
    assert main.parse_duration_seconds("30") == 30
    assert main.parse_duration_seconds("45s") == 45
    assert main.parse_duration_seconds("30m") == 1800
    with pytest.raises(ValueError):
        main.parse_duration_seconds("5")
    with pytest.raises(ValueError):
        main.parse_duration_seconds("120m")
    with pytest.raises(ValueError):
        main.parse_duration_seconds("abc")


@pytest.mark.asyncio
async def test_local_save_for_30s_clip(tmp_path, monkeypatch):
    """Verify 30s clip with save_local=True attaches .wav to Discord AND writes .wav/.txt locally."""
    guild_id = 54321
    monkeypatch.setattr(main, "RECORDINGS_DIR", str(tmp_path))

    mock_sink = MagicMock()
    mock_sink.buffer_seconds = 30
    mock_sink.get_audio_clip.return_value = np.zeros(16000, dtype=np.float32)
    mock_sink.get_wav_bytes.return_value = b"RIFF_30S_WAV_DATA"

    mock_text_channel = AsyncMock()
    mock_text_channel.send.return_value = AsyncMock()
    mock_text_channel.name = "general"

    main.active_sinks[guild_id] = mock_sink
    main.active_text_channels[guild_id] = mock_text_channel
    main.active_durations[guild_id] = 30
    main.active_save_local[guild_id] = True

    try:
        with patch.object(main.asr_engine, "transcribe", return_value=("Short 30s decision.", 0.30)):
            await main.process_and_send_notula(guild_id, trigger_source="Test 30s Local")

        final_kwargs = mock_text_channel.send.call_args_list[1].kwargs
        file_att = final_kwargs["file"]
        embed = final_kwargs["embed"]
        assert file_att.filename.endswith(".wav")
        assert any(f.name == "💾 Saved Locally" for f in embed.fields)
        assert len(list(tmp_path.glob("clip_*.wav"))) == 1
        assert len(list(tmp_path.glob("clip_*.txt"))) == 1
    finally:
        main.active_sinks.pop(guild_id, None)
        main.active_text_channels.pop(guild_id, None)
        main.active_durations.pop(guild_id, None)
        main.active_save_local.pop(guild_id, None)


@pytest.mark.asyncio
async def test_long_clip_saves_wav_locally_and_attaches_txt(tmp_path, monkeypatch):
    """Verify >30s clip saves .wav locally, attaches .txt in Discord, and notes upload limit."""
    guild_id = 98765
    monkeypatch.setattr(main, "RECORDINGS_DIR", str(tmp_path))

    mock_sink = MagicMock()
    mock_sink.buffer_seconds = 1800
    mock_sink.get_audio_clip.return_value = np.zeros(16000, dtype=np.float32)
    mock_sink.get_wav_bytes.return_value = b"RIFF_LONG_WAV_DATA"

    mock_text_channel = AsyncMock()
    mock_text_channel.send.return_value = AsyncMock()
    mock_text_channel.name = "meeting-notes"

    main.active_sinks[guild_id] = mock_sink
    main.active_text_channels[guild_id] = mock_text_channel
    main.active_durations[guild_id] = 1800
    main.active_save_local[guild_id] = False

    long_transcript = "word " * 400  # 2000 chars > Discord 1024 embed field limit
    try:
        with patch.object(main.asr_engine, "transcribe", return_value=(long_transcript.strip(), 1.25)):
            await main.process_and_send_notula(guild_id, trigger_source="Test 30m")

        final_kwargs = mock_text_channel.send.call_args_list[1].kwargs
        embed = final_kwargs["embed"]
        file_att = final_kwargs["file"]

        assert file_att.filename.endswith(".txt")
        assert "not uploaded to Discord" in embed.description
        transcript_field = next(f for f in embed.fields if "Transcription" in f.name)
        assert len(transcript_field.value) <= 1024

        wav_files = list(tmp_path.glob("clip_*.wav"))
        txt_files = list(tmp_path.glob("clip_*.txt"))
        assert len(wav_files) == 1 and len(txt_files) == 1
        assert long_transcript.strip() in txt_files[0].read_text(encoding="utf-8")
    finally:
        main.active_sinks.pop(guild_id, None)
        main.active_text_channels.pop(guild_id, None)
        main.active_durations.pop(guild_id, None)
        main.active_save_local.pop(guild_id, None)

