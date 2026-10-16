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
    assert "Instant Meeting Minutes" in embed.title
    assert file_att is not None
    assert file_att.filename.endswith(".wav")

