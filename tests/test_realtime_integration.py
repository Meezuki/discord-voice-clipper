import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import main

@pytest.mark.asyncio
async def test_realtime_commands_toggle_flow():
    """Verify !realtime on, status, and off commands properly manage guild state and send embeds."""
    guild_id = 999
    mock_ctx = AsyncMock()
    mock_ctx.guild.id = guild_id
    mock_ctx.channel.send = AsyncMock()
    
    mock_msg = AsyncMock()
    mock_ctx.channel.send.return_value = mock_msg

    # 1. Calling when not in VC must fail gracefully
    main.active_sinks.pop(guild_id, None)
    await main.realtime(mock_ctx, mode="on")
    mock_ctx.send.assert_called_with("❌ Bot is not currently listening in a voice channel. Use `!join` first.")

    # 2. Simulate bot in VC
    mock_sink = MagicMock()
    main.active_sinks[guild_id] = mock_sink

    # 3. Enable realtime
    await main.realtime(mock_ctx, mode="on")
    assert main.realtime_enabled[guild_id] is True
    assert main.realtime_messages[guild_id] == mock_msg
    assert "Live Subtitles enabled" in mock_ctx.send.call_args_list[-1][0][0]

    # 4. Check status
    await main.realtime(mock_ctx, mode="status")
    assert "ENABLED" in mock_ctx.send.call_args_list[-1][0][0]

    # 5. Disable realtime
    await main.realtime(mock_ctx, mode="off")
    assert main.realtime_enabled[guild_id] is False
    assert "disabled" in mock_ctx.send.call_args_list[-1][0][0]
    mock_msg.edit.assert_called()

    # Clean up
    main.active_sinks.pop(guild_id, None)
    main.realtime_enabled.pop(guild_id, None)
    main.realtime_messages.pop(guild_id, None)
