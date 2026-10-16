import pytest
import discord
from unittest.mock import MagicMock
from discord.ext import voice_recv
import main

def test_packet_decoder_corrupted_stream_handled_gracefully():
    """Verify that OpusError 'corrupted stream' does not throw an uncaught exception."""
    mock_router = MagicMock()
    decoder = voice_recv.opus.PacketDecoder(mock_router, ssrc=12345)
    
    # Mock internal decoder to simulate throwing OpusError: corrupted stream
    mock_opus_decoder = MagicMock()
    mock_opus_decoder.decode.side_effect = discord.opus.OpusError(-4)  # OPUS_INVALID_PACKET / corrupted stream
    decoder._decoder = mock_opus_decoder

    mock_packet = MagicMock()
    mock_packet.decrypted_data = b"corrupted_bytes"

    # Must return (packet, silence_bytes) without crashing
    pkt, pcm = decoder._decode_packet(mock_packet)
    assert pkt == mock_packet
    assert len(pcm) == discord.opus.Decoder.FRAME_SIZE
    assert pcm == b"\x00" * discord.opus.Decoder.FRAME_SIZE

