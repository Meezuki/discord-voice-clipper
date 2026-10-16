import os
import io
import time
import logging
import asyncio
import datetime
import threading
import collections
import discord
from discord.ext import commands, voice_recv
from dotenv import load_dotenv
import numpy as np

from audio_sink import SynchronizedRollingSink, SpeechSegmenter
from asr_engine import LocalASREngine
from keyword_spotter import KeywordSpotter
from download_model import EXTRACTED_PATH, ensure_vosk_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MeetingAssistantBot")

# Silence noisy RTCP sender report packet logs from voice_recv.reader
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Ensure libopus is loaded on Windows
if not discord.opus.is_loaded():
    discord_dir = os.path.dirname(discord.__file__)
    opus_path = os.path.join(discord_dir, "bin", "libopus-0.x64.dll")
    if os.path.exists(opus_path):
        try:
            discord.opus.load_opus(opus_path)
            logger.info(f"Loaded libopus from: {opus_path}")
        except Exception as e:
            logger.warning(f"Failed to load libopus: {e}")
    else:
        logger.warning("libopus-0.x64.dll not found in discord/bin folder.")

# Patch discord-ext-voice-recv to handle DAVE E2EE and corrupted stream OpusError
def _patch_voice_recv():
    import davey
    orig_decode_packet = voice_recv.opus.PacketDecoder._decode_packet

    def safe_decode_packet(self, packet):
        assert self._decoder is not None
        if not packet:
            return packet, self._decoder.decode(None, fec=False)

        data = packet.decrypted_data
        if not data:
            return packet, b'\x00' * discord.opus.Decoder.FRAME_SIZE

        # DAVE E2EE Decryption Layer
        vc = getattr(self.router.reader, 'voice_client', None)
        if vc and hasattr(vc, '_connection'):
            dave_session = getattr(vc._connection, 'dave_session', None)
            if dave_session and getattr(dave_session, 'ready', False):
                user_id = self._cached_id or vc._get_id_from_ssrc(self.ssrc)
                if user_id:
                    try:
                        decrypted_opus = dave_session.decrypt(user_id, davey.MediaType.audio, data)
                        data = decrypted_opus
                    except Exception as e:
                        logger.debug(f"DAVE decrypt error for user {user_id}: {e}")

        try:
            pcm = self._decoder.decode(data, fec=False)
            return packet, pcm
        except discord.opus.OpusError as e:
            logger.debug(f"Ignoring corrupted stream packet (SSRC {self.ssrc}): {e}")
            return packet, b'\x00' * discord.opus.Decoder.FRAME_SIZE
        except Exception as e:
            logger.debug(f"Error decoding packet SSRC {self.ssrc}: {e}")
            return packet, b'\x00' * discord.opus.Decoder.FRAME_SIZE

    voice_recv.opus.PacketDecoder._decode_packet = safe_decode_packet

    def safe_do_run(self):
        while not self._end_thread.is_set():
            self.waiter.wait()
            with self._lock:
                for decoder in list(self.waiter.items):
                    try:
                        data = decoder.pop_data()
                        if data is not None:
                            self.sink.write(data.source, data)
                    except Exception as e:
                        logger.debug(f"Ignoring decoder error SSRC {getattr(decoder, 'ssrc', '?')}: {e}")

    voice_recv.router.PacketRouter._do_run = safe_do_run
    logger.info("PacketRouter resilience patch (DAVE E2EE & anti corrupted-stream) enabled.")

_patch_voice_recv()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize local English ASR engine (faster-whisper)
asr_engine = LocalASREngine(model_size="base", device="cpu", compute_type="int8")

# Guild state
active_sinks = {}
active_spotters = {}
active_text_channels = {}

@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (ID: {bot.user.id})")
    await asyncio.to_thread(ensure_vosk_model)

async def process_and_send_notula(guild_id: int, trigger_source: str = "Voice: 'bot record'"):
    """
    Freezes the last 30s buffer, transcribes locally via background worker,
    and sends structured meeting minutes with a WAV attachment to Discord.
    """
    sink = active_sinks.get(guild_id)
    text_channel = active_text_channels.get(guild_id)

    if not sink or not text_channel:
        logger.warning(f"Guild {guild_id}: Active sink or text channel not found.")
        return

    # Atomic snapshot of 30-second buffer
    audio_data = sink.get_audio_clip()
    wav_bytes = sink.get_wav_bytes()

    if len(audio_data) == 0 or len(wav_bytes) == 0:
        await text_channel.send("⚠️ Audio buffer is empty (no speech captured yet).")
        return

    processing_msg = await text_channel.send("⏳ **Freezing 30-second buffer & transcribing locally...**")

    # Run local English ASR in non-blocking worker thread
    text, latency = await asyncio.to_thread(asr_engine.transcribe, audio_data, "en")

    if not text.strip():
        text = "*(No intelligible speech detected)*"

    timestamp_now = int(time.time())
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    embed = discord.Embed(
        title="🎙️ Instant Meeting Minutes (Last 30 Seconds)",
        description="The last 30 seconds of audio were captured and transcribed completely offline.",
        color=discord.Color.brand_green()
    )
    embed.add_field(name="🕒 Meeting Time", value=f"<t:{timestamp_now}:F> (`{time_str}`)", inline=False)
    embed.add_field(name="🎯 Trigger Source", value=f"`{trigger_source}`", inline=True)
    embed.add_field(name="⚡ Inference Latency", value=f"`{latency:.2f} s` ({latency * 1000:.0f} ms)", inline=True)
    embed.add_field(name="📝 Transcription", value=f">>> {text}", inline=False)
    embed.set_footer(text="Discord Voice Clipper • Offline Local ASR & KWS")

    wav_io = io.BytesIO(wav_bytes)
    file_attachment = discord.File(fp=wav_io, filename=f"minutes_{timestamp_now}.wav")

    try:
        await processing_msg.delete()
    except Exception:
        pass

    await text_channel.send(embed=embed, file=file_attachment)
    logger.info(f"Meeting minutes sent to #{text_channel.name} (Guild: {guild_id})")

# Real-time live transcription state per guild
realtime_enabled = {}
realtime_messages = {}
realtime_lines = {}
realtime_segmenters = {}
realtime_last_edit = {}
realtime_pending_update = {}

async def trigger_embed_update(guild_id: int):
    msg = realtime_messages.get(guild_id)
    if not msg or not realtime_enabled.get(guild_id, False):
        return

    now = time.time()
    last_edit = realtime_last_edit.get(guild_id, 0.0)

    # Respect Discord rate limit (max 1 edit per 1.2s)
    if now - last_edit < 1.2:
        if not realtime_pending_update.get(guild_id, False):
            realtime_pending_update[guild_id] = True
            await asyncio.sleep(1.2 - (now - last_edit))
            realtime_pending_update[guild_id] = False
            await trigger_embed_update(guild_id)
        return

    realtime_last_edit[guild_id] = now
    lines = list(realtime_lines.get(guild_id, []))
    content = "\n\n".join(lines) if lines else "*(Listening for speech...)*"

    embed = discord.Embed(
        title="🔴 Live Meeting Transcription",
        description=content,
        color=discord.Color.red()
    )
    embed.set_footer(text="Discord Voice Clipper • Live Subtitles Active (Last 5 sentences)")
    try:
        await msg.edit(embed=embed)
    except Exception as e:
        logger.debug(f"Failed to edit live embed: {e}")

def make_speech_segment_callback(guild_id: int):
    """Callback when SpeechSegmenter flushes an active sentence segment."""
    def on_speech_segment(user_id, speaker_name, audio_float32):
        if not realtime_enabled.get(guild_id, False):
            return

        def transcribe_and_update():
            text, _ = asr_engine.transcribe(audio_float32, language="en")
            text = text.strip()
            if not text:
                return

            clean_lower = text.lower()
            # If the user spoke a wake word, trigger minutes snapshot
            if any(w in clean_lower for w in ["bot record", "record", "but record"]):
                logger.info(f"Wake word detected inside real-time stream: '{clean_lower}'")
                asyncio.run_coroutine_threadsafe(
                    process_and_send_notula(guild_id, trigger_source=f"Realtime: '{clean_lower}'"),
                    bot.loop
                )
                return

            ts = int(time.time())
            line = f"<t:{ts}:T> **{speaker_name}**: \"{text}\""
            if guild_id not in realtime_lines:
                realtime_lines[guild_id] = collections.deque(maxlen=5)
            realtime_lines[guild_id].append(line)
            logger.info(f"💬 [Realtime Subtitle]: {speaker_name}: \"{text}\"")

            asyncio.run_coroutine_threadsafe(trigger_embed_update(guild_id), bot.loop)

        threading.Thread(target=transcribe_and_update, daemon=True).start()

    return on_speech_segment

def make_keyword_callback(guild_id: int):
    """Callback when Vosk detects 'bot record' or 'record'."""
    def on_trigger(recognized_phrase):
        logger.info(f"Trigger received in guild {guild_id}: '{recognized_phrase}'")
        asyncio.run_coroutine_threadsafe(
            process_and_send_notula(guild_id, trigger_source=f"Voice: '{recognized_phrase}'"),
            bot.loop
        )
    return on_trigger


@bot.command(name="join", help="Connects bot to the user's voice channel and begins rolling recording.")
async def join(ctx):
    if not ctx.author.voice:
        await ctx.send("❌ You must be in a voice channel first.")
        return

    channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    # Store origin text channel for minutes dispatch
    active_text_channels[guild_id] = ctx.channel

    # Dual-Engine English Keyword Spotter: Grammar-Constrained Vosk + Rolling Whisper Fallback
    spotter = KeywordSpotter(
        model_path=EXTRACTED_PATH,
        target_phrases=["bot record", "record", "but record", "boat record", "recording", "start record"],
        cooldown_seconds=8.0,
        trigger_callback=make_keyword_callback(guild_id),
        asr_engine=asr_engine
    )
    active_spotters[guild_id] = spotter

    # Speech Segmenter for Real-Time Live Subtitles
    segmenter = SpeechSegmenter(
        segment_callback=make_speech_segment_callback(guild_id),
        speech_threshold=0.012,
        silence_timeout_sec=0.7,
        min_duration_sec=0.8
    )
    realtime_segmenters[guild_id] = segmenter
    realtime_enabled[guild_id] = False
    realtime_lines[guild_id] = collections.deque(maxlen=5)

    # Rolling Audio Sink with 100ms streaming chunk feed to Vosk and segmenter hook
    sink = SynchronizedRollingSink(
        buffer_seconds=30,
        chunk_callback=spotter.process_chunk,
        chunk_ms=100,
        segmenter=segmenter
    )
    active_sinks[guild_id] = sink

    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    conn = vc._connection
    dave_ver = getattr(conn, 'dave_protocol_version', 0)
    dave_sess = getattr(conn, 'dave_session', None)
    logger.info(f"Voice Channel: '{channel.name}' | DAVE Version: {dave_ver} | DAVE Ready: {getattr(dave_sess, 'ready', False)}")
    vc.listen(sink)

    embed = discord.Embed(
        title="🎙️ Meeting Assistant Active",
        description=(
            f"Bot is listening in voice channel `{channel.name}`.\n\n"
            f"**How to Capture Meeting Minutes:**\n"
            f"• Say the wake word: **`\"bot record\"`** or **`\"record\"`**\n"
            f"• Or use text command: **`!clip`**\n\n"
            f"**Live Speech Subtitles:**\n"
            f"• Start live transcription: **`!realtime on`** *(or `!live on`)*\n"
            f"• Stop live transcription: **`!realtime off`**\n\n"
            f"Meeting minutes and the 30-second WAV clip will be sent automatically to #{ctx.channel.name}."
        ),
        color=discord.Color.blurple()
    )
    await ctx.send(embed=embed)

@bot.command(name="realtime", aliases=["live"], help="Toggle live speech-to-text subtitles in chat: !realtime on | !realtime off | !realtime status")
async def realtime(ctx, mode: str = "status"):
    guild_id = ctx.guild.id
    mode = mode.lower().strip()

    if guild_id not in active_sinks:
        await ctx.send("❌ Bot is not currently listening in a voice channel. Use `!join` first.")
        return

    if mode in ["on", "start", "enable"]:
        if realtime_enabled.get(guild_id, False) and realtime_messages.get(guild_id):
            await ctx.send("ℹ️ Real-time live subtitles are already enabled.")
            return

        realtime_enabled[guild_id] = True
        embed = discord.Embed(
            title="🔴 Live Meeting Transcription",
            description="*(Listening for speech...)*",
            color=discord.Color.red()
        )
        embed.set_footer(text="Discord Voice Clipper • Live Subtitles Active (Last 5 sentences)")
        msg = await ctx.channel.send(embed=embed)
        realtime_messages[guild_id] = msg
        realtime_last_edit[guild_id] = time.time()
        await ctx.send("✅ **Live Subtitles enabled!** As participants speak, the embed above will update in real-time.")

    elif mode in ["off", "stop", "disable"]:
        realtime_enabled[guild_id] = False
        msg = realtime_messages.get(guild_id)
        if msg:
            lines = list(realtime_lines.get(guild_id, []))
            content = "\n\n".join(lines) if lines else "*(Session paused)*"
            embed = discord.Embed(
                title="⚪ Live Meeting Transcription (Paused)",
                description=content,
                color=discord.Color.light_grey()
            )
            embed.set_footer(text="Discord Voice Clipper • Live Subtitles Paused")
            try:
                await msg.edit(embed=embed)
            except Exception:
                pass
        await ctx.send("⏹️ Real-time live subtitles disabled.")

    elif mode == "status":
        is_on = realtime_enabled.get(guild_id, False)
        status_str = "🟢 **ENABLED**" if is_on else "⚪ **DISABLED**"
        await ctx.send(f"📊 Real-time live subtitles: {status_str}\nUse `!realtime on` or `!realtime off` to change.")
    else:
        await ctx.send("❌ Invalid option. Use `!realtime on`, `!realtime off`, or `!realtime status`.")

@bot.command(name="clip", help="Manually freeze and transcribe the last 30 seconds of audio.")
async def clip(ctx):
    guild_id = ctx.guild.id
    if guild_id not in active_sinks:
        await ctx.send("❌ Bot is not currently listening in a voice channel. Use `!join` first.")
        return

    active_text_channels[guild_id] = ctx.channel
    await process_and_send_notula(guild_id, trigger_source=f"Command: !clip by {ctx.author.display_name}")

@bot.command(name="leave", help="Disconnects bot from voice channel and frees buffers.")
async def leave(ctx):
    guild_id = ctx.guild.id
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        if guild_id in active_sinks:
            active_sinks[guild_id].cleanup()
            del active_sinks[guild_id]
        if guild_id in active_spotters:
            del active_spotters[guild_id]
        if guild_id in active_text_channels:
            del active_text_channels[guild_id]
        if guild_id in realtime_enabled:
            msg = realtime_messages.get(guild_id)
            if msg:
                try:
                    embed = discord.Embed(
                        title="⚪ Live Meeting Transcription (Ended)",
                        description="\n\n".join(list(realtime_lines.get(guild_id, []))) or "*(Meeting closed)*",
                        color=discord.Color.light_grey()
                    )
                    embed.set_footer(text="Discord Voice Clipper • Session Ended")
                    await msg.edit(embed=embed)
                except Exception:
                    pass
            del realtime_enabled[guild_id]
        if guild_id in realtime_messages:
            del realtime_messages[guild_id]
        if guild_id in realtime_lines:
            del realtime_lines[guild_id]
        if guild_id in realtime_segmenters:
            del realtime_segmenters[guild_id]
        if guild_id in realtime_last_edit:
            del realtime_last_edit[guild_id]
        if guild_id in realtime_pending_update:
            del realtime_pending_update[guild_id]

        await ctx.send("👋 Bot left the voice channel and cleared all audio buffers.")
    else:
        await ctx.send("Bot is not in a voice channel.")


if __name__ == "__main__":
    if not TOKEN:
        print("WARNING: DISCORD_TOKEN is missing in .env file!")
    bot.run(TOKEN)