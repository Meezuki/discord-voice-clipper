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

from audio_sink import SynchronizedRollingSink
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
active_languages = {}  # guild_id -> 'en' | 'id' | 'auto'

@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (ID: {bot.user.id})")
    await asyncio.to_thread(ensure_vosk_model)

async def process_and_send_notula(guild_id: int, trigger_source: str = "Voice: 'bot clip that'"):
    """
    Freezes the last 30s buffer, transcribes locally via background worker,
    and sends structured meeting minutes / voice clip with a WAV attachment to Discord.
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

    # Run local multilingual ASR with guild-selected language (defaults to 'en', or 'id' / 'auto')
    guild_lang = active_languages.get(guild_id, "en")
    lang_param = None if guild_lang == "auto" else guild_lang
    text, latency = await asyncio.to_thread(asr_engine.transcribe, audio_data, lang_param)

    if not text.strip():
        text = "*(No intelligible speech detected)*"

    timestamp_now = int(time.time())
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    embed = discord.Embed(
        title="🎬 Instant Voice Clip (Last 30 Seconds)",
        description="The last 30 seconds of audio were captured and transcribed completely offline.",
        color=discord.Color.brand_green()
    )
    embed.add_field(name="🕒 Meeting Time", value=f"<t:{timestamp_now}:F> (`{time_str}`)", inline=False)
    embed.add_field(name="🎯 Trigger Source", value=f"`{trigger_source}`", inline=True)
    embed.add_field(name="⚡ Inference Latency", value=f"`{latency:.2f} s` ({latency * 1000:.0f} ms)", inline=True)
    embed.add_field(name="📝 Transcription", value=f">>> {text}", inline=False)
    embed.set_footer(text="Discord Voice Clipper • Offline Local ASR & KWS")

    wav_io = io.BytesIO(wav_bytes)
    file_attachment = discord.File(fp=wav_io, filename=f"clip_{timestamp_now}.wav")

    try:
        await processing_msg.delete()
    except Exception:
        pass

    await text_channel.send(embed=embed, file=file_attachment)
    logger.info(f"Voice clip sent to #{text_channel.name} (Guild: {guild_id})")

def make_keyword_callback(guild_id: int):
    """Callback when wake word 'bot clip that' is detected."""
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

    # Two-Stage Keyword Spotter: Open-Vocabulary Vosk + Faster-Whisper Verification
    spotter = KeywordSpotter(
        model_path=EXTRACTED_PATH,
        target_phrases=["bot clip that", "clip that", "bot clip this", "please clip that"],
        cooldown_seconds=8.0,
        trigger_callback=make_keyword_callback(guild_id),
        asr_engine=asr_engine
    )
    active_spotters[guild_id] = spotter

    # Rolling Audio Sink with 100ms streaming chunk feed to KWS
    sink = SynchronizedRollingSink(
        buffer_seconds=30,
        chunk_callback=spotter.process_chunk,
        chunk_ms=100
    )
    active_sinks[guild_id] = sink

    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    conn = vc._connection
    dave_ver = getattr(conn, 'dave_protocol_version', 0)
    dave_sess = getattr(conn, 'dave_session', None)
    logger.info(f"Voice Channel: '{channel.name}' | DAVE Version: {dave_ver} | DAVE Ready: {getattr(dave_sess, 'ready', False)}")
    vc.listen(sink)

    embed = discord.Embed(
        title="🎙️ Meeting Voice Clipper Active",
        description=(
            f"Bot is listening in voice channel `{channel.name}`.\n\n"
            f"**How to Capture Audio Clips:**\n"
            f"• Say the wake word: **`\"bot clip that\"`**\n"
            f"• Or use text command: **`!clip`**\n\n"
            f"The 30-second audio clip and transcription will be sent to #{ctx.channel.name}."
        ),
        color=discord.Color.blurple()
    )
    await ctx.send(embed=embed)

@bot.command(name="clip", help="Manually freeze and transcribe the last 30 seconds of audio.")
async def clip(ctx):
    guild_id = ctx.guild.id
    if guild_id not in active_sinks:
        await ctx.send("❌ Bot is not currently listening in a voice channel. Use `!join` first.")
        return

    active_text_channels[guild_id] = ctx.channel
    await process_and_send_notula(guild_id, trigger_source=f"Command: !clip by {ctx.author.display_name}")

@bot.command(name="lang", aliases=["language"], help="Sets transcription language ('en', 'id', or 'auto').")
async def set_language(ctx, lang_code: str = None):
    guild_id = ctx.guild.id
    if not lang_code:
        current = active_languages.get(guild_id, "en")
        await ctx.send(f"🌐 Current transcription language is: `{current}`. Use `!lang en`, `!lang id`, or `!lang auto` to change.")
        return

    lang_code = lang_code.lower().strip()
    if lang_code not in ["en", "id", "auto"]:
        await ctx.send("❌ Unsupported language! Please choose `en` (English), `id` (Indonesian), or `auto`.")
        return

    active_languages[guild_id] = lang_code
    names = {"en": "English", "id": "Indonesian (Bahasa Indonesia)", "auto": "Auto-Detect (English / Indonesian)"}
    await ctx.send(f"✅ Transcription language set to **{names[lang_code]}** (`{lang_code}`).")

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
        if guild_id in active_languages:
            del active_languages[guild_id]
        await ctx.send("👋 Bot left the voice channel and cleared all audio buffers.")
    else:
        await ctx.send("Bot is not in a voice channel.")


if __name__ == "__main__":
    if not TOKEN:
        print("WARNING: DISCORD_TOKEN is missing in .env file!")
    bot.run(TOKEN)