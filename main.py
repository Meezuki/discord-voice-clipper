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
from download_model import (
    MEL_MODEL_PATH,
    EMB_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_CONFIG_PATH,
    ensure_wakeword_backbone,
)

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

MIN_BUFFER_SECONDS = 10
MAX_BUFFER_SECONDS = 3600
DISCORD_WAV_MAX_SECONDS = 30
EMBED_TEXT_PREVIEW_LIMIT = 950


def parse_duration_seconds(raw: str) -> int:
    """Parses duration strings like '30', '120s', or '30m' into bounded seconds."""
    val = raw.strip().lower()
    if not val:
        raise ValueError("Duration cannot be empty.")
    multiplier = 1
    if val.endswith("m"):
        multiplier = 60
        val = val[:-1].strip()
    elif val.endswith("s"):
        val = val[:-1].strip()
    if not val.isdigit():
        raise ValueError(f"Invalid duration '{raw}'. Use seconds (e.g. 30, 1800) or minutes (e.g. 5m, 30m).")
    seconds = int(val) * multiplier
    if not (MIN_BUFFER_SECONDS <= seconds <= MAX_BUFFER_SECONDS):
        raise ValueError(
            f"Duration must be between {MIN_BUFFER_SECONDS}s and {MAX_BUFFER_SECONDS}s (60m)."
        )
    return seconds


def format_duration_label(seconds: int) -> str:
    if seconds >= 60 and seconds % 60 == 0:
        mins = seconds // 60
        unit = "minute" if mins == 1 else "minutes"
        return f"{mins} {unit} ({seconds}s)"
    return f"{seconds} seconds"


try:
    DEFAULT_BUFFER_SECONDS = parse_duration_seconds(os.getenv("DEFAULT_BUFFER_SECONDS", "30"))
except ValueError:
    DEFAULT_BUFFER_SECONDS = 30

DEFAULT_SAVE_LOCALLY = os.getenv("SAVE_LOCALLY", "false").strip().lower() in {"1", "true", "yes", "on"}
_env_lang = os.getenv("DEFAULT_LANGUAGE", "en").strip().lower()
DEFAULT_LANGUAGE = _env_lang if _env_lang in {"en", "id", "auto"} else "en"
RECORDINGS_DIR = os.getenv(
    "RECORDINGS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
)

# Guild state
active_sinks = {}
active_spotters = {}
active_text_channels = {}
active_languages = {}   # guild_id -> 'en' | 'id' | 'auto'
active_durations = {}   # guild_id -> int seconds (default 30)
active_save_local = {}  # guild_id -> bool (default False)

@bot.event
async def on_ready():
    logger.info(f"Bot connected as {bot.user} (ID: {bot.user.id})")
    await asyncio.to_thread(ensure_wakeword_backbone)

def _format_transcript_document(
    timestamp_now: int,
    time_str: str,
    duration_sec: int,
    trigger_source: str,
    latency: float,
    text: str,
) -> str:
    return (
        f"Discord Voice Clipper — Transcript\n"
        f"Timestamp: {time_str} ({timestamp_now})\n"
        f"Captured Window: {format_duration_label(duration_sec)}\n"
        f"Trigger Source: {trigger_source}\n"
        f"ASR Latency: {latency:.2f} s\n"
        f"{'-' * 50}\n"
        f"{text}\n"
    )


def _format_local_path(path: str) -> str:
    try:
        return os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
    except ValueError:
        return path


def _save_clip_locally(
    recordings_dir: str,
    timestamp_now: int,
    wav_bytes: bytes,
    transcript_doc: str,
) -> tuple[str, str]:
    os.makedirs(recordings_dir, exist_ok=True)
    wav_path = os.path.join(recordings_dir, f"clip_{timestamp_now}.wav")
    txt_path = os.path.join(recordings_dir, f"clip_{timestamp_now}.txt")
    with open(wav_path, "wb") as wf:
        wf.write(wav_bytes)
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(transcript_doc)
    return wav_path, txt_path


async def process_and_send_notula(guild_id: int, trigger_source: str = "Voice: 'bot clip that'"):
    """
    Freezes the rolling buffer, transcribes locally via background worker,
    saves locally when configured or > 30s, and dispatches to Discord.
    """
    sink = active_sinks.get(guild_id)
    text_channel = active_text_channels.get(guild_id)

    if not sink or not text_channel:
        logger.warning(f"Guild {guild_id}: Active sink or text channel not found.")
        return

    raw_dur = getattr(sink, "buffer_seconds", None)
    buffer_seconds = int(raw_dur) if isinstance(raw_dur, (int, float)) else active_durations.get(guild_id, DEFAULT_BUFFER_SECONDS)
    exceeds_discord_limit = buffer_seconds > DISCORD_WAV_MAX_SECONDS
    save_local = active_save_local.get(guild_id, DEFAULT_SAVE_LOCALLY) or exceeds_discord_limit

    # Atomic snapshot of rolling buffer
    audio_data = sink.get_audio_clip()
    wav_bytes = sink.get_wav_bytes()

    if len(audio_data) == 0 or len(wav_bytes) == 0:
        await text_channel.send("⚠️ Audio buffer is empty (no speech captured yet).")
        return

    dur_label = format_duration_label(buffer_seconds)
    logger.info(f"Transcribing {dur_label} clip ({trigger_source})...")
    processing_msg = await text_channel.send(
        f"⏳ **Freezing {dur_label} buffer & transcribing locally...**"
    )

    # Run local multilingual ASR with guild-selected language (defaults to DEFAULT_LANGUAGE)
    guild_lang = active_languages.get(guild_id, DEFAULT_LANGUAGE)
    lang_param = None if guild_lang == "auto" else guild_lang
    text, latency = await asyncio.to_thread(asr_engine.transcribe, audio_data, lang_param)

    if not text.strip():
        text = "*(No intelligible speech detected)*"

    timestamp_now = int(time.time())
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    transcript_doc = _format_transcript_document(
        timestamp_now, time_str, buffer_seconds, trigger_source, latency, text
    )

    saved_paths = None
    if save_local:
        saved_paths = await asyncio.to_thread(
            _save_clip_locally, RECORDINGS_DIR, timestamp_now, wav_bytes, transcript_doc
        )

    if exceeds_discord_limit:
        title = f"🎬 Instant Voice Clip (Last {dur_label})"
        description = (
            f"The last {dur_label} of audio were captured and transcribed completely offline.\n"
            f"ℹ️ **Audio longer than {DISCORD_WAV_MAX_SECONDS}s is not uploaded to Discord due to "
            f"text-channel file size limits — saved locally instead.**"
        )
        preview = text
        if len(preview) > EMBED_TEXT_PREVIEW_LIMIT:
            preview = preview[:EMBED_TEXT_PREVIEW_LIMIT].rstrip() + "\n… *(see attached `.txt` file for full transcript)*"
        transcript_field_name = "📝 Transcription Preview"
        file_attachment = discord.File(
            fp=io.BytesIO(transcript_doc.encode("utf-8")),
            filename=f"clip_{timestamp_now}.txt",
        )
    else:
        title = f"🎬 Instant Voice Clip (Last {buffer_seconds} Seconds)"
        description = f"The last {buffer_seconds} seconds of audio were captured and transcribed completely offline."
        preview = text if len(text) <= 1015 else text[:EMBED_TEXT_PREVIEW_LIMIT].rstrip() + "…"
        transcript_field_name = "📝 Transcription"
        file_attachment = discord.File(fp=io.BytesIO(wav_bytes), filename=f"clip_{timestamp_now}.wav")

    embed = discord.Embed(
        title=title,
        description=description,
        color=discord.Color.brand_green()
    )
    embed.add_field(name="🕒 Meeting Time", value=f"<t:{timestamp_now}:F> (`{time_str}`)", inline=False)
    embed.add_field(name="🎯 Trigger Source", value=f"`{trigger_source}`", inline=True)
    embed.add_field(name="⚡ Inference Latency", value=f"`{latency:.2f} s` ({latency * 1000:.0f} ms)", inline=True)
    embed.add_field(name=transcript_field_name, value=f">>> {preview}", inline=False)
    if saved_paths:
        rel_wav = _format_local_path(saved_paths[0])
        rel_txt = _format_local_path(saved_paths[1])
        embed.add_field(name="💾 Saved Locally", value=f"`{rel_wav}`\n`{rel_txt}`", inline=False)
    embed.set_footer(text="Discord Voice Clipper • Offline Local ASR & KWS")

    try:
        await processing_msg.delete()
    except Exception:
        pass

    await text_channel.send(embed=embed, file=file_attachment)
    dest_summary = (
        f"Local: {_format_local_path(saved_paths[0])}"
        if exceeds_discord_limit and saved_paths
        else (
            f"Discord #{text_channel.name} + Local: {_format_local_path(saved_paths[0])}"
            if saved_paths
            else f"Discord #{text_channel.name}"
        )
    )
    clean_text = " ".join(text.splitlines()).strip()
    logger.info(
        f"[CLIP_EVENT] time={time_str} | trigger={trigger_source} | latency={latency:.2f}s | dest={dest_summary} | text={clean_text}"
    )
    logger.info(f"Voice clip dispatched to #{text_channel.name} (Guild: {guild_id}, duration={buffer_seconds}s, local={save_local})")

def make_keyword_callback(guild_id: int):
    """Callback when wake word 'bot clip that' is detected."""
    def on_trigger(recognized_phrase):
        logger.info(f"Trigger received in guild {guild_id}: '{recognized_phrase}'")
        asyncio.run_coroutine_threadsafe(
            process_and_send_notula(guild_id, trigger_source=f"Voice: '{recognized_phrase}'"),
            bot.loop
        )
    return on_trigger


@bot.command(name="join", help="Connects bot to your voice channel. Optional duration: '!join 30' or '!join 30m'.")
async def join(ctx, duration: str = None):
    if not ctx.author.voice:
        await ctx.send("❌ You must be in a voice channel first.")
        return

    channel = ctx.author.voice.channel
    guild_id = ctx.guild.id

    if duration is not None:
        try:
            buffer_seconds = parse_duration_seconds(duration)
            active_durations[guild_id] = buffer_seconds
        except ValueError as e:
            await ctx.send(f"❌ {e}")
            return
    else:
        buffer_seconds = active_durations.get(guild_id, DEFAULT_BUFFER_SECONDS)

    save_local = active_save_local.get(guild_id, DEFAULT_SAVE_LOCALLY)

    # Store origin text channel for minutes dispatch
    active_text_channels[guild_id] = ctx.channel

    # Trained ONNX Wake-Word Spotter (Frozen Speech Embeddings + Custom BiGRU Classifier)
    spotter = KeywordSpotter(
        model_path=CLASSIFIER_MODEL_PATH,
        mel_model_path=MEL_MODEL_PATH,
        emb_model_path=EMB_MODEL_PATH,
        config_path=CLASSIFIER_CONFIG_PATH,
        cooldown_seconds=8.0,
        trigger_callback=make_keyword_callback(guild_id),
    )
    active_spotters[guild_id] = spotter

    # Rolling Audio Sink with 100ms streaming chunk feed to KWS
    sink = SynchronizedRollingSink(
        buffer_seconds=buffer_seconds,
        chunk_callback=spotter.process_chunk,
        chunk_ms=100
    )
    active_sinks[guild_id] = sink

    vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    conn = vc._connection
    dave_ver = getattr(conn, 'dave_protocol_version', 0)
    dave_sess = getattr(conn, 'dave_session', None)
    logger.info(f"Voice Channel: '{channel.name}' | DAVE Version: {dave_ver} | DAVE Ready: {getattr(dave_sess, 'ready', False)} | Buffer: {buffer_seconds}s")
    vc.listen(sink)

    dur_label = format_duration_label(buffer_seconds)
    delivery_note = (
        f"Clips `<= {DISCORD_WAV_MAX_SECONDS}s` (default) are sent directly to #{ctx.channel.name}.\n"
        f"Clips longer than `{DISCORD_WAV_MAX_SECONDS}s` will **not** be sent as audio inside the text channel "
        f"due to Discord upload limits — the `.wav` and `.txt` will be saved locally instead (with `.txt` attached in chat)."
    )
    save_status = "ON (always)" if save_local else f"Auto when > {DISCORD_WAV_MAX_SECONDS}s"

    embed = discord.Embed(
        title="🎙️ Meeting Voice Clipper Active",
        description=(
            f"Bot is listening in voice channel `{channel.name}`.\n\n"
            f"**Settings:**\n"
            f"• Rolling Window: **`{dur_label}`** (change anytime via `!duration <sec|Xm>`)\n"
            f"• Local Save: **`{save_status}`** (toggle via `!savelocal on|off`)\n\n"
            f"**How to Capture Audio Clips:**\n"
            f"• Say the wake word: **`\"bot clip that\"`**\n"
            f"• Or use text command: **`!clip`**\n\n"
            f"ℹ️ {delivery_note}"
        ),
        color=discord.Color.blurple()
    )
    await ctx.send(embed=embed)

@bot.command(name="duration", aliases=["buffer"], help="Gets or sets rolling clip duration (e.g. '!duration 30', '!duration 5m', '!duration 30m').")
async def set_duration(ctx, duration: str = None):
    guild_id = ctx.guild.id
    if not duration:
        current = active_durations.get(guild_id, DEFAULT_BUFFER_SECONDS)
        await ctx.send(
            f"⏱️ Current rolling buffer duration: **{format_duration_label(current)}**.\n"
            f"Use `!duration 30` (default, sends `.wav` in Discord) or `!duration 30m` (clips `> {DISCORD_WAV_MAX_SECONDS}s` save `.wav` locally and attach `.txt` here due to upload limits)."
        )
        return

    try:
        seconds = parse_duration_seconds(duration)
    except ValueError as e:
        await ctx.send(f"❌ {e}")
        return

    active_durations[guild_id] = seconds
    if guild_id in active_sinks:
        active_sinks[guild_id].set_buffer_seconds(seconds)

    mode_note = (
        f"Clips `<= {DISCORD_WAV_MAX_SECONDS}s` will be sent directly inside this text channel."
        if seconds <= DISCORD_WAV_MAX_SECONDS
        else f"Because this is longer than `{DISCORD_WAV_MAX_SECONDS}s`, audio `.wav` files will **not** be uploaded to Discord (upload limit) and will be saved locally instead (with `.txt` transcript attached here)."
    )
    await ctx.send(f"✅ Rolling clip duration set to **{format_duration_label(seconds)}**. {mode_note}")


@bot.command(name="savelocal", aliases=["save"], help="Toggles saving clips locally even when duration <= 30s ('on' or 'off').")
async def set_save_local(ctx, mode: str = None):
    guild_id = ctx.guild.id
    if not mode:
        current = active_save_local.get(guild_id, DEFAULT_SAVE_LOCALLY)
        state = "ON" if current else f"OFF (still auto-saves when duration > {DISCORD_WAV_MAX_SECONDS}s)"
        await ctx.send(f"💾 Local clip saving is currently **{state}**. Use `!savelocal on` or `!savelocal off`.")
        return

    val = mode.strip().lower()
    if val not in {"on", "off", "true", "false"}:
        await ctx.send("❌ Invalid option! Use `!savelocal on` or `!savelocal off`.")
        return

    enabled = val in {"on", "true"}
    active_save_local[guild_id] = enabled
    if enabled:
        await ctx.send(f"✅ Local saving **enabled** — clips will be saved under `recordings/` in addition to Discord output.")
    else:
        await ctx.send(
            f"✅ Local saving **disabled** for `<= {DISCORD_WAV_MAX_SECONDS}s` clips "
            f"(clips `> {DISCORD_WAV_MAX_SECONDS}s` still save locally automatically)."
        )


@bot.command(name="clip", help="Manually freeze and transcribe the configured rolling audio buffer.")
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
        if guild_id in active_durations:
            del active_durations[guild_id]
        if guild_id in active_save_local:
            del active_save_local[guild_id]
        await ctx.send("👋 Bot left the voice channel and cleared all audio buffers.")
        logger.info(f"Disconnected from voice channel in guild {guild_id}.")
    else:
        await ctx.send("Bot is not in a voice channel.")


async def _graceful_shutdown():
    """Disconnects all active voice clients, clears buffers, and closes the Discord bot."""
    logger.info("Graceful shutdown requested; disconnecting Discord client...")
    for vc in list(bot.voice_clients):
        try:
            await vc.disconnect(force=True)
        except Exception as e:
            logger.debug(f"Error disconnecting voice client during shutdown: {e}")

    for sink in list(active_sinks.values()):
        try:
            sink.cleanup()
        except Exception:
            pass
    active_sinks.clear()
    active_spotters.clear()
    active_text_channels.clear()
    active_languages.clear()
    active_durations.clear()
    active_save_local.clear()

    if not bot.is_closed():
        try:
            await bot.change_presence(status=discord.Status.offline)
        except Exception as e:
            logger.debug(f"Error setting offline presence during shutdown: {e}")
        await bot.close()


def _start_gui_control_listener():
    """Listens on stdin for 'STOP' (or EOF) when launched via gui.py --gui-managed."""
    def _listen():
        try:
            for raw in sys.stdin:
                if raw.strip().upper() == "STOP":
                    break
        except Exception:
            pass

        loop = getattr(bot, "loop", None)
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(_graceful_shutdown(), loop)
        else:
            os._exit(0)

    threading.Thread(target=_listen, name="GUIControlListener", daemon=True).start()


if __name__ == "__main__":
    import sys
    if not TOKEN:
        print("WARNING: DISCORD_TOKEN is missing in .env file!")
    if "--gui-managed" in sys.argv:
        _start_gui_control_listener()
    bot.run(TOKEN)