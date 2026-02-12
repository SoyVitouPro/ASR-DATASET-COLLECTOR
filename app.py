import os
import hashlib
import sqlite3
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Request, Form, BackgroundTasks, UploadFile, File
from fastapi import Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse, FileResponse, Response
from fastapi import Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Heavy deps used inside background task
import yt_dlp  # type: ignore
from pydub import AudioSegment  # type: ignore
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
    _TORCH_ERROR = ""
except Exception as _torch_err:  # noqa: F841
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    _TORCH_ERROR = str(_torch_err)

try:
    import torchaudio  # type: ignore
    _TORCHAUDIO_AVAILABLE = True
    _TORCHAUDIO_ERROR = ""
except Exception as _ta_err:  # noqa: F841
    torchaudio = None  # type: ignore
    _TORCHAUDIO_AVAILABLE = False
    _TORCHAUDIO_ERROR = str(_ta_err)
import re
import requests
import base64
import secrets
import time
import random
import hmac
import wave
import contextlib


# ---- Paths & App Setup ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "ChlatAudio"

AUDIO_ROOT = os.path.join(BASE_DIR, "audio")
SUB_AUDIO_ROOT = os.path.join(BASE_DIR, "sub_audio")
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")
DB_PATH = os.path.join(BASE_DIR, "chlat_audio.db")

os.makedirs(AUDIO_ROOT, exist_ok=True)
os.makedirs(SUB_AUDIO_ROOT, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = FastAPI(title=APP_NAME)

# Serve generated audio files
app.mount("/sub_audio", StaticFiles(directory=SUB_AUDIO_ROOT), name="sub_audio")

templates = Jinja2Templates(directory=TEMPLATE_DIR)


"""Auth: allow either Basic header or signed session cookie, with a login page."""
BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER", "test")
BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS", "test123")
# Lock admin password for unlocking protected records
ADMIN_LOCK_PASSWORD = os.environ.get("ADMIN_LOCK_PASSWORD", "admintest123")
AUTH_SECRET = os.environ.get("AUTH_SECRET", "please-change-secret")
SESSION_COOKIE = "auth_session"
SESSION_TTL_SECS = 7 * 24 * 3600


def _sign(value: str) -> str:
    mac = hmac.new(AUTH_SECRET.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return mac


def _make_session(username: str) -> str:
    exp = int(time.time()) + SESSION_TTL_SECS
    payload = f"{username}:{exp}"
    sig = _sign(payload)
    return f"{payload}:{sig}"


def _verify_session(token: str) -> bool:
    try:
        parts = token.split(":")
        if len(parts) < 3:
            return False
        username = parts[0]
        exp = int(parts[1])
        sig = parts[2]
        if exp < int(time.time()):
            return False
        payload = f"{username}:{exp}"
        return secrets.compare_digest(sig, _sign(payload))
    except Exception:
        return False


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path or "/"
    # Allow login endpoints without auth
    if path.startswith("/login"):
        return await call_next(request)

    # Cookie session check
    cookie = request.cookies.get(SESSION_COOKIE)
    if cookie and _verify_session(cookie):
        return await call_next(request)

    # Fallback to Basic header support
    header = request.headers.get("authorization")
    if header and header.lower().startswith("basic "):
        try:
            encoded = header.split(" ", 1)[1]
            decoded = base64.b64decode(encoded).decode("utf-8")
            if ":" in decoded:
                user, pwd = decoded.split(":", 1)
                if secrets.compare_digest(user, BASIC_AUTH_USER) and secrets.compare_digest(pwd, BASIC_AUTH_PASS):
                    return await call_next(request)
        except Exception:
            pass

    # Redirect to login UI
    next_url = request.url.path
    return RedirectResponse(url=f"/login?next={next_url}", status_code=302)


# ---- Database ----
def get_db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS audios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                transcript TEXT DEFAULT NULL,
                gender TEXT DEFAULT NULL,
                youtube_channel TEXT DEFAULT NULL,
                verify INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Ensure new column `language` exists (backward-compatible migration)
        cur.execute("PRAGMA table_info(audios)")
        cols = [row[1] for row in cur.fetchall()]
        if "language" not in cols:
            cur.execute("ALTER TABLE audios ADD COLUMN language TEXT DEFAULT NULL")
        # Ensure new column `speaker` exists
        cur.execute("PRAGMA table_info(audios)")
        cols = [row[1] for row in cur.fetchall()]
        if "speaker" not in cols:
            cur.execute("ALTER TABLE audios ADD COLUMN speaker TEXT DEFAULT NULL")
        # Ensure new column `duration_seconds` exists
        cur.execute("PRAGMA table_info(audios)")
        cols = [row[1] for row in cur.fetchall()]
        if "duration_seconds" not in cols:
            cur.execute("ALTER TABLE audios ADD COLUMN duration_seconds INTEGER DEFAULT NULL")
        # Ensure lock column exists
        cur.execute("PRAGMA table_info(audios)")
        cols = [row[1] for row in cur.fetchall()]
        if "locked" not in cols:
            cur.execute("ALTER TABLE audios ADD COLUMN locked INTEGER DEFAULT 0")
        # Backfill NULL to 0 to ensure consistent boolean behavior
        try:
            cur.execute("UPDATE audios SET locked = 0 WHERE locked IS NULL")
        except Exception:
            pass
        conn.commit()


def insert_audio_rows(rows: List[tuple]):
    # rows: (video_url, audio_path, transcript, gender, youtube_channel, verify, duration_seconds)
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO audios (video_url, audio_path, transcript, gender, youtube_channel, verify, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def latest_audios(limit: int = 5):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, video_url, audio_path, transcript, gender, youtube_channel, verify, locked, created_at, language, speaker, duration_seconds FROM audios ORDER BY datetime(created_at) DESC, id ASC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


def video_url_exists(url: str) -> bool:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM audios WHERE video_url = ? LIMIT 1", (url,))
        return cur.fetchone() is not None


def get_audio_count() -> int:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM audios")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

def get_verified_count() -> int:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM audios WHERE verify = 1")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

def get_all_channels():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT youtube_channel FROM audios WHERE youtube_channel IS NOT NULL AND youtube_channel <> '' ORDER BY youtube_channel"
        )
        rows = cur.fetchall()
        return [r[0] for r in rows]

def count_audios(verify: Optional[str] = None, channel: Optional[str] = None, q: Optional[str] = None, label: Optional[str] = None) -> int:
    """Return count of audios matching optional filters.
    - verify: 'done' for verify=1, 'none' for verify=0/NULL, anything else for all
    - channel: exact match channel name
    - q: substring filter on audio_path (case-insensitive)
    - label: substring filter on transcript (case-insensitive)
    """
    with get_db_conn() as conn:
        cur = conn.cursor()
        sql = "SELECT COUNT(*) FROM audios WHERE 1=1"
        params: list = []
        if verify == "done":
            sql += " AND verify = 1"
        elif verify == "none":
            sql += " AND (verify = 0 OR verify IS NULL)"
        elif verify == "done_unlocked":
            sql += " AND verify = 1 AND (locked = 0 OR locked IS NULL)"
        if channel and channel != "all":
            sql += " AND youtube_channel = ?"
            params.append(channel)
        if q:
            sql += " AND LOWER(audio_path) LIKE ?"
            params.append(f"%{q.lower()}%")
        if label:
            sql += " AND LOWER(transcript) LIKE ?"
            params.append(f"%{label.lower()}%")
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def paged_audios(offset: int = 0, limit: int = 5, verify: Optional[str] = None, channel: Optional[str] = None, q: Optional[str] = None, lock: Optional[str] = None, label: Optional[str] = None):
    with get_db_conn() as conn:
        cur = conn.cursor()
        sql = (
            "SELECT id, video_url, audio_path, transcript, gender, youtube_channel, verify, locked, created_at, language, duration_seconds "
            "FROM audios WHERE 1=1"
        )
        params: list = []
        if verify == "done":
            sql += " AND verify = 1"
        elif verify == "none":
            sql += " AND (verify = 0 OR verify IS NULL)"
        elif verify == "done_unlocked":
            sql += " AND verify = 1 AND (locked = 0 OR locked IS NULL)"
        if channel and channel != "all":
            sql += " AND youtube_channel = ?"
            params.append(channel)
        if q:
            sql += " AND LOWER(audio_path) LIKE ?"
            params.append(f"%{q.lower()}%")
        if label:
            sql += " AND LOWER(transcript) LIKE ?"
            params.append(f"%{label.lower()}%")
        if lock == "locked":
            sql += " AND locked = 1"
        elif lock == "unlocked":
            sql += " AND (locked = 0 OR locked IS NULL)"
        sql += " ORDER BY datetime(created_at) DESC, id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cur.execute(sql, tuple(params))
        return cur.fetchall()


# ---- Utils ----
def slugify_url(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"url_{h}"


# ---- Lock helpers ----
def is_locked_by_id(rec_id: int) -> bool:
    try:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT locked FROM audios WHERE id = ? LIMIT 1", (int(rec_id),))
            row = cur.fetchone()
            return bool(row[0]) if row else False
    except Exception:
        return False


def is_locked_by_filename(filename: str) -> bool:
    try:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT locked FROM audios WHERE audio_path LIKE ? ORDER BY id DESC LIMIT 1", (f"%/{filename}",))
            row = cur.fetchone()
            return bool(row[0]) if row else False
    except Exception:
        return False


def _locked_error():
    return JSONResponse({"error": "record is locked"}, status_code=423)


def download_audio_to_wav(url: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    temp_template = os.path.join(out_dir, "temp_audio.%(ext)s")

    # Optional runtime configuration via environment variables to help with 403s
    cookiefile = os.environ.get("YTDLP_COOKIES_FILE")
    proxy = os.environ.get("YTDLP_PROXY")
    user_agent = os.environ.get("YTDLP_USER_AGENT")
    referer = os.environ.get("YTDLP_REFERER")
    headers_json = os.environ.get("YTDLP_HEADERS_JSON")
    extractor_args_json = os.environ.get("YTDLP_EXTRACTOR_ARGS_JSON")
    force_ipv4 = os.environ.get("YTDLP_FORCE_IPV4", "").lower() in {"1", "true", "yes"}
    format_override = os.environ.get("YTDLP_FORMAT")
    # Skip downloads for very long videos if configured (minutes)
    max_duration_min_env = os.environ.get("YTDLP_MAX_DURATION_MIN")
    try:
        max_duration_min = float(max_duration_min_env) if max_duration_min_env else None
    except Exception:
        max_duration_min = None
    # Optional fine-tuning for SABR/403 behavior
    http_chunk_size_env = os.environ.get("YTDLP_HTTP_CHUNK_SIZE")  # e.g. "5M" or bytes
    def _parse_size(v: str):
        try:
            s = v.strip().upper()
            if s.endswith("M"):
                return int(float(s[:-1]) * 1024 * 1024)
            if s.endswith("K"):
                return int(float(s[:-1]) * 1024)
            return int(s)
        except Exception:
            return None
    http_chunk_size = _parse_size(http_chunk_size_env) if http_chunk_size_env else None
    external_downloader = os.environ.get("YTDLP_EXTERNAL_DOWNLOADER")  # e.g. "aria2c"
    external_downloader_args = os.environ.get("YTDLP_EXTERNAL_DOWNLOADER_ARGS")  # e.g. "-x16 -s16 -k1M"

    # Build headers
    http_headers = {}
    if headers_json:
        try:
            http_headers.update(__import__("json").loads(headers_json))
        except Exception:
            pass
    if user_agent:
        http_headers.setdefault("User-Agent", user_agent)
    if referer:
        http_headers.setdefault("Referer", referer)

    # Build extractor args
    extractor_args = None
    if extractor_args_json:
        try:
            extractor_args = __import__("json").loads(extractor_args_json)
        except Exception:
            extractor_args = None

    # Base options
    base_opts = {
        "format": format_override or "bestaudio/best",
        "outtmpl": temp_template,
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
        "nocheckcertificate": True,
        "concurrent_fragment_downloads": 1,
        # Smaller chunks can help some CDNs avoid 403/timeouts
        "http_chunk_size": http_chunk_size if http_chunk_size else 10 * 1024 * 1024,
        "skip_unavailable_fragments": True,
        "fragment_retries": 3,
        "continuedl": True,
    }
    if external_downloader:
        base_opts["external_downloader"] = external_downloader
        if external_downloader_args:
            # yt-dlp accepts a string or dict/list here; keep as string to preserve spacing
            base_opts["external_downloader_args"] = {external_downloader: external_downloader_args}
    if http_headers:
        base_opts["http_headers"] = http_headers
    if cookiefile and os.path.exists(cookiefile):
        base_opts["cookiefile"] = cookiefile
    if proxy:
        base_opts["proxy"] = proxy
    if extractor_args:
        base_opts["extractor_args"] = extractor_args
    if force_ipv4:
        base_opts["source_address"] = "0.0.0.0"

    # Preflight: extract info to check duration and choose site-specific tweaks
    info_dict = None
    try:
        pre_opts = dict(base_opts)
        # prevent accidental file writes during preflight
        pre_opts.pop("outtmpl", None)
        with yt_dlp.YoutubeDL(pre_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
    except Exception:
        info_dict = None

    if max_duration_min and info_dict and isinstance(info_dict, dict):
        duration = info_dict.get("duration")  # seconds
        if duration and (duration / 60.0) > max_duration_min:
            raise RuntimeError(f"Video too long: {duration/60.0:.1f} min exceeds limit {max_duration_min} min")

    # Prefer YouTube web client to avoid PO-token warnings
    if info_dict and isinstance(info_dict, dict):
        extractor_key = (info_dict.get("extractor_key") or "").lower()
        if "youtube" in extractor_key:
            yargs = base_opts.get("extractor_args") or {}
            yargs.setdefault("youtube", {})
            # use web client first; avoid android unless explicitly set via env
            if not (extractor_args and "youtube" in extractor_args and extractor_args["youtube"].get("player_client")):
                yargs["youtube"]["player_client"] = ["web"]
            base_opts["extractor_args"] = yargs
            # Auto add Referer/Origin to stabilize SABR-requested streams
            try:
                vid_id = info_dict.get("id")
                if isinstance(vid_id, str) and vid_id:
                    if "http_headers" not in base_opts:
                        base_opts["http_headers"] = {}
                    base_opts["http_headers"].setdefault("Referer", f"https://www.youtube.com/watch?v={vid_id}")
                    base_opts["http_headers"].setdefault("Origin", "https://www.youtube.com")
                    base_opts["http_headers"].setdefault("Accept-Language", "en-US,en;q=0.9")
            except Exception:
                pass

    # Try primary download
    def _do_download(opts):
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    try:
        _do_download(base_opts)
    except Exception as e:
        # If YouTube + 403/SABR, try alternative clients to bypass
        msg = str(e)
        is_403 = "403" in msg or "Forbidden" in msg
        # Only branch to multi-client fallback for YouTube
        is_yt = False
        if info_dict and isinstance(info_dict, dict):
            ek = (info_dict.get("extractor_key") or "").lower()
            is_yt = "youtube" in ek
        if not (is_403 and is_yt):
            raise

        client_list_env = os.environ.get("YTDLP_YT_PLAYER_CLIENTS")
        candidates = (
            [c.strip() for c in client_list_env.split(",") if c.strip()]
            if client_list_env else ["web", "ios", "tv_embedded", "mweb", "web_creator"]
        )

        last_err = e
        for client in candidates:
            try:
                retry_opts = dict(base_opts)
                headers = dict(retry_opts.get("http_headers", {}))
                headers.setdefault(
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                )
                headers.setdefault("Accept-Language", "en-US,en;q=0.9")
                retry_opts["http_headers"] = headers
                retry_opts["source_address"] = "0.0.0.0"
                yt_args = retry_opts.get("extractor_args") or {}
                yt_args.setdefault("youtube", {})
                yt_args["youtube"]["player_client"] = [client]
                retry_opts["extractor_args"] = yt_args

                # Preflight to ensure at least one audio fmt has URL
                pre2 = dict(retry_opts)
                pre2.pop("outtmpl", None)
                try:
                    with yt_dlp.YoutubeDL(pre2) as ydl:
                        inf2 = ydl.extract_info(url, download=False)
                    fmts = (inf2 or {}).get("formats") or []
                    if not any((f.get("acodec") and f.get("url")) for f in fmts if isinstance(f, dict)):
                        last_err = RuntimeError(f"youtube client '{client}' has no usable formats")
                        continue
                except Exception as pe:
                    last_err = pe
                    continue

                _do_download(retry_opts)
                last_err = None
                break
            except Exception as ex:
                last_err = ex
                continue
        if last_err:
            raise last_err

    # Find downloaded file
    downloaded_file = None
    for f in os.listdir(out_dir):
        if f.startswith("temp_audio") and f.endswith((".m4a", ".webm", ".mp3", ".opus", ".wav")):
            downloaded_file = os.path.join(out_dir, f)
            break

    if not downloaded_file:
        raise RuntimeError("Download failed: no audio file found.")

    # Convert to WAV if needed
    output_wav = os.path.join(out_dir, "output.wav")
    if not downloaded_file.endswith(".wav"):
        audio = AudioSegment.from_file(downloaded_file)
        audio.export(output_wav, format="wav")
        try:
            os.remove(downloaded_file)
        except Exception:
            pass
    else:
        # Ensure consistent filename
        audio = AudioSegment.from_wav(downloaded_file)
        audio.export(output_wav, format="wav")
        if os.path.abspath(downloaded_file) != os.path.abspath(output_wav):
            try:
                os.remove(downloaded_file)
            except Exception:
                pass

    return output_wav


def _find_quiet_cut_ms(
    audio: AudioSegment,
    target_ms: int,
    search_window_ms: int = 600,
    step_ms: int = 10,
    window_ms: int = 30,
    prefer_after: bool = True,
) -> int:
    """Find a cut point near target_ms where amplitude is minimal.
    Scans a window around target and picks the position with lowest dBFS.

    - prefer_after: when tie, prefer a position after target to avoid very short first segment.
    """
    dur = len(audio)
    if dur <= 0:
        return max(0, min(target_ms, dur))
    start = max(0, target_ms - search_window_ms)
    end = min(dur, target_ms + search_window_ms)
    best_ms = max(0, min(target_ms, dur))
    best_val = None
    # Ensure window at least one step
    if end <= start:
        return best_ms
    # Iterate over candidate centers
    for pos in range(start, end, max(1, step_ms)):
        w0 = max(0, pos - window_ms // 2)
        w1 = min(dur, pos + window_ms // 2)
        seg = audio[w0:w1]
        # dBFS can be -inf for true silence, which is ideal (lower is quieter)
        val = seg.dBFS if len(seg) > 0 else float("-inf")
        if best_val is None or val < best_val or (val == best_val and prefer_after and pos >= target_ms and best_ms < target_ms):
            best_val = val
            best_ms = pos
    return best_ms


def _split_by_quiet_points(
    audio: AudioSegment,
    max_ms: int = 25000,
    min_ms: int = 7000,
    search_window_ms: int = 600,
    step_ms: int = 10,
    window_ms: int = 30,
) -> List[AudioSegment]:
    """Split an AudioSegment into chunks in [min_ms, max_ms] using quiet boundaries.

    Picks a random target duration per chunk in [min_ms, max_ms] and snaps cut
    to a nearby low-amplitude point. Tries to avoid a final tiny remainder by
    reserving at least min_ms for the last piece when possible.
    """
    parts: List[AudioSegment] = []
    total = len(audio)
    if total <= 0:
        return parts
    start = 0
    while True:
        remaining = total - start
        if remaining <= max_ms:
            break
        # Choose a random target ensuring we leave at least min_ms for the tail
        max_target = min(max_ms, max(min_ms, remaining - min_ms))
        target = random.randint(min_ms, max_target) if max_target > min_ms else max_target
        sub = audio[start:]
        rel_cut = _find_quiet_cut_ms(
            sub,
            target_ms=target,
            search_window_ms=search_window_ms,
            step_ms=step_ms,
            window_ms=window_ms,
            prefer_after=True,
        )
        # keep chunk >= min_ms and leave >= min_ms when possible
        if len(sub) > 2 * min_ms:
            rel_cut = max(min_ms, min(rel_cut, len(sub) - min_ms))
        else:
            rel_cut = min(target, len(sub))
        cut = start + rel_cut
        parts.append(audio[start:cut])
        start = cut

    # Handle remainder
    remainder = audio[start:]
    if len(remainder) == 0:
        return parts
    if len(remainder) >= min_ms:
        parts.append(remainder)
        return parts
    # If remainder is shorter than min_ms, try to merge with previous
    if parts:
        last = parts.pop()
        combined = last + remainder
        if len(combined) <= max_ms:
            parts.append(combined)
            return parts
        # Need to split combined into two valid chunks
        max_target = min(max_ms, len(combined) - min_ms)
        target = random.randint(min_ms, max_target)
        cut = _find_quiet_cut_ms(
            combined,
            target_ms=target,
            search_window_ms=search_window_ms,
            step_ms=step_ms,
            window_ms=window_ms,
            prefer_after=True,
        )
        cut = max(min_ms, min(cut, len(combined) - min_ms))
        parts.append(combined[:cut])
        parts.append(combined[cut:])
        return parts
    # No previous parts, just return the small tail
    parts.append(remainder)
    return parts


def split_audio_with_vad(
    input_wav: str,
    out_dir: str,
    threshold: float = 0.5,
    min_ms: int = 7000,
    max_ms: int = 25000,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    if not _TORCH_AVAILABLE or not _TORCHAUDIO_AVAILABLE:
        # Fallback: no torch/torchaudio available; split by quiet points only.
        if not _TORCH_AVAILABLE:
            print(f"[warn] torch unavailable, skipping VAD: {_TORCH_ERROR}")
        if not _TORCHAUDIO_AVAILABLE:
            print(f"[warn] torchaudio unavailable, skipping VAD: {_TORCHAUDIO_ERROR}")
        whole = AudioSegment.from_file(input_wav)
        whole = whole.set_channels(1)
        chunks = _split_by_quiet_points(whole, max_ms=max_ms, min_ms=min_ms)
        saved_paths: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            if len(ch) < min_ms:
                continue
            out_path = os.path.join(out_dir, f"audio_{i}.wav")
            _to_pcm16_16k_mono(ch).export(out_path, format="wav")
            saved_paths.append(out_path)
        return saved_paths

    # Load Silero VAD
    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False)
    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = utils

    wav, sr = torchaudio.load(input_wav)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # to mono

    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=threshold)
    if not speech_timestamps:
        # Fallback: no clear speech; split whole file by quiet points into <=25s
        whole = AudioSegment.from_file(input_wav)
        whole = whole.set_channels(1)
        chunks = _split_by_quiet_points(whole, max_ms=max_ms, min_ms=min_ms)
        saved_paths: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            # Enforce minimum duration; skip too-short chunks
            if len(ch) < min_ms:
                continue
            out_path = os.path.join(out_dir, f"audio_{i}.wav")
            _to_pcm16_16k_mono(ch).export(out_path, format="wav")
            saved_paths.append(out_path)
        return saved_paths

    # Save raw segments first
    temp_segments = []
    for idx, seg in enumerate(speech_timestamps):
        start, end = seg["start"], seg["end"]
        chunk = wav[:, start:end]
        temp_path = os.path.join(out_dir, f"seg_{idx+1}.wav")
        torchaudio.save(temp_path, chunk, sr)
        temp_segments.append(temp_path)

    # Merge groups of three to create longer context, then enforce <=25s using quiet cuts
    saved_paths: List[str] = []
    seq = 0
    for i in range(0, len(temp_segments), 3):
        group = temp_segments[i:i+3]
        combined: Optional[AudioSegment] = None
        for fp in group:
            seg = AudioSegment.from_wav(fp)
            seg = seg.set_channels(1)
            combined = seg if combined is None else (combined + seg)
        if combined is None:
            continue
        # Enforce randomized [min_ms, max_ms] chunks by cutting at low-amplitude points
        chunks = _split_by_quiet_points(combined, max_ms=max_ms, min_ms=min_ms)
        for ch in chunks:
            # Enforce minimum duration; skip too-short chunks
            if len(ch) < min_ms:
                continue
            seq += 1
            out_path = os.path.join(out_dir, f"audio_{seq}.wav")
            # Normalize export to PCM 16k mono for better ASR consistency
            _to_pcm16_16k_mono(ch).export(out_path, format="wav")
            saved_paths.append(out_path)

    # Cleanup temp segments
    for p in temp_segments:
        try:
            os.remove(p)
        except Exception:
            pass

    return saved_paths


def wav_duration_seconds(path: str) -> Optional[int]:
    """Return duration in whole seconds for a WAV file path."""
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate and frames:
                secs = frames / float(rate)
                return int(round(secs))
    except Exception:
        return None
    return None


def human_duration(secs: Optional[int]) -> str:
    if secs is None:
        return "0:00"
    m, s = divmod(int(secs), 60)
    return f"{m}:{s:02d}"


from uuid import uuid4
from threading import Lock

JOBS = {}
JOBS_LOCK = Lock()


def set_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        job = JOBS.get(job_id, {})
        job.update(kwargs)
        JOBS[job_id] = job


def process_url(
    url: str,
    youtube_channel: Optional[str] = None,
    job_id: Optional[str] = None,
    min_ms: int = 7000,
    max_ms: int = 25000,
):
    slug = slugify_url(url)
    url_audio_dir = os.path.join(AUDIO_ROOT, slug)
    url_sub_audio_dir = os.path.join(SUB_AUDIO_ROOT, slug)
    try:
        if job_id:
            set_job(job_id, status="downloading")
        # 1) Download to WAV
        wav_path = download_audio_to_wav(url, url_audio_dir)

        if job_id:
            set_job(job_id, status="splitting")
        # 2) Split into segments (merged pairs) in sub_audio/<slug>
        parts = split_audio_with_vad(
            wav_path,
            url_sub_audio_dir,
            threshold=0.5,
            min_ms=min_ms,
            max_ms=max_ms,
        )

        # 3) Insert rows for each split audio
        rows = []
        for p in parts:
            dur = wav_duration_seconds(p)
            rows.append((url, p, None, None, youtube_channel, 0, dur))

        if rows:
            insert_audio_rows(rows)
        if job_id:
            set_job(job_id, status="done", count=len(rows), min_ms=min_ms, max_ms=max_ms)
    except Exception as e:
        if job_id:
            set_job(job_id, status="error", error=str(e))
        else:
            raise


# ---- Routes ----
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/favicon.png")
def favicon_png():
    path = os.path.join(TEMPLATE_DIR, "chlataudio.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "favicon not found"}, status_code=404)


@app.get("/favicon.ico")
def favicon_ico():
    # serve the same PNG for ICO requests for broad browser support
    path = os.path.join(TEMPLATE_DIR, "chlataudio.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "favicon not found"}, status_code=404)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Fetch latest 5
    recs = latest_audios(limit=5)
    total_count = get_audio_count()
    verified_count = get_verified_count()
    # Adapt for template: compute public path under /sub_audio
    items = []
    for r in recs:
        (
            id_,
            video_url,
            audio_path,
            transcript,
            gender,
            youtube_channel,
            verify,
            locked,
            created_at,
            language,
            speaker,
            duration_seconds,
        ) = r
        rel = os.path.relpath(audio_path, SUB_AUDIO_ROOT)
        public_url = f"/sub_audio/{rel}"
        # filesize
        try:
            size_bytes = os.path.getsize(audio_path)
        except Exception:
            size_bytes = 0
        size_kb = round(size_bytes / 1024, 2)
        filename = os.path.basename(audio_path)
        items.append(
            {
                "id": id_,
                "video_url": video_url,
                "audio_path": audio_path,
                "public_url": public_url,
                "transcript": transcript,
                "gender": gender,
                "youtube_channel": youtube_channel,
                "verify": bool(verify),
                "locked": bool(locked),
                "created_at": created_at,
                "language": language,
                "speaker": speaker,
                "size_kb": size_kb,
                "filename": filename,
                "duration_seconds": duration_seconds,
                "duration_human": human_duration(duration_seconds),
                }
        )
    all_channels = get_all_channels()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "latest": items,
            "channels": all_channels,
            "audio_count": total_count,
            "verified_count": verified_count,
        },
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: Optional[str] = None):
    return templates.TemplateResponse("login.html", {"request": request, "next": next or "/", "app_name": APP_NAME})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: Optional[str] = Form("/")):
    if secrets.compare_digest(username, BASIC_AUTH_USER) and secrets.compare_digest(password, BASIC_AUTH_PASS):
        token = _make_session(username)
        resp = RedirectResponse(url=next or "/", status_code=302)
        resp.set_cookie(SESSION_COOKIE, token, httponly=True, max_age=SESSION_TTL_SECS, samesite="lax")
        return resp
    # invalid
    return templates.TemplateResponse("login.html", {"request": request, "next": next or "/", "error": "Invalid credentials", "app_name": APP_NAME}, status_code=401)


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/tts", response_class=HTMLResponse)
def tts_page(request: Request):
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "app_name": APP_NAME,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, password: Optional[str] = None):
    # Check admin password
    if not password or not secrets.compare_digest(password, ADMIN_LOCK_PASSWORD):
        return templates.TemplateResponse(
            "admin_login.html",
            {
                "request": request,
                "app_name": APP_NAME,
                "error": None,
            },
        )

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "app_name": APP_NAME,
        },
    )


@app.post("/admin", response_class=HTMLResponse)
def admin_login_submit(request: Request, password: str = Form(...)):
    # Check admin password
    if not secrets.compare_digest(password, ADMIN_LOCK_PASSWORD):
        return templates.TemplateResponse(
            "admin_login.html",
            {
                "request": request,
                "app_name": APP_NAME,
                "error": "Invalid admin password",
            },
        )

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "app_name": APP_NAME,
        },
    )


@app.post("/process")
def start_process(
    background_tasks: BackgroundTasks,
    video_url: str = Form(...),
    youtube_channel: Optional[str] = Form(None),
    min_seconds: Optional[int] = Form(None),
    max_seconds: Optional[int] = Form(None),
):
    # Reject if URL already processed
    if video_url_exists(video_url):
        return JSONResponse({"error": "This URL has already been processed."}, status_code=409)

    # Reject if a job is already running/queued for same URL
    with JOBS_LOCK:
        for j_id, j in JOBS.items():
            if j.get("url") == video_url and j.get("status") in {"queued", "downloading", "splitting"}:
                return JSONResponse({"error": "A job for this URL is already in progress."}, status_code=409)

    # Clamp and validate ranges (defaults 7..25 seconds)
    try:
        min_s = int(min_seconds) if min_seconds is not None else 7
    except Exception:
        min_s = 7
    try:
        max_s = int(max_seconds) if max_seconds is not None else 25
    except Exception:
        max_s = 25
    # Enforce bounds 3..60, but then clamp to 7..25 as requested
    min_s = max(1, min(min_s, 60))
    max_s = max(1, min(max_s, 60))
    if min_s > max_s:
        min_s, max_s = max_s, min_s
    # Final clamp to 7..25 seconds window per request
    min_s = max(7, min(min_s, 25))
    max_s = max(7, min(max_s, 25))

    job_id = uuid4().hex
    set_job(job_id, status="queued", submitted=datetime.utcnow().isoformat(), url=video_url, min_seconds=min_s, max_seconds=max_s)
    background_tasks.add_task(process_url, video_url, youtube_channel, job_id, min_s * 1000, max_s * 1000)
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/status/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        data = JOBS.get(job_id)
    if not data:
        return JSONResponse({"job_id": job_id, "status": "unknown"}, status_code=404)
    return JSONResponse({"job_id": job_id, **data})


@app.get("/list")
def list_audios(
    offset: int = 0,
    limit: int = 5,
    verify: Optional[str] = None,
    channel: Optional[str] = None,
    q: Optional[str] = None,
    lock_state: Optional[str] = Query(None, alias="lock"),
    label: Optional[str] = None,
):
    recs = paged_audios(offset=offset, limit=limit, verify=verify, channel=channel if channel != "all" else None, q=q, lock=lock_state, label=label)
    items = []
    for r in recs:
        (
            id_,
            video_url,
            audio_path,
            transcript,
            gender,
            youtube_channel,
            verify,
            locked,
            created_at,
            language,
            duration_seconds,
        ) = r
        rel = os.path.relpath(audio_path, SUB_AUDIO_ROOT)
        public_url = f"/sub_audio/{rel}"
        filename = os.path.basename(audio_path)
        items.append(
            {
                "id": id_,
                "video_url": video_url,
                "audio_path": audio_path,
                "public_url": public_url,
                "transcript": transcript,
                "gender": gender,
                "youtube_channel": youtube_channel,
                "verify": bool(verify),
                "locked": bool(locked),
                "created_at": created_at,
                "language": language,
                "filename": filename,
                "duration_seconds": duration_seconds,
                "duration_human": human_duration(duration_seconds),
            }
        )
    # Build totals for UI stats
    # Global totals (ignoring q and label):
    total_all = get_audio_count()
    verified_all = get_verified_count()
    # Channel totals (respect channel + search q + label to reflect current view)
    ch_val = channel if channel and channel != "all" else None
    total_channel = count_audios(channel=ch_val, q=q, label=label)
    verified_channel = count_audios(verify="done", channel=ch_val, q=q, label=label)
    left_channel = max(0, total_channel - verified_channel)
    return {
        "items": items,
        "channels": get_all_channels(),
        "totals": {
            "total_all": total_all,
            "verified_all": verified_all,
            "total_channel": total_channel,
            "verified_channel": verified_channel,
            "left_channel": left_channel,
        },
    }


@app.post("/lock")
def lock_toggle(payload: dict = Body(...)):
    """Lock or unlock a record. Unlocking requires admin password.
    Payload: { id?: int, filename?: str, locked: bool, password?: str }
    """
    target_locked = bool(payload.get("locked"))
    rec_id = payload.get("id")
    filename = payload.get("filename")
    password = payload.get("password") or ""
    # Ensure schema has locked column
    try:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(audios)")
            cols = [row[1] for row in cur.fetchall()]
            if "locked" not in cols:
                cur.execute("ALTER TABLE audios ADD COLUMN locked INTEGER DEFAULT 0")
                try:
                    cur.execute("UPDATE audios SET locked = 0 WHERE locked IS NULL")
                except Exception:
                    pass
                conn.commit()
    except Exception:
        pass
    # Require password for both lock and unlock
    if password != ADMIN_LOCK_PASSWORD:
        return JSONResponse({"error": "invalid password"}, status_code=403)
    # Resolve target id
    rec_id_int: Optional[int] = None
    if rec_id is not None:
        try:
            rec_id_int = int(rec_id)
        except Exception:
            return JSONResponse({"error": "invalid id"}, status_code=400)
    elif filename:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM audios WHERE audio_path LIKE ? ORDER BY id DESC LIMIT 1", (f"%/{filename}",))
            row = cur.fetchone()
            if row:
                rec_id_int = int(row[0])
    if rec_id_int is None:
        return JSONResponse({"error": "id or filename required"}, status_code=400)
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE audios SET locked = ? WHERE id = ?", (1 if target_locked else 0, rec_id_int))
        conn.commit()
    return {"id": rec_id_int, "locked": target_locked, "updated": 1}


@app.post("/re-calculate-time")
def recalculate_time(force: Optional[bool] = False):
    """Recalculate and store duration_seconds for all records.
    If force=false, only fills missing/null values.
    """
    updated = 0
    with get_db_conn() as conn:
        cur = conn.cursor()
        if force:
            cur.execute("SELECT id, audio_path FROM audios")
        else:
            cur.execute("SELECT id, audio_path FROM audios WHERE duration_seconds IS NULL")
        rows = cur.fetchall()
        for rec_id, apath in rows:
            try:
                if not apath or not os.path.exists(apath):
                    continue
                dur = wav_duration_seconds(apath)
                if dur is None and force is False:
                    continue
                cur.execute("UPDATE audios SET duration_seconds = ? WHERE id = ?", (dur, rec_id))
                updated += 1
            except Exception:
                continue
        conn.commit()
    return {"updated": updated}


# ---- Update endpoints for table actions ----
def _update_by_filename(filename: str, field: str, value):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE audios SET {} = ? WHERE audio_path LIKE ?".format(field), (value, f"%/{filename}"))
        conn.commit()
        return cur.rowcount

def _update_by_id(rec_id: int, field: str, value):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE audios SET {} = ? WHERE id = ?".format(field), (value, rec_id))
        conn.commit()
        return cur.rowcount

def _delete_by_filename(filename: str) -> int:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM audios WHERE audio_path LIKE ?", (f"%/{filename}",))
        conn.commit()
        return cur.rowcount


@app.post("/verify")
def set_verify(payload: dict = Body(...)):
    verified = bool(payload.get("verified"))
    rec_id = payload.get("id")
    filename = payload.get("filename")
    # Prefer id to avoid basename collisions across different videos
    if rec_id is not None:
        try:
            rec_id_int = int(rec_id)
        except Exception:
            return JSONResponse({"error": "invalid id"}, status_code=400)
        if is_locked_by_id(rec_id_int):
            return _locked_error()
        changed = _update_by_id(rec_id_int, "verify", 1 if verified else 0)
        return {"id": rec_id_int, "verified": verified, "updated": changed}
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    if is_locked_by_filename(filename):
        return _locked_error()
    changed = _update_by_filename(filename, "verify", 1 if verified else 0)
    return {"filename": filename, "verified": verified, "updated": changed}


@app.post("/gender")
def set_gender(payload: dict = Body(...)):
    filename = payload.get("filename")
    gender = payload.get("gender")
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    if is_locked_by_filename(filename):
        return _locked_error()
    changed = _update_by_filename(filename, "gender", gender)
    return {"filename": filename, "gender": gender, "updated": changed}


@app.post("/lang")
def set_lang(payload: dict = Body(...)):
    filename = payload.get("filename")
    lang = payload.get("lang")
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    if is_locked_by_filename(filename):
        return _locked_error()
    changed = _update_by_filename(filename, "language", lang)
    return {"filename": filename, "language": lang, "updated": changed}


@app.post("/label")
def set_label(payload: dict = Body(...)):
    filename = payload.get("filename")
    label = payload.get("label")
    rec_id = payload.get("id")
    # Prefer update by id to avoid basename collisions across different videos/channels
    if rec_id is not None:
        try:
            rec_id_int = int(rec_id)
        except Exception:
            return JSONResponse({"error": "invalid id"}, status_code=400)
        if is_locked_by_id(rec_id_int):
            return _locked_error()
        changed = _update_by_id(rec_id_int, "transcript", label)
        return {"id": rec_id_int, "label": label, "updated": changed}
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    if is_locked_by_filename(filename):
        return _locked_error()
    changed = _update_by_filename(filename, "transcript", label)
    return {"filename": filename, "label": label, "updated": changed}


@app.post("/delete")
def delete_record(payload: dict = Body(...)):
    filename = payload.get("filename")
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    if is_locked_by_filename(filename):
        return _locked_error()
    removed = _delete_by_filename(filename)
    return {"filename": filename, "deleted": removed}


@app.post("/split")
def split_file(payload: dict = Body(...)):
    return JSONResponse({"error": "split feature is disabled"}, status_code=403)


# ---- Transcribe wrapper ----
def _get_audio_path_by_filename(filename: str) -> Optional[str]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT audio_path FROM audios WHERE audio_path LIKE ? ORDER BY id DESC LIMIT 1",
            (f"%/{filename}",),
        )
        row = cur.fetchone()
        return row[0] if row else None

def _get_row_by_id(rec_id: int) -> Optional[tuple]:
    """Return (id, video_url, audio_path, youtube_channel, created_at) for a given id."""
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, video_url, audio_path, youtube_channel, created_at FROM audios WHERE id = ? LIMIT 1",
            (rec_id,),
        )
        row = cur.fetchone()
        return row if row else None

def _get_row_meta_by_filename(filename: str) -> Optional[tuple]:
    """Return (id, video_url, audio_path, youtube_channel) for a given filename."""
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, video_url, audio_path, youtube_channel FROM audios WHERE audio_path LIKE ? ORDER BY id DESC LIMIT 1",
            (f"%/{filename}",),
        )
        row = cur.fetchone()
        return row if row else None

def _derive_nearby_child_path(base_path: str) -> str:
    """Return a child path meant to sort near the original by appending 'a' to the stem.
    If it exists, append more 'a' until free: e.g., audio_63.wav -> audio_63a.wav, audio_63aa.wav
    """
    dirn = os.path.dirname(base_path)
    stem, ext = os.path.splitext(os.path.basename(base_path))
    # Do not strip any suffix; always append 'a' for adjacency in lexicographic views
    suffix = "a"
    candidate = os.path.join(dirn, f"{stem}{suffix}{ext}")
    while os.path.exists(candidate):
        suffix += "a"
        candidate = os.path.join(dirn, f"{stem}{suffix}{ext}")
    return candidate

def split_into_two_vad(input_wav: str, position_ms: Optional[int] = None, keep_original_path: bool = False) -> Optional[tuple[str, str]]:
    """Split a wav file into two parts by cutting near a quiet point.
    - If position_ms is provided, cut near that playhead by finding a low-amplitude boundary.
    - Otherwise, cut near the midpoint using a quiet boundary.
    """
    if not os.path.exists(input_wav):
        return None
    try:
        audio = AudioSegment.from_wav(input_wav)
        total_ms = len(audio)
        if total_ms <= 1:
            return None
        # Find a quiet boundary near desired split
        desired = total_ms // 2 if position_ms is None else max(1, min(total_ms - 1, int(position_ms)))
        split_ms = _find_quiet_cut_ms(audio, desired, search_window_ms=800, step_ms=10, window_ms=30, prefer_after=True)
        # Guard against extremes
        split_ms = max(1, min(total_ms - 1, split_ms))
        a1 = audio[:split_ms]
        a2 = audio[split_ms:]
        if keep_original_path:
            part1 = input_wav
            part2 = _derive_nearby_child_path(input_wav)
            _to_pcm16_16k_mono(a1).export(part1, format="wav")
            _to_pcm16_16k_mono(a2).export(part2, format="wav")
            return part1, part2
        else:
            part1 = _derive_nearby_child_path(input_wav)
            part2 = _derive_nearby_child_path(part1)
            _to_pcm16_16k_mono(a1).export(part1, format="wav")
            _to_pcm16_16k_mono(a2).export(part2, format="wav")
            return part1, part2
    except Exception:
        return None


def _derive_trim_path(base_path: str) -> str:
    """Derive a child path for trimming using compact 't' suffix scheme.
    Examples: audio_93.wav -> audio_93_t.wav; audio_93_tt.wav -> audio_93_ttt.wav
    """
    dirn = os.path.dirname(base_path)
    stem, ext = os.path.splitext(os.path.basename(base_path))
    depth = 0
    s = stem
    m = re.search(r"(.*?)(?:_t+)$", s)
    if m:
        base = m.group(1)
        t_run = s[len(base) + 1:]
        depth += len(t_run)
        s = base
    base_stem = s
    run_len = depth + 1
    candidate = os.path.join(dirn, f"{base_stem}_{'t' * run_len}{ext}")
    while os.path.exists(candidate):
        run_len += 1
        candidate = os.path.join(dirn, f"{base_stem}_{'t' * run_len}{ext}")
    return candidate


@app.post("/trim")
def trim_file(payload: dict = Body(...)):
    """Trim the audio to a [start_ms, end_ms] window in-place (overwrite original).
    No new file or DB row is created; duration is updated and verify reset.
    """
    rec_id = payload.get("id")
    filename = payload.get("filename")
    start_ms = int(payload.get("start_ms") or 0)
    end_ms = int(payload.get("end_ms") or 0)
    video_url = ""
    youtube_channel = None
    created_at: Optional[str] = None
    audio_path = None
    # Prefer lookup by id to avoid filename collisions
    if rec_id:
        meta = _get_row_by_id(int(rec_id))
        if meta:
            _, video_url, audio_path, youtube_channel, created_at = meta
        if is_locked_by_id(int(rec_id)):
            return _locked_error()
    if not audio_path and filename:
        if is_locked_by_filename(filename):
            return _locked_error()
        audio_path = _get_audio_path_by_filename(filename)
    if not audio_path or not os.path.exists(audio_path):
        return JSONResponse({"error": "audio not found"}, status_code=404)

    try:
        audio = AudioSegment.from_wav(audio_path)
        dur_ms = len(audio)
        start_ms = max(0, min(start_ms, dur_ms - 1))
        end_ms = max(start_ms + 1, min(end_ms if end_ms > 0 else dur_ms, dur_ms))
        clip = audio[start_ms:end_ms]
        # Overwrite original file with trimmed clip
        _to_pcm16_16k_mono(clip).export(audio_path, format="wav")
        d = wav_duration_seconds(audio_path)
        # Update DB row: duration and verify reset
        with get_db_conn() as conn:
            cur = conn.cursor()
            if rec_id:
                cur.execute("UPDATE audios SET duration_seconds = ?, verify = 0 WHERE id = ?", (d, int(rec_id)))
            else:
                cur.execute("UPDATE audios SET duration_seconds = ?, verify = 0 WHERE audio_path LIKE ?", (d, f"%/{filename}"))
            conn.commit()
        return {"updated": 1, "filename": os.path.basename(audio_path), "duration_seconds": d, "replaced": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _clean_transcript(text: Optional[str]) -> str:
    if not text:
        return ""
    # Clean each line: remove leading timecode and speaker label like "00:01 A:"
    cleaned_lines = []
    for line in str(text).splitlines():
        s = line.strip()
        s = re.sub(r"^\s*\d{1,2}:\d{2}\s+[A-Za-z]+:\s*", "", s)  # 00:01 A:
        s = re.sub(r"^\s*\d{1,2}:\d{2}\s+", "", s)               # 00:01
        s = re.sub(r"^\s*[A-Za-z]+:\s*", "", s)                    # A:
        if s:
            cleaned_lines.append(s)
    # Join lines with space, collapse excessive whitespace
    joined = " ".join(cleaned_lines).strip()
    joined = re.sub(r"\s+", " ", joined)
    return joined


@app.post("/gen")
def gen_transcript(payload: dict = Body(...)):
    """
    Proxy endpoint to call local ASR service and store transcript.
    Expects: { filename: str, prompt?: str }
    """
    filename = payload.get("filename")
    rec_id = payload.get("id")
    prompt = payload.get("prompt") or (
        "Transcribe the audio as plain text in the original language without timestamps or speaker labels. "
        "Return only the transcription text."
    )
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    # Check lock state
    if rec_id is not None:
        try:
            if is_locked_by_id(int(rec_id)):
                return _locked_error()
        except Exception:
            pass
    else:
        if is_locked_by_filename(filename):
            return _locked_error()

    audio_path: Optional[str] = None
    # Prefer lookup by id to avoid basename collisions across different videos/channels
    if rec_id is not None:
        try:
            row = _get_row_by_id(int(rec_id))
            if row:
                # row: (id, video_url, audio_path, youtube_channel, created_at)
                audio_path = row[2]
        except Exception:
            audio_path = None
    if not audio_path and filename:
        audio_path = _get_audio_path_by_filename(filename)
    if not audio_path or not os.path.exists(audio_path):
        return JSONResponse({"error": "audio not found"}, status_code=404)

    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (filename, f, "audio/wav")}
            data = {"prompt": prompt}
            resp = requests.post(
                "http://127.0.0.1:2030/transcribe",
                files=files,
                data=data,
                timeout=300,
            )
        if resp.status_code != 200:
            return JSONResponse({"error": f"transcribe service error: {resp.status_code}"}, status_code=502)
        ct = resp.headers.get("content-type", "").lower()
        raw: Optional[str] = None
        if "application/json" in ct:
            try:
                js = resp.json()
                if isinstance(js, dict):
                    raw = js.get("transcript") or js.get("text") or js.get("result")
                else:
                    raw = str(js)
            except Exception:
                raw = resp.text
        else:
            # text/plain or others -> take body as transcript
            raw = resp.text
        cleaned = _clean_transcript(raw)
        # Do not persist here; UI saves only when verifying
        return {"filename": filename, "transcript": cleaned, "success": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/speaker")
def set_speaker(payload: dict = Body(...)):
    filename = payload.get("filename")
    speaker = payload.get("speaker")
    if not filename:
        return JSONResponse({"error": "filename required"}, status_code=400)
    if is_locked_by_filename(filename):
        return _locked_error()
    changed = _update_by_filename(filename, "speaker", speaker)
    return {"filename": filename, "speaker": speaker, "updated": changed}


@app.post("/clear-channel")
def clear_channel(payload: Optional[dict] = Body(None), channel: Optional[str] = None, password: Optional[str] = None):
    """
    Clear annotations for a specific channel.
    Resets: transcript -> NULL, verify -> 0, language -> NULL, gender -> NULL.
    Accepts either JSON body {"channel": "..."} or query param ?channel=...
    """
    ch = None
    if payload and isinstance(payload, dict):
        ch = payload.get("channel") or payload.get("name") or payload.get("youtube_channel")
        if not password:
            password = payload.get("password")
    if not ch:
        ch = channel
    if not ch or not str(ch).strip():
        return JSONResponse({"error": "channel required"}, status_code=400)

    # Require admin password (not user login password)
    if not password or not secrets.compare_digest(str(password), ADMIN_LOCK_PASSWORD):
        return JSONResponse({"error": "invalid password"}, status_code=403)

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE audios SET transcript = NULL, verify = 0, language = NULL, gender = NULL WHERE youtube_channel = ?",
            (ch,),
        )
        conn.commit()
        updated = cur.rowcount
    return {"channel": ch, "updated": int(updated)}


@app.post("/remove-audio")
def remove_audio(payload: dict = Body(...)):
    """
    Remove all unverified audio records and corresponding files for a channel.
    - Deletes rows where youtube_channel = channel AND (verify = 0 OR verify IS NULL)
    - Attempts to remove audio files on disk referenced by those rows
    Returns: { channel, removed_records, removed_files }
    """
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid body"}, status_code=400)
    ch = payload.get("channel") or payload.get("name") or payload.get("youtube_channel")
    password = payload.get("password")
    if not ch or not str(ch).strip():
        return JSONResponse({"error": "channel required"}, status_code=400)
    # Require admin password
    if not password or not secrets.compare_digest(str(password), ADMIN_LOCK_PASSWORD):
        return JSONResponse({"error": "invalid password"}, status_code=403)
    removed_files = 0
    removed_records = 0
    # Collect candidate file paths first
    paths: list[str] = []
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT audio_path FROM audios WHERE youtube_channel = ? AND (verify = 0 OR verify IS NULL)",
            (ch,),
        )
        rows = cur.fetchall()
        paths = [r[0] for r in rows if r and r[0]]
        # Delete records
        cur.execute(
            "DELETE FROM audios WHERE youtube_channel = ? AND (verify = 0 OR verify IS NULL)",
            (ch,),
        )
        removed_records = cur.rowcount or 0
        conn.commit()
    # Remove files on disk (best effort)
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
                removed_files += 1
        except Exception:
            continue
    return {"channel": ch, "removed_records": int(removed_records), "removed_files": int(removed_files)}


# ---- Download API: next verified not in client metadata ----
@app.post("/api/next-verified")
async def api_next_verified(request: Request, metadata_file: Optional[UploadFile] = File(None), metadata: Optional[dict] = Body(None)):
    """
    Return metadata for the next verified audio the client doesn't already have.

    Accepts either:
    - multipart/form-data with a JSON file field named `metadata_file`, or
    - application/json body matching the same schema.

    Expected JSON keys (all optional):
    - known_ids: [int, ...]                # database ids already present on client
    - known_paths: [str, ...]              # relative paths under sub_audio (e.g. "url_abc/audio_1.wav")
    - known_filenames: [str, ...]          # basenames like "audio_1.wav" (may collide; least preferred)

    Response 200 JSON on success:
    {
      id, filename, rel_path, public_url, size_bytes,
      duration_seconds, channel, language, sha256, remaining_after
    }

    If nothing remains, returns 204 No Content with JSON {"message": "no more verified"}.
    """
    # Parse metadata from file or body
    provided: dict = {}
    if metadata_file is not None:
        try:
            raw = await metadata_file.read()
            provided = {} if not raw else (__import__("json").loads(raw.decode("utf-8")))
        except Exception:
            return JSONResponse({"error": "invalid metadata_file JSON"}, status_code=400)
    elif isinstance(metadata, dict) and metadata:
        # JSON sent under the key 'metadata'
        provided = metadata
    else:
        # Accept raw top-level JSON: {"known_ids": [...], ...}
        try:
            if request.headers.get("content-type", "").lower().startswith("application/json"):
                provided = await request.json()
                if not isinstance(provided, dict):
                    provided = {}
        except Exception:
            provided = {}

    known_ids = provided.get("known_ids") or []
    known_paths = provided.get("known_paths") or []
    known_filenames = provided.get("known_filenames") or []
    exclude_basenames = bool(provided.get("exclude_basenames") or False)

    # Normalize lists
    try:
        known_ids = [int(x) for x in known_ids]
    except Exception:
        return JSONResponse({"error": "known_ids must be integers"}, status_code=400)
    try:
        known_paths = [str(x).strip() for x in known_paths if str(x).strip()]
        known_filenames = [str(x).strip() for x in known_filenames if str(x).strip()]
    except Exception:
        return JSONResponse({"error": "known_paths/known_filenames must be strings"}, status_code=400)

    # Build SQL with exclusions
    with get_db_conn() as conn:
        cur = conn.cursor()
        sql = (
            "SELECT id, audio_path, youtube_channel, language, duration_seconds, transcript, gender "
            "FROM audios WHERE verify = 1"
        )
        params: list = []

        if known_ids:
            placeholders = ",".join(["?"] * len(known_ids))
            sql += f" AND id NOT IN ({placeholders})"
            params.extend(known_ids)

        # Map known_paths (relative under SUB_AUDIO_ROOT) to absolute stored paths
        abs_paths: list[str] = []
        for p in known_paths:
            # Accept either relative (preferred) or absolute paths; also strip leading "/sub_audio/"
            rp = p
            if rp.startswith("/sub_audio/"):
                rp = rp[len("/sub_audio/"):]
            ap = rp if os.path.isabs(rp) else os.path.join(SUB_AUDIO_ROOT, rp)
            abs_paths.append(os.path.normpath(ap))
        if abs_paths:
            placeholders = ",".join(["?"] * len(abs_paths))
            sql += f" AND audio_path NOT IN ({placeholders})"
            params.extend(abs_paths)

        # Least-preferred: exclude by basename patterns (can collide across different videos)
        if exclude_basenames:
            for bn in known_filenames:
                sql += " AND audio_path NOT LIKE ?"
                params.append(f"%/{bn}")

        # Optional: filter by gender tag
        fg = provided.get("filter_gender")
        if isinstance(fg, str) and fg.strip() != "":
            fgl = fg.strip().lower()
            if fgl in {"none", "null", "empty"}:
                sql += " AND (gender IS NULL OR TRIM(gender) = '')"
            else:
                sql += " AND LOWER(TRIM(gender)) = ?"
                params.append(fgl)

        # Deterministic order
        sql += " ORDER BY id ASC LIMIT 1"
        cur.execute(sql, tuple(params))
        row = cur.fetchone()

        # Nothing left
        if not row:
            # No content must not include a body; return empty 204
            return Response(status_code=204)

        rec_id, audio_path, channel, language, duration_seconds, transcript, gender = row
        if not audio_path or not os.path.exists(audio_path):
            return JSONResponse({"error": "audio file missing on server"}, status_code=404)

        # Compute rel path and metadata
        rel_path = os.path.relpath(audio_path, SUB_AUDIO_ROOT).replace("\\", "/")
        filename = os.path.basename(audio_path)
        public_url = f"/sub_audio/{rel_path}"
        try:
            size_bytes = os.path.getsize(audio_path)
        except Exception:
            size_bytes = 0

        # Hash for integrity (sha256)
        sha256 = None
        try:
            h = hashlib.sha256()
            with open(audio_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            sha256 = h.hexdigest()
        except Exception:
            sha256 = None

        # Count remaining after this one
        count_sql = "SELECT COUNT(*) FROM audios WHERE verify = 1"
        count_params = []
        if known_ids:
            placeholders = ",".join(["?"] * len(known_ids))
            count_sql += f" AND id NOT IN ({placeholders})"
            count_params.extend(known_ids)
        if abs_paths:
            placeholders = ",".join(["?"] * len(abs_paths))
            count_sql += f" AND audio_path NOT IN ({placeholders})"
            count_params.extend(abs_paths)
        if exclude_basenames:
            for bn in known_filenames:
                count_sql += " AND audio_path NOT LIKE ?"
                count_params.append(f"%/{bn}")
        # Apply same gender filter to remaining count
        if isinstance(fg, str) and fg.strip() != "":
            fgl = fg.strip().lower()
            if fgl in {"none", "null", "empty"}:
                count_sql += " AND (gender IS NULL OR TRIM(gender) = '')"
            else:
                count_sql += " AND LOWER(TRIM(gender)) = ?"
                count_params.append(fgl)
        # Also exclude the one we're about to return
        count_sql += " AND id <> ?"
        count_params.append(int(rec_id))
        cur.execute(count_sql, tuple(count_params))
        left_after = int(cur.fetchone()[0] or 0)

    return {
        "id": int(rec_id),
        "filename": filename,
        "rel_path": rel_path,
        "public_url": public_url,
        "size_bytes": int(size_bytes),
        "duration_seconds": duration_seconds,
        "channel": channel,
        "language": language,
        "transcript": transcript,
        "gender": gender,
        "sha256": sha256,
        "remaining_after": left_after,
    }


if __name__ == "__main__":
    import uvicorn, os as _os
    uvicorn.run("app:app", host="0.0.0.0", port=int(_os.getenv("PORT", "2000")), reload=False)

def _to_pcm16_16k_mono(seg: AudioSegment) -> AudioSegment:
    """Return a new AudioSegment resampled to 16kHz, mono, 16-bit PCM.
    This helps keep ASR input consistent across original and split clips.
    """
    try:
        if not isinstance(seg, AudioSegment):
            return seg
        # Force mono, 16kHz, 16-bit
        s = seg.set_channels(1)
        s = s.set_frame_rate(16000)
        # set_sample_width(2) -> 16-bit PCM
        s = s.set_sample_width(2)
        return s
    except Exception:
        return seg
