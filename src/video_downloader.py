from pytubefix import YouTube
import os
import sys
import yt_dlp
from urllib.parse import urlparse, parse_qs
import re

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def extract_video_id(url):
    """
    Extract the YouTube video ID from any URL format.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
    raise ValueError("Invalid YouTube URL")

_INVALID = '<>:"/\\|?*'

def _safe_filename(title: str) -> str:
    cleaned = "".join(ch for ch in (title or "video") if ch not in _INVALID)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned[:120] or "video") + ".mp4"


def _download_ytdlp(url: str, output_dir: str, filename: str | None) -> str:
    """Primary backend: yt-dlp (no PO token needed). Returns output path."""
    if filename is None:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            filename = _safe_filename(info.get("title", "video"))
    base = os.path.splitext(filename)[0]

    outtmpl = os.path.join(output_dir, base + ".%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        # We only need audio for transcription; cap resolution to keep downloads fast.
        # Modern YouTube only exposes split video+audio streams, so merge them (ffmpeg).
        "format": "bestvideo[height<=480]+bestaudio/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    candidates = [
        f for f in os.listdir(output_dir)
        if f.startswith(base) and f.lower().endswith((".mp4", ".webm", ".mkv", ".m4v"))
    ]
    if not candidates:
        raise RuntimeError("yt-dlp finished but produced no output file.")
    candidates.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
    return os.path.join(output_dir, candidates[-1])


def _download_pytubefix(url: str, output_dir: str, filename: str | None) -> str:
    """Fallback backend: pytubefix. Returns output path."""
    yt = YouTube(url)
    if filename is None:
        filename = _safe_filename(yt.title)
    print(f"\nTitle: {yt.title}")
    print(f"Target file: {filename}")

    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    if stream is None:
        stream = yt.streams.filter(file_extension="mp4").first()
    if stream is None:
        raise RuntimeError("No downloadable MP4 stream found for this video.")
    return stream.download(output_path=output_dir, filename=filename)


def download_video(url, output_dir="./videos", filename=None):
    """Download a YouTube video, returning the output file path.

    Tries yt-dlp first (works without a PO token), then falls back to pytubefix.
    Raises RuntimeError listing both backends' failures if all attempts fail.
    """
    os.makedirs(output_dir, exist_ok=True)
    errors = []

    try:
        path = _download_ytdlp(url, output_dir, filename)
        print(f"Downloaded (yt-dlp): {path}")
        return path
    except Exception as e:
        errors.append(f"yt-dlp: {e}")

    try:
        path = _download_pytubefix(url, output_dir, filename)
        print(f"Downloaded (pytubefix): {path}")
        return path
    except Exception as e:
        errors.append(f"pytubefix: {e}")

    raise RuntimeError("All YouTube download backends failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    download_video("https://www.youtube.com/watch?v=ZJ2rmi4wSsk")
