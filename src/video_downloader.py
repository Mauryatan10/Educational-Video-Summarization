import os
import sys
import yt_dlp
from urllib.parse import urlparse, parse_qs
import re

try:
    from pytubefix import YouTube
except ImportError:
    YouTube = None

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


def _get_ytdlp_opts(output_dir: str, base: str | None = None, skip_download: bool = False, browser: str | None = None) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
    }
    if browser:
        opts["cookiesfrombrowser"] = (browser, None, None, None)
    if skip_download:
        opts["skip_download"] = True
    else:
        outtmpl = os.path.join(output_dir, (base or "%(title)s") + ".%(ext)s")
        opts.update({
            "outtmpl": outtmpl,
            "format": "bestvideo[height<=480]+bestaudio/bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
        })
    return opts


def _download_ytdlp(url: str, output_dir: str, filename: str | None) -> str:
    """Primary backend: yt-dlp with node js runtime and automatic browser cookie discovery."""
    candidate_browsers = ["firefox", "chrome", "chromium", "brave", None]
    
    # First find a working configuration (browser cookies if required by YouTube bot-detection)
    working_browser = None
    info = None
    last_err = None

    for browser in candidate_browsers:
        try:
            test_opts = _get_ytdlp_opts(output_dir, skip_download=True, browser=browser)
            with yt_dlp.YoutubeDL(test_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and info.get("title"):
                    working_browser = browser
                    break
        except Exception as e:
            last_err = e
            continue

    if info is None:
        raise RuntimeError(f"yt-dlp could not extract video info with any browser profile: {last_err}")

    if filename is None:
        filename = _safe_filename(info.get("title", "video"))
    base = os.path.splitext(filename)[0]

    dl_opts = _get_ytdlp_opts(output_dir, base=base, skip_download=False, browser=working_browser)
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
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
    if YouTube is None:
        raise RuntimeError("pytubefix is not installed.")
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
