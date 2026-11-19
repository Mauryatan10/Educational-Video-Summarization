"""Scaled Dataset Downloader with Strict Disk Budget Enforcer.

Downloads Tier-1 Granular training videos up to a strict maximum total size (e.g. 38.0 GB),
preventing any disk overflow and ensuring optimal training scale.
Uses authenticated cookies from Firefox and Node.js JS challenge solver.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_dir_size_gb(directory: Path) -> float:
    total_bytes = 0
    if not directory.exists():
        return 0.0
    for entry in directory.glob("*"):
        if entry.is_file():
            try:
                total_bytes += entry.stat().st_size
            except OSError:
                pass
    return total_bytes / (1024 ** 3)


def ensure_cookies(cookies_path: Path) -> bool:
    """Ensure valid cookies file exists by extracting from Firefox if needed."""
    if cookies_path.exists() and cookies_path.stat().st_size > 1000:
        logger.info(f"Using existing cookies file: {cookies_path} ({cookies_path.stat().st_size / 1024:.1f} KB)")
        return True

    logger.info(f"Extracting cookies from Firefox into {cookies_path}...")
    cookies_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", "firefox",
        "--cookies", str(cookies_path),
        "--simulate",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if cookies_path.exists() and cookies_path.stat().st_size > 1000:
            logger.info(f"Successfully exported {cookies_path.stat().st_size / 1024:.1f} KB cookies from Firefox.")
            return True
        else:
            logger.warning(f"Could not export cookies to file: {res.stderr[:200]}")
            return False
    except Exception as e:
        logger.warning(f"Failed to export cookies: {e}")
        return False


def download_single_video(item: dict, output_dir: Path, cookies_path: Path | None = None) -> dict:
    vid = item["video_id"]
    url = item.get("url", f"https://www.youtube.com/watch?v={vid}")

    for ext in ("mp4", "webm", "mkv"):
        candidate = output_dir / f"{vid}.{ext}"
        if candidate.exists() and candidate.stat().st_size > 100_000:
            return {
                "video_id": vid,
                "status": "already_exists",
                "file_path": str(candidate),
                "size_mb": candidate.stat().st_size / (1024 * 1024),
                "duration_sec": item.get("duration_sec", 0),
            }

    t0 = time.time()
    out_template = str(output_dir / f"{vid}.%(ext)s")
    cmd = ["yt-dlp"]

    if cookies_path and cookies_path.exists():
        cmd.extend(["--cookies", str(cookies_path)])
    else:
        cmd.extend(["--cookies-from-browser", "firefox"])

    cmd.extend([
        "--js-runtimes", "node:/usr/bin/node",
        "--remote-components", "ejs:github",
        "-f", "best[height<=360]/bestvideo[height<=360]+bestaudio/best[height<=360]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-mtime",
        "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", "30",
        "--quiet",
        "--no-warnings",
        "-o", out_template,
        url,
    ])

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        elapsed = time.time() - t0
        if res.returncode == 0:
            found = None
            for ext in ("mp4", "webm", "mkv"):
                cand = output_dir / f"{vid}.{ext}"
                if cand.exists() and cand.stat().st_size > 100_000:
                    found = cand
                    break
            if found:
                sz = found.stat().st_size / (1024 * 1024)
                return {
                    "video_id": vid,
                    "status": "success",
                    "file_path": str(found),
                    "size_mb": sz,
                    "duration_sec": item.get("duration_sec", 0),
                    "elapsed_sec": elapsed,
                }
            else:
                return {"video_id": vid, "status": "failed", "error": "file missing post-download"}
        else:
            return {"video_id": vid, "status": "failed", "error": res.stderr[:200].strip()}
    except subprocess.TimeoutExpired:
        return {"video_id": vid, "status": "failed", "error": "download timeout (>180s)"}
    except Exception as e:
        return {"video_id": vid, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/ytseg_train_scale/manifest_1600.json")
    parser.add_argument("--video-dir", type=str, default="data/ytseg_train_scale/videos")
    parser.add_argument("--cookies-file", type=str, default="data/ytseg_train_scale/firefox_cookies.txt")
    parser.add_argument("--max-total-gb", type=float, default=38.0, help="Strict upper bound size of videos dir in GB (< 40 GB)")
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cookies_p = Path(args.cookies_file)

    ensure_cookies(cookies_p)

    with open(args.manifest) as f:
        manifest = json.load(f)

    cur_gb = get_dir_size_gb(out_dir)
    logger.info(f"=== Starting Scaled Download ({len(manifest)} items, budget <= {args.max_total_gb:.1f} GB, current: {cur_gb:.2f} GB) ===")

    results = []
    completed = 0
    limit_reached = False

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {}
        for item in manifest:
            current_sz = get_dir_size_gb(out_dir)
            if current_sz >= args.max_total_gb:
                logger.warning(f"[BUDGET REACHED] Video directory size {current_sz:.2f} GB >= {args.max_total_gb:.1f} GB. Halting scheduling.")
                limit_reached = True
                break

            fut = executor.submit(download_single_video, item, out_dir, cookies_p)
            future_to_item[fut] = item

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            res = future.result()
            results.append(res)
            completed += 1

            cur_gb = get_dir_size_gb(out_dir)

            # Hard safety barrier: if exceeding 39.5 GB, delete file and enforce strict < 40 GB limit
            if cur_gb > 39.5 and res.get("status") == "success" and "file_path" in res:
                fp = Path(res["file_path"])
                if fp.exists():
                    fp.unlink()
                res["status"] = "budget_culled"
                cur_gb = get_dir_size_gb(out_dir)

            if res["status"] in ("success", "already_exists"):
                status_str = "CACHED" if res["status"] == "already_exists" else f"DOWNLOADED ({res.get('size_mb', 0):.1f} MB)"
                print(f"[{completed:04d}/{len(future_to_item)}] ✓ {res['video_id']}: {status_str} | Total Dir: {cur_gb:.2f} GB", flush=True)
            else:
                print(f"[{completed:04d}/{len(future_to_item)}] ✗ {res['video_id']}: {res['status'].upper()} ({res.get('error', '')})", flush=True)

            if cur_gb >= args.max_total_gb and not limit_reached:
                logger.warning(f"[BUDGET REACHED] Video directory size reached {cur_gb:.2f} GB >= {args.max_total_gb:.1f} GB. Cancelling remaining.")
                limit_reached = True
                for f in future_to_item:
                    f.cancel()

    final_gb = get_dir_size_gb(out_dir)
    success_count = sum(1 for r in results if r["status"] in ("success", "already_exists"))
    logger.info("=" * 70)
    logger.info("SCALED DOWNLOAD COMPLETE")
    logger.info(f"Total Available Videos: {success_count}/{len(results)}")
    logger.info(f"Final Total Disk Used: {final_gb:.2f} GB (Strict Limit: {args.max_total_gb:.1f} GB)")
    logger.info("=" * 70)

    report_path = out_dir.parent / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved download report to {report_path}")


if __name__ == "__main__":
    main()
