"""Download 400 Tier-1 Granular Training Videos using yt-dlp.

Saves videos into data/ytseg_train_400/videos/ at 360p resolution.
Tracks progress, supports resumable downloads, and logs errors cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def download_single_video(item: dict, output_dir: Path) -> dict:
    vid = item["video_id"]
    url = item.get("url", f"https://www.youtube.com/watch?v={vid}")

    # Check if already downloaded
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
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=360]+bestaudio/best[height<=360]",
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
    ]

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
                size_mb = found.stat().st_size / (1024 * 1024)
                return {
                    "video_id": vid,
                    "status": "success",
                    "file_path": str(found),
                    "size_mb": size_mb,
                    "elapsed_sec": elapsed,
                    "duration_sec": item.get("duration_sec", 0),
                }
            else:
                return {"video_id": vid, "status": "file_not_found_after_dl", "error": "No file on disk"}
        else:
            return {"video_id": vid, "status": "failed", "error": res.stderr.strip()[:200]}
    except subprocess.TimeoutExpired:
        return {"video_id": vid, "status": "timeout", "error": "Exceeded 180s limit"}
    except Exception as e:
        return {"video_id": vid, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/ytseg_train_400/manifest_400.json")
    parser.add_argument("--output-dir", type=str, default="data/ytseg_train_400/videos")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)[:args.limit]

    print(f"=== Starting download of {len(manifest)} training videos to {out_dir} (workers={args.max_workers}) ===")

    completed = 0
    total_size_mb = 0.0
    results = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_item = {executor.submit(download_single_video, item, out_dir): item for item in manifest}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            res = future.result()
            results.append(res)
            completed += 1

            if res["status"] in ("success", "already_exists"):
                sz = res.get("size_mb", 0.0)
                total_size_mb += sz
                status_str = "CACHED" if res["status"] == "already_exists" else f"DOWNLOADED ({sz:.1f} MB)"
                print(f"[{completed:03d}/{len(manifest)}] ✓ {res['video_id']}: {status_str} | Total: {total_size_mb/1024:.2f} GB")
            else:
                print(f"[{completed:03d}/{len(manifest)}] ✗ {res['video_id']}: FAILED ({res.get('error', 'unknown error')})")

    success_count = sum(1 for r in results if r["status"] in ("success", "already_exists"))
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print(f"Successful: {success_count}/{len(manifest)}")
    print(f"Total Disk Used: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    print("=" * 60)

    report_path = out_dir.parent / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved download report to {report_path}")


if __name__ == "__main__":
    main()
