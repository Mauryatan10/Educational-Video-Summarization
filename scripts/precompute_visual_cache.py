"""Precompute and Cache Visual Features for Multimodal Discourse Segmentation.

Extracts canonical slide keyframes, computes visual transition deltas, and encodes
slide representations into compact disk caches (visual_cache.pt).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from src.vision.keyframe_extractor import LectureKeyframeExtractor
from src.vision.slide_encoder import QwenSlideRepresentationEncoder
from transformers import AutoModel, AutoImageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_cache_for_manifest(
    manifest_path: str,
    video_dir: str,
    output_cache_path: str,
    arrow_files: list[str],
    device: str = "cuda",
    max_sentences: int = 400,
    slide_threshold: float = 0.48,
    min_interval: float = 18.0,
    continuous: bool = False,
) -> None:
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_p) as f:
        manifest = json.load(f)

    video_d = Path(video_dir)
    cache_p = Path(output_cache_path)
    cache_p.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if present (resumable)
    cache: dict[str, dict[str, torch.Tensor]] = {}
    if cache_p.exists():
        try:
            cache = torch.load(cache_p, map_location="cpu", weights_only=False)
            logger.info(f"Loaded existing cache with {len(cache)} entries from {cache_p}")
        except Exception as e:
            logger.warning(f"Could not load existing cache: {e}")

    # Index arrow tables by (shard_idx, row_idx) or doc_idx
    arrow_tables = []
    for af in arrow_files:
        with pa.memory_map(af, "r") as src:
            arrow_tables.append(ipc.open_stream(src).read_all())

    # Load Qwen vision tower
    logger.info(f"Loading Qwen 3.5 0.8B Vision Tower on {device}...")
    base_model = AutoModel.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.bfloat16)
    visual_model = base_model.visual.to(device).eval()
    if hasattr(base_model, "model") and hasattr(base_model.model, "language_model"):
        del base_model.model.language_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    slide_encoder = QwenSlideRepresentationEncoder(visual_model=visual_model, device=device)
    keyframe_extractor = LectureKeyframeExtractor(
        fps_sample=0.5,
        min_slide_change_threshold=slide_threshold,
        min_interval_sec=min_interval,
    )

    tmp_dir = Path("data/tmp_cache_keyframes")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    failed_vids = set()
    total_manifest = len(manifest)
    logger.info(f"Visual precomputation initialized for {total_manifest} items (currently cached: {len(cache)}).")

    t0 = time.time()
    processed_count = 0

    while True:
        # Scan for pending videos that have finished downloading
        pending_items = []
        for item in manifest:
            vid = item["video_id"]
            if vid in cache or vid in failed_vids:
                continue

            found_path = None
            for ext in ("mp4", "webm", "mkv"):
                c = video_d / f"{vid}.{ext}"
                if c.exists() and c.stat().st_size > 100_000:
                    part_c = video_d / f"{vid}.{ext}.part"
                    if not part_c.exists() and (time.time() - c.stat().st_mtime > 3.0):
                        found_path = c
                        break
            if found_path:
                item["local_path"] = str(found_path)
                pending_items.append(item)

        if pending_items:
            logger.info(f"Found batch of {len(pending_items)} ready videos to encode.")
            for item in pending_items:
                vid = item["video_id"]
                v_path = item["local_path"]
                dur = item["duration_sec"]

                s_idx = item.get("shard_idx", 0)
                r_idx = item.get("row_idx", item.get("doc_idx", 0))
                table = arrow_tables[s_idx]
                sents = table["text_ref"][r_idx].as_py()
                if isinstance(sents, str):
                    sents = [sents]
                sents = sents[:max_sentences]
                n_sents = len(sents)

                sent_times = [
                    (i * (dur / n_sents), (i + 1) * (dur / n_sents))
                    for i in range(n_sents)
                ]

                cur_kf_dir = tmp_dir / vid
                try:
                    kfs = keyframe_extractor.extract_keyframes(v_path, output_dir=str(cur_kf_dir))
                    v_shifts = keyframe_extractor.compute_sentence_visual_shifts(sent_times, kfs)
                    slide_embeddings = slide_encoder.encode_keyframes(kfs, batch_size=4)
                    v_aligned = slide_encoder.align_slides_to_sentences(sent_times, kfs, slide_embeddings)

                    cache[vid] = {
                        "visual_embeddings": v_aligned.cpu().to(torch.float32),
                        "visual_shifts": torch.tensor(v_shifts, dtype=torch.float32),
                        "num_keyframes": len(kfs),
                        "duration_sec": dur,
                        "n_sentences": n_sents,
                    }
                    processed_count += 1

                    if processed_count % 10 == 0:
                        torch.save(cache, cache_p)
                        elapsed = time.time() - t0
                        logger.info(
                            f"[{len(cache):03d}/{total_manifest}] Cached {vid} ({len(kfs)} slides, {n_sents} sents). "
                            f"Total cached: {len(cache)}. Elapsed: {elapsed:.1f}s"
                        )
                except Exception as e:
                    logger.warning(f"Failed to process video {vid}: {e}")
                    failed_vids.add(vid)
                finally:
                    if cur_kf_dir.exists():
                        shutil.rmtree(cur_kf_dir, ignore_errors=True)

            # Checkpoint after batch
            torch.save(cache, cache_p)

        else:
            # Check if downloader has finished
            report_p = video_d.parent / "download_report.json"
            if continuous and not report_p.exists():
                logger.info("Waiting for more downloaded videos (sleeping 15s)...")
                time.sleep(15)
                continue
            else:
                logger.info("No further pending videos found and download finished. Exiting precomputation loop.")
                break

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    torch.save(cache, cache_p)
    logger.info(f"Saved complete visual cache with {len(cache)} entries to {cache_p} ({cache_p.stat().st_size / (1024*1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "train", "both", "custom"], default="val")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--output-cache", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--slide-threshold", type=float, default=0.48)
    parser.add_argument("--min-interval", type=float, default=18.0)
    parser.add_argument("--continuous", action="store_true", help="Continuously poll for newly downloaded videos until completion")
    args = parser.parse_args()

    if args.manifest and args.video_dir and args.output_cache:
        logger.info(f"=== Precomputing Visual Cache for Custom Path: {args.manifest} ===")
        train_arrows = sorted(glob.glob("datasets/ytseg-dataset/train/*.arrow"))
        build_cache_for_manifest(
            manifest_path=args.manifest,
            video_dir=args.video_dir,
            output_cache_path=args.output_cache,
            arrow_files=train_arrows,
            device=args.device,
            slide_threshold=args.slide_threshold,
            min_interval=args.min_interval,
            continuous=args.continuous,
        )
        return

    if args.split in ("val", "both"):
        val_arrows = ["datasets/ytseg-dataset/validation/data-00000-of-00001.arrow"]
        logger.info("=== Precomputing Visual Cache for Validation Split ===")
        build_cache_for_manifest(
            manifest_path="data/ytseg_val_100/manifest_100.json",
            video_dir="data/ytseg_val_100/videos",
            output_cache_path="data/ytseg_val_100/visual_cache.pt",
            arrow_files=val_arrows,
            device=args.device,
            slide_threshold=args.slide_threshold,
            min_interval=args.min_interval,
            continuous=args.continuous,
        )

    if args.split in ("train", "both"):
        train_arrows = sorted(glob.glob("datasets/ytseg-dataset/train/*.arrow"))
        logger.info("=== Precomputing Visual Cache for Training Split ===")
        build_cache_for_manifest(
            manifest_path="data/ytseg_train_400/manifest_400.json",
            video_dir="data/ytseg_train_400/videos",
            output_cache_path="data/ytseg_train_400/visual_cache.pt",
            arrow_files=train_arrows,
            device=args.device,
            slide_threshold=args.slide_threshold,
            min_interval=args.min_interval,
            continuous=args.continuous,
        )


if __name__ == "__main__":
    main()
