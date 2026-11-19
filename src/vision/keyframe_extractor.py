"""Sparse Keyframe Extraction and Visual Transition Differencing for Lecture Videos.

Operates with zero GPU VRAM using FFmpeg scene detection and perceptual visual differencing.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class KeyframeInfo:
    """Metadata describing an extracted canonical slide keyframe."""
    index: int
    timestamp_sec: float
    image_path: str
    visual_shift_score: float  # Difference metric against preceding canonical slide [0.0, 1.0]


def compute_image_difference(img1: Image.Image, img2: Image.Image) -> float:
    """Compute normalized perceptual visual difference between two RGB images in [0.0, 1.0]."""
    g1 = np.asarray(img1.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    g2 = np.asarray(img2.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)

    mae = float(np.mean(np.abs(g1 - g2)) / 255.0)

    std1, std2 = float(np.std(g1)), float(np.std(g2))
    cov = float(np.mean((g1 - np.mean(g1)) * (g2 - np.mean(g2))))
    cs = (2.0 * cov + 1e-4) / (std1**2 + std2**2 + 1e-4)
    cs = float(np.clip(cs, 0.0, 1.0))

    diff_score = 0.6 * mae + 0.4 * (1.0 - cs)
    return float(np.clip(diff_score, 0.0, 1.0))


class LectureKeyframeExtractor:
    """Extracts sparse canonical lecture slide keyframes and per-sentence visual transitions."""

    def __init__(
        self,
        fps_sample: float = 0.5,
        min_slide_change_threshold: float = 0.48,
        max_keyframes: int = 60,
        min_interval_sec: float = 18.0,
    ) -> None:
        self.fps_sample = fps_sample
        self.min_slide_change_threshold = min_slide_change_threshold
        self.max_keyframes = max_keyframes
        self.min_interval_sec = min_interval_sec

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> List[KeyframeInfo]:
        """Extract canonical slide keyframes from video file."""
        video_p = Path(video_path)
        if not video_p.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if output_dir is None:
            out_p = video_p.parent / f"{video_p.stem}_keyframes"
        else:
            out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        frame_pattern = out_p / "raw_frame_%05d.jpg"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_p),
            "-vf", f"fps={self.fps_sample},scale=640:-1",
            "-q:v", "4",
            str(frame_pattern),
        ]

        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        raw_frames = sorted(list(out_p.glob("raw_frame_*.jpg")))
        if not raw_frames:
            return []

        canonical_keyframes: List[KeyframeInfo] = []
        last_canonical_img: Optional[Image.Image] = None
        last_canonical_time: float = -999.0

        for idx, f_path in enumerate(raw_frames):
            timestamp = idx / self.fps_sample
            try:
                curr_img = Image.open(f_path).convert("RGB")
            except Exception:
                continue

            if last_canonical_img is None:
                canonical_path = out_p / f"canonical_slide_{len(canonical_keyframes):03d}.jpg"
                curr_img.save(canonical_path, quality=85)
                canonical_keyframes.append(
                    KeyframeInfo(
                        index=0,
                        timestamp_sec=timestamp,
                        image_path=str(canonical_path),
                        visual_shift_score=1.0,
                    )
                )
                last_canonical_img = curr_img
                last_canonical_time = timestamp
                continue

            diff = compute_image_difference(last_canonical_img, curr_img)
            time_since_last = timestamp - last_canonical_time

            if diff >= self.min_slide_change_threshold and time_since_last >= self.min_interval_sec:
                canonical_path = out_p / f"canonical_slide_{len(canonical_keyframes):03d}.jpg"
                curr_img.save(canonical_path, quality=85)
                canonical_keyframes.append(
                    KeyframeInfo(
                        index=len(canonical_keyframes),
                        timestamp_sec=timestamp,
                        image_path=str(canonical_path),
                        visual_shift_score=diff,
                    )
                )
                last_canonical_img = curr_img
                last_canonical_time = timestamp

                if len(canonical_keyframes) >= self.max_keyframes:
                    break

        for rf in raw_frames:
            try:
                rf.unlink()
            except OSError:
                pass

        return canonical_keyframes

    def compute_sentence_visual_shifts(
        self,
        sentence_timestamps: List[Tuple[float, float]],
        keyframes: List[KeyframeInfo],
    ) -> List[float]:
        """Compute visual transition delta Delta V_i in [0.0, 1.0] for each sentence."""
        shifts: List[float] = [0.0] * len(sentence_timestamps)
        if not keyframes:
            return shifts

        kf_times = [kf.timestamp_sec for kf in keyframes]
        kf_scores = [kf.visual_shift_score for kf in keyframes]

        for s_idx, (t_start, t_end) in enumerate(sentence_timestamps):
            max_score = 0.0
            for kt, ks in zip(kf_times, kf_scores):
                if (t_start - 1.0) <= kt <= (t_end + 1.0):
                    if ks > max_score:
                        max_score = ks
            shifts[s_idx] = float(np.clip(max_score, 0.0, 1.0))

        return shifts
