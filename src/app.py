"""Streamlit Web UI for End-to-End Lecture Notes & Hindi Translation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.run_e2e_lecture import E2ELecturePipeline, E2EPipelineConfig
from src.video_downloader import download_video

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

st.set_page_config(page_title="Lecture Notes & Translation", layout="wide")


def is_youtube_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
        netloc = (parsed.netloc or "").lower()
        return "youtube.com" in netloc or "youtu.be" in netloc
    except Exception:
        return False


def _extract_wav_16k(video_path: Path, audios_dir: Path) -> Path:
    """Extract 16 kHz mono PCM WAV required by Phase 1 ASR."""
    audios_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audios_dir / f"{video_path.stem}_16k.wav"
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
        "-y",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds into concise human-readable duration."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    rem = seconds % 60
    return f"{mins}m {rem:.0f}s"


def process_video(video_input, is_uploaded: bool = False, translate_hindi: bool = False):
    """Run the default end-to-end multimodal pipeline with live stage-level status updates."""
    videos_dir = REPO_ROOT / "videos"
    audios_dir = REPO_ROOT / "audios"
    outputs_root = REPO_ROOT / "outputs" / "ui_runs"

    videos_dir.mkdir(parents=True, exist_ok=True)
    audios_dir.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    total_steps = 6 if translate_hindi else 5
    stage_meta = {
        "video": (1, "Saving uploaded video" if is_uploaded else "Downloading YouTube video", "Video ready"),
        "audio": (2, "Extracting 16 kHz mono audio", "Audio extracted"),
        "asr": (3, "Transcribing speech (ASR)", "ASR transcription complete"),
        "segmentation": (4, "Segmenting topics (Multimodal DBCA)", "Topic segmentation complete"),
        "notes": (5, "Generating English notes", "English notes generated"),
        "translation": (6, "Translating notes to Hindi", "Hindi translation complete"),
    }

    t_overall = time.perf_counter()
    current_label = "Starting pipeline"
    progress_bar = st.progress(0.0, text="Initializing pipeline...")

    with st.status("Initializing pipeline...", expanded=True) as status:
        def set_stage(stage_id: str, event: str, elapsed_sec: float | None = None, detail: str = "") -> None:
            nonlocal current_label
            step_num, active_text, done_text = stage_meta[stage_id]
            if event == "start":
                current_label = active_text
                progress_bar.progress(
                    max(0.0, (step_num - 1) / float(total_steps)),
                    text=f"Step {step_num}/{total_steps}: {active_text}...",
                )
                status.update(
                    label=f"Step {step_num}/{total_steps}: {active_text}...",
                    state="running",
                    expanded=True,
                )
            elif event == "done":
                suffix = f" ({_format_duration(elapsed_sec)})" if elapsed_sec is not None else ""
                extra = f" — `{detail}`" if detail else ""
                st.write(f"✅ **{done_text}**{extra}{suffix}")
                progress_bar.progress(
                    min(1.0, step_num / float(total_steps)),
                    text=f"Step {step_num}/{total_steps}: {done_text}{suffix}",
                )

        # 1) Acquire video file
        try:
            set_stage("video", "start")
            t0 = time.perf_counter()
            if is_uploaded:
                filename = os.path.basename(video_input.name)
                video_path = videos_dir / filename
                with open(video_path, "wb") as f:
                    f.write(video_input.read())
            elif is_youtube_url(video_input):
                downloaded_path = download_video(video_input, output_dir=str(videos_dir))
                if not downloaded_path or not os.path.exists(downloaded_path):
                    raise RuntimeError("YouTube download produced no output file.")
                video_path = Path(downloaded_path)
                filename = video_path.name
            else:
                raise ValueError("Please enter a valid YouTube URL or upload a local video file.")
            set_stage("video", "done", time.perf_counter() - t0, detail=filename)
        except Exception as e:
            status.update(label=f"Failed during: {current_label}", state="error", expanded=True)
            st.error(f"Video acquisition failed: {e}")
            st.code(traceback.format_exc())
            return None

        # 2) Extract 16 kHz mono WAV audio
        try:
            set_stage("audio", "start")
            t0 = time.perf_counter()
            wav_path = _extract_wav_16k(video_path, audios_dir)
            set_stage("audio", "done", time.perf_counter() - t0, detail=wav_path.name)
        except Exception as e:
            status.update(label=f"Failed during: {current_label}", state="error", expanded=True)
            st.error(f"Audio extraction failed: {e}")
            st.code(traceback.format_exc())
            return None

        # 3) Run unified end-to-end multimodal pipeline (ASR -> Segmentation -> Notes -> optional Hindi)
        try:
            stem = video_path.stem
            lecture_title = stem.replace("_", " ")
            run_output_dir = outputs_root / stem

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline_config = E2EPipelineConfig(
                device=device,
                translate_hindi=translate_hindi,
                output_dir=run_output_dir,
                title=lecture_title,
                course="Lecture",
            )
            pipeline = E2ELecturePipeline(config=pipeline_config)

            e2e_result = pipeline.run(
                audio_path=wav_path,
                video_path=video_path,
                metadata={"course": "Lecture", "lecture_title": lecture_title},
                stage_callback=lambda sid, ev, dt: set_stage(sid, ev, dt),
            )

            total_elapsed = time.perf_counter() - t_overall
            progress_bar.progress(1.0, text=f"Completed in {_format_duration(total_elapsed)}")
            status.update(
                label=f"Notes generated in {_format_duration(total_elapsed)}",
                state="complete",
                expanded=False,
            )

            return {
                "study_notes_markdown": e2e_result.study_notes_markdown,
                "study_notes_json": e2e_result.study_notes_json,
                "study_notes_hindi_markdown": e2e_result.study_notes_hindi_markdown,
                "study_notes_hindi_json": e2e_result.study_notes_hindi_json,
                "execution_summary": e2e_result.execution_summary,
                "filename": stem,
                "output_dir": str(run_output_dir),
            }
        except Exception as e:
            status.update(label=f"Failed during: {current_label}", state="error", expanded=True)
            st.error(f"Pipeline execution failed: {e}")
            st.code(traceback.format_exc())
            return None


# ==============================================================================
# Streamlit UI Layout
# ==============================================================================

st.title("Lecture Notes & Hindi Translation")

col1, col2 = st.columns([2, 1])

with col1:
    option = st.radio("Input type:", ["YouTube URL", "Local file upload"], horizontal=True)

    if option == "YouTube URL":
        video_input = st.text_input("Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        video_input = st.file_uploader(
            "Upload local video file (mp4/mkv/webm/mov)",
            type=["mp4", "mkv", "webm", "mov"],
        )

with col2:
    st.markdown("### Translation")
    translate_hindi = st.checkbox(
        "Translate Notes to Hindi (हिन्दी अनुवाद)",
        value=True,
        help="Translates generated notes into Hindi using the merged Sarvam-1 Step 200 GGUF model with LaTeX and code preservation.",
    )

    st.markdown("**Active Pipeline:**")
    st.write("- **ASR**: `faster-whisper large-v3-turbo` (Batched)")
    st.write("- **Segmentation**: `Qwen3-Embedding + Vision + DBCA`")
    st.write("- **Notes**: `Qwen3.5-4B` (Ollama)")
    st.write(
        f"- **Hindi Translation**: `{'Sarvam-1 Step 200 GGUF (Ollama)' if translate_hindi else 'Disabled'}`"
    )

st.markdown("---")

if st.button("Generate Notes", type="primary"):
    if option == "YouTube URL" and (not video_input or not video_input.strip()):
        st.error("Please provide a YouTube URL.")
    elif option == "Local file upload" and not video_input:
        st.error("Please upload a local video file.")
    else:
        res = process_video(
            video_input,
            is_uploaded=(option != "YouTube URL"),
            translate_hindi=translate_hindi,
        )
        if res:
            st.session_state["last_notes_result"] = res
        else:
            st.error("Processing failed; see details above.")

result = st.session_state.get("last_notes_result")
if result:
    tab1, tab2 = st.tabs([
        "English Notes",
        "Hindi Notes (हिन्दी नोट्स)",
    ])

    with tab1:
        st.download_button(
            "Download English Notes (.md)",
            result["study_notes_markdown"],
            file_name=f"{result['filename']}_notes.md",
            mime="text/markdown",
        )
        st.markdown(result["study_notes_markdown"])

    with tab2:
        if result.get("study_notes_hindi_markdown"):
            st.download_button(
                "Download Hindi Notes (.md)",
                result["study_notes_hindi_markdown"],
                file_name=f"{result['filename']}_notes_hi.md",
                mime="text/markdown",
            )
            st.markdown(result["study_notes_hindi_markdown"])
        else:
            st.info(
                "Hindi translation was not enabled for this run. Check 'Translate Notes to Hindi' and click Generate Notes."            )
