# app.py
import streamlit as st
import os
from urllib.parse import urlparse
from inference import load_model, generate_summary
from video_downloader import download_video
from video_to_audio import extract_audio_ffmpeg
from audio_to_text import transcribe_audio
import traceback

st.set_page_config(page_title="Video Summarization", layout="wide")


def is_youtube_url(path: str) -> bool:
    try:
        parsed = urlparse(path)
        netloc = (parsed.netloc or "").lower()
        return "youtube.com" in netloc or "youtu.be" in netloc
    except Exception:
        return False

def get_file_name_from_path(path):
    return os.path.basename(path)

@st.cache_resource(show_spinner=False)
def cached_load_model(adapter_choice: str):
    """
    adapter_choice: 'hier' or 'lora'
    returns: (model, tokenizer)
    """
    return load_model(adapter=adapter_choice)


def process_video(video_input, adapter_choice="hier", is_uploaded=False):
    videos_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\videos"
    audios_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\audios"
    texts_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\texts"

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(audios_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)

    try:
        if is_uploaded:
            st.info("Saving uploaded video...")
            filename = get_file_name_from_path(video_input.name)
            video_path = os.path.join(videos_dir, filename)
            with open(video_path, "wb") as f:
                f.write(video_input.read())

        elif is_youtube_url(video_input):
            st.info("Downloading YouTube video...")
            download_video(video_input, output_dir=videos_dir)
            # pick newest mp4 in folder
            mp4s = sorted([f for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")],
                          key=lambda x: os.path.getmtime(os.path.join(videos_dir, x)))
            if not mp4s:
                st.error("YouTube download failed (no .mp4 found).")
                return None
            filename = mp4s[-1]
            video_path = os.path.join(videos_dir, filename)

        else:
            # assume local path
            st.info("Copying local video file...")
            filename = get_file_name_from_path(video_input)
            video_path = os.path.join(videos_dir, filename)
            # use python copy that works cross-platform
            with open(video_input, "rb") as src, open(video_path, "wb") as dst:
                dst.write(src.read())

        st.success(f"Video ready: {filename}")

    except Exception as e:
        st.error("Video step failed: " + str(e))
        st.error(traceback.format_exc())
        return None

    try:
        st.info("Extracting audio (ffmpeg)...")
        extract_audio_ffmpeg(filename, video_dir=videos_dir, audio_dir=audios_dir)
        audio_filename = filename.rsplit(".", 1)[0] + ".mp3"
        audio_path = os.path.join(audios_dir, audio_filename)
        if not os.path.exists(audio_path):
            st.warning("Audio file not found after extraction; continuing but transcription may fail.")
        else:
            st.success("Audio extracted")
    except Exception as e:
        st.error("Audio extraction failed: " + str(e))
        st.error(traceback.format_exc())
        return None

    # 3) Transcribe (Whisper)
    try:
        st.info("Transcribing audio with Whisper (small)...")
        # transcribe_audio should return the transcript text (string)
        transcript = transcribe_audio(audio_filename, audio_dir=audios_dir, text_dir=texts_dir, model_size="small")
        print(transcript[:300])
        if not transcript:
            st.warning("Transcription returned empty text.")
        else:
            st.success("Transcription complete")
    except Exception as e:
        st.error("Transcription failed: " + str(e))
        st.error(traceback.format_exc())
        return None

    # 4) Load model (cached) and generate summary
    try:
        st.info(f"Loading summarization model...")
        with st.spinner("Loading model..."):
            model, tokenizer = cached_load_model(adapter_choice)
        st.success("Model loaded")

        st.info("Generating summary...")
        with st.spinner("Generating summary..."):
            summary = generate_summary(transcript, model=model, tokenizer=tokenizer, adapter=adapter_choice)
        if summary:
            st.success("Summary generated")
        else:
            st.warning("Summary generation returned empty result.")
        return summary

    except Exception as e:
        st.error("Model inference failed: " + str(e))
        st.error(traceback.format_exc())
        return None

st.title("Video Summarization")

col1, col2 = st.columns([2, 1])

with col1:
    option = st.radio("Input type:", ["YouTube URL", "Local file upload"], horizontal=True)

    if option == "YouTube URL":
        video_input = st.text_input("Enter YouTube URL")
    else:
        video_input = st.file_uploader("Upload local video file (mp4/mkv/mov)", type=["mp4", "mkv", "mov"])

with col2:
    adapter_choice_human = st.selectbox("Choose adapter", ["Hierarchical", "LoRA-only"])
    adapter_choice = "hier" if adapter_choice_human.startswith("Hierarchical") else "lora"
    st.markdown("**Model options:**")
    st.write(f"- Adapter: **{adapter_choice_human}**")

st.markdown("---")

if st.button("Process Video"):
    if option == "YouTube URL" and (not video_input or not video_input.strip()):
        st.error("Please provide a YouTube URL.")
    elif option == "Local file upload" and not video_input:
        st.error("Please upload a local video file.")
    else:
        summary = process_video(video_input if option == "YouTube URL" else video_input, adapter_choice=adapter_choice, is_uploaded=(option!="YouTube URL"))
        if summary:
            st.subheader("Generated Summary")
            st.write(summary)
            st.download_button("Download summary ", summary, file_name="summary.txt")
        else:
            st.error("Processing failed; see messages above.")
