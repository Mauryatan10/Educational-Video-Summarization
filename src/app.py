# app.py
import streamlit as st
import os
import sys
from typing import Optional
import torch
from pathlib import Path
from urllib.parse import urlparse
from inference import load_model, generate_summary
from topic_aware_miniseg_led import (
    format_text_output,
    generate_topic_aware_summaries,
    load_led_hier_model,
    load_miniseg_model,
)
from video_downloader import download_video
from video_to_audio import extract_audio_ffmpeg
from audio_to_text import transcribe_audio
ENABLE_TRANSLATION = False
if ENABLE_TRANSLATION:
    from translate_stage import load_translation_model, translate_texts, translate_one, LANG_CODES
    from glossary import glossary_from_text
    from translation_qe import translate_with_qe
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Master switch for the translation stage. Set to True once the IndicTrans2
# fine-tune is ready. When False, the pipeline and GUI are summarize-only.
ENABLE_TRANSLATION = False

# Target-language picker (display label -> IndicTrans2 code).
LANG_LABELS = {
    "Hindi (hin_Deva)": "hin_Deva",
    "Bengali (ben_Beng)": "ben_Beng",
    "Marathi (mar_Deva)": "mar_Deva",
    "Tamil (tam_Taml)": "tam_Taml",
    "Nepali (npi_Deva)": "npi_Deva",
    "Urdu (urd_Arab)": "urd_Arab",
    "Malayalam (mal_Mlym)": "mal_Mlym",
    "Kannada (kan_Knda)": "kan_Knda",
    "Gujarati (guj_Gujr)": "guj_Gujr",
    "Telugu (tel_Telu)": "tel_Telu",
    "Punjabi (pan_Guru)": "pan_Guru",
    "Odia (ory_Orya)": "ory_Orya",
    "Sindhi (snd_Arab)": "snd_Arab",
    "Assamese (asm_Beng)": "asm_Beng",
    "Bhojpuri (bho_Deva)": "bho_Deva",
    "Maithili (mai_Deva)": "mai_Deva",
    "Sanskrit (san_Deva)": "san_Deva",
    "Konkani (gom_Deva)": "gom_Deva",
    "Bodo (brx_Deva)": "brx_Deva",
    "Dogri (doi_Deva)": "doi_Deva",
}

st.set_page_config(page_title="Video Summarization", layout="wide")
REPO_ROOT = Path(__file__).resolve().parent.parent


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


@st.cache_resource(show_spinner=False)
def cached_load_translation_model(load_4bit: bool, adapter_dir: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_translation_model(
        model_name="ai4bharat/indictrans2-en-indic-1B",
        device=device,
        load_4bit=load_4bit,
        adapter_dir=adapter_dir,
    )


def translate_english_to_indic(
    texts,
    tgt_lang,
    use_context,
    context,
    enable_glossary,
    enable_router,
    tau,
    model,
    tokenizer,
    ip,
):
    """Translate English summary text(s) into `tgt_lang` with the research modules.

    `context` is the video summary used as the B2+ backbone prefix when enabled.
    Glossary terms are extracted from the concatenated English text and
    self-translated; the QE router re-decodes weak/glossary-violated segments.
    Returns (translations, glossary_pairs, qe_report_or_None).
    """
    if not texts:
        return [], None, None

    glossary_pairs = None
    if enable_glossary:
        translate_fn = lambda t: translate_one(model, tokenizer, ip, t, tgt_lang)
        glossary_pairs = glossary_from_text(
            " ".join(texts), translate_fn=translate_fn, backend="ner"
        )

    ctx = context if use_context else None
    if enable_router:
        translations, report = translate_with_qe(
            model, tokenizer, ip, list(texts), tgt_lang,
            context=ctx, glossary_pairs=glossary_pairs, tau=tau,
        )
        return translations, glossary_pairs, report

    translations = translate_texts(
        model, tokenizer, ip, list(texts), tgt_lang,
        context=ctx, glossary_pairs=glossary_pairs,
    )
    return translations, glossary_pairs, None


@st.cache_resource(show_spinner=False)
def cached_load_topic_models(
    miniseg_model_dir: str,
    miniseg_checkpoint: str | None,
    led_base_model_name: str,
    led_hier_dir: str,
):
    miniseg_model, miniseg_tokenizer, saved_threshold, miniseg_config, _, _ = load_miniseg_model(
        model_dir=miniseg_model_dir,
        checkpoint=miniseg_checkpoint,
    )
    led_model, led_tokenizer = load_led_hier_model(
        base_model_name=led_base_model_name,
        hier_dir=led_hier_dir,
    )
    return miniseg_model, miniseg_tokenizer, saved_threshold, miniseg_config, led_model, led_tokenizer


def process_video(
    video_input,
    adapter_choice="hier",
    is_uploaded=False,
    summary_mode="standard",
    target_lang="hin_Deva",
    use_context=False,
    enable_glossary=False,
    enable_router=False,
    load_4bit=False,
    adapter_dir=None,
    qe_tau=-1.5,
):
    videos_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\videos"
    audios_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\audios"
    texts_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\texts"
    summaries_dir = r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\summaries"

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(audios_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    try:
        if is_uploaded:
            st.info("Saving uploaded video...")
            filename = get_file_name_from_path(video_input.name)
            video_path = os.path.join(videos_dir, filename)
            with open(video_path, "wb") as f:
                f.write(video_input.read())

        elif is_youtube_url(video_input):
            st.info("Downloading YouTube video...")
            try:
                downloaded_path = download_video(video_input, output_dir=videos_dir)
            except Exception as e:
                st.error("YouTube download failed: " + str(e))
                st.error(traceback.format_exc())
                return None
            if not downloaded_path or not os.path.exists(downloaded_path):
                st.error("YouTube download produced no output file.")
                return None
            filename = os.path.basename(downloaded_path)
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

    # 3) Load transcript if available, otherwise transcribe (Whisper)
    try:
        transcript_filename = os.path.splitext(audio_filename)[0] + ".txt"
        transcript_path = os.path.join(texts_dir, transcript_filename)

        if os.path.exists(transcript_path):
            st.info("Using existing transcript from texts folder...")
            with open(transcript_path, "r", encoding="utf-8") as transcript_file:
                transcript = transcript_file.read().strip()
            print(f"Using existing transcript: {transcript_path}")
            if transcript:
                st.success("Loaded existing transcript")
            else:
                st.warning("Existing transcript file is empty.")
        else:
            st.info("Transcribing audio with Whisper (small)...")
            transcript = transcribe_audio(audio_filename, audio_dir=audios_dir, text_dir=texts_dir, model_size="small")
            print(transcript[:300] if transcript else "")
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
        if summary_mode == "topic_aware":
            st.info("Loading MiniSeg + LED models...")
            with st.spinner("Loading topic-aware models..."):
                miniseg_model, miniseg_tokenizer, miniseg_threshold, miniseg_config, led_model, led_tokenizer = (
                    cached_load_topic_models(
                        miniseg_model_dir=str(REPO_ROOT / "models" / "miniseg"),
                        miniseg_checkpoint=None,
                        led_base_model_name="allenai/led-base-16384",
                        led_hier_dir=str(REPO_ROOT / "models" / "led-tib-hier" / "led_hier_adapter_run"),
                    )
                )
            st.success("Topic-aware models loaded")

            st.info("Generating topic-aware summary...")
            with st.spinner("Segmenting transcript and summarizing topics..."):
                topic_result = generate_topic_aware_summaries(
                    text=transcript,
                    miniseg_model=miniseg_model,
                    miniseg_tokenizer=miniseg_tokenizer,
                    miniseg_threshold=miniseg_threshold,
                    miniseg_max_sentences=min(512, miniseg_config.max_sentences),
                    miniseg_max_tokens=128,
                    min_chunk_sentences=2,
                    led_model=led_model,
                    led_tokenizer=led_tokenizer,
                    led_min_length=50,
                    led_max_target_topic=256,
                    led_max_target_global=320,
                    title_backend="t5",
                )
            st.success("Topic-aware summary generated")
            topic_names = [topic["title"] for topic in topic_result["topics"]]
            print(f"Topics generated: {len(topic_names)}")
            if topic_names:
                print("Topic names: " + ", ".join(topic_names))
            else:
                print("Topic names: (none)")

            summary_text = format_text_output(topic_result["global_summary"], topic_result["topics"])
            summary_path = os.path.join(summaries_dir, transcript_filename)
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                summary_file.write(summary_text)
            print(f"Summary saved to: {summary_path}")

            translated_global = None
            translated_topics = []
            glossary = None
            qe_report = None
            if ENABLE_TRANSLATION and target_lang != "eng_Latn":
                try:
                    st.info("Loading translation model (IndicTrans2)...")
                    with st.spinner("Loading translation model..."):
                        tmodel, ttokenizer, tip = cached_load_translation_model(load_4bit, adapter_dir=adapter_dir)
                    st.success("Translation model loaded")

                    topic_texts = [t["summary"] for t in topic_result["topics"]]
                    all_texts = [topic_result["global_summary"]] + topic_texts
                    translations, glossary, qe_report = translate_english_to_indic(
                        all_texts,
                        tgt_lang=target_lang,
                        use_context=use_context,
                        context=topic_result["global_summary"],
                        enable_glossary=enable_glossary,
                        enable_router=enable_router,
                        tau=qe_tau,
                        model=tmodel,
                        tokenizer=ttokenizer,
                        ip=tip,
                    )
                    if translations:
                        translated_global = translations[0]
                        translated_topics = translations[1:]
                except Exception as e:
                    st.warning("Translation step failed: " + str(e))

            return {
                "mode": "topic_aware",
                "global_summary": topic_result["global_summary"],
                "translated_global_summary": translated_global,
                "topics": topic_result["topics"],
                "translated_topics": translated_topics,
                "glossary": glossary,
                "qe_report": qe_report,
                "summary_text": summary_text,
                "summary_filename": transcript_filename,
                "summary_path": summary_path,
            }

        st.info("Loading summarization model...")
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
        summary_text = summary or ""
        summary_path = os.path.join(summaries_dir, transcript_filename)
        with open(summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary_text)
        print(f"Summary saved to: {summary_path}")
        translated_summary = None
        glossary = None
        qe_report = None
        if target_lang != "eng_Latn":
            try:
                st.info("Loading translation model (IndicTrans2)...")
                with st.spinner("Loading translation model..."):
                    tmodel, ttokenizer, tip = cached_load_translation_model(load_4bit, adapter_dir=adapter_dir)
                st.success("Translation model loaded")

                translations, glossary, qe_report = translate_english_to_indic(
                    [summary_text],
                    tgt_lang=target_lang,
                    use_context=use_context,
                    context=summary_text,
                    enable_glossary=enable_glossary,
                    enable_router=enable_router,
                    tau=qe_tau,
                    model=tmodel,
                    tokenizer=ttokenizer,
                    ip=tip,
                )
                if translations:
                    translated_summary = translations[0]
            except Exception as e:
                st.warning("Translation step failed: " + str(e))

        return {
            "mode": "standard",
            "summary_text": summary_text,
            "translated_summary": translated_summary,
            "glossary": glossary,
            "qe_report": qe_report,
            "summary_filename": transcript_filename,
            "summary_path": summary_path,
        }

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
    summary_mode_human = st.selectbox("Summary mode", ["Standard LED", "Topic-aware (MiniSeg + LED)"])
    summary_mode = "topic_aware" if summary_mode_human.startswith("Topic-aware") else "standard"
    adapter_choice_human = st.selectbox(
        "Choose adapter",
        ["Hierarchical", "LoRA-only"],
        disabled=(summary_mode == "topic_aware"),
    )
    adapter_choice = "hier" if adapter_choice_human.startswith("Hierarchical") else "lora"

    # Translation temporarily disabled (summarize-only mode). Defaults are set here
    # so the pipeline call signature stays unchanged; the widgets below only appear
    # when ENABLE_TRANSLATION is toggled back on.
    target_lang = "eng_Latn"
    use_context = False
    enable_glossary = False
    enable_router = False
    load_4bit = True
    adapter_dir = None

    if ENABLE_TRANSLATION:
        target_lang_label = st.selectbox("Translate summary to", list(LANG_LABELS.keys()))
        target_lang = LANG_LABELS[target_lang_label]
        translate_enabled = target_lang != "eng_Latn"

        # Translation model variant (Base zero-shot / B2 / B2+). Fine-tuned adapters are
        # produced by train_indictrans2.py; fall back to base if the adapter dir is absent.
        variant_choice = st.selectbox(
            "Translation model", ["Base (zero-shot)", "B2 fine-tune", "B2+ context"]
        )
        if variant_choice == "B2 fine-tune":
            adapter_dir = os.path.join(REPO_ROOT, "models", "indictrans2-en-indic-1B-qlora")
        elif variant_choice == "B2+ context":
            adapter_dir = os.path.join(REPO_ROOT, "models", "indictrans2-en-indic-1B-qlora-ctx")
        else:
            adapter_dir = None
        if adapter_dir is not None and not os.path.isdir(adapter_dir):
            st.warning(f"Adapter not found at {adapter_dir}; using base model.")
            adapter_dir = None

        use_context = st.checkbox("Use summary context (B2+ backbone)", value=False,
                                  disabled=not translate_enabled)
        enable_glossary = st.checkbox("Inject glossary (NER -> self-translate)", value=False,
                                      disabled=not translate_enabled)
        enable_router = st.checkbox("QE router (re-decode weak segments)", value=False,
                                    disabled=not translate_enabled)
        load_4bit = st.checkbox("Load translation model in 4-bit", value=True,
                                disabled=not translate_enabled)

    st.markdown("**Model options:**")
    st.write(f"- Mode: **{summary_mode_human}**")
    if summary_mode == "topic_aware":
        st.write("- Segmentation: **MiniSeg**")
        st.write("- Summarizer: **LED + Hierarchical adapter**")
    else:
        st.write(f"- Adapter: **{adapter_choice_human}**")
    if ENABLE_TRANSLATION:
        st.write(f"- Translation: **{target_lang}**")

st.markdown("---")

if st.button("Process Video"):
    if option == "YouTube URL" and (not video_input or not video_input.strip()):
        st.error("Please provide a YouTube URL.")
    elif option == "Local file upload" and not video_input:
        st.error("Please upload a local video file.")
    else:
        result = process_video(
            video_input,
            adapter_choice=adapter_choice,
            is_uploaded=(option != "YouTube URL"),
            summary_mode=summary_mode,
            target_lang=target_lang,
            use_context=use_context,
            enable_glossary=enable_glossary,
            enable_router=enable_router,
            load_4bit=load_4bit,
            adapter_dir=adapter_dir,
        )
        if result:
            if result["mode"] == "topic_aware":
                st.subheader("Global Summary")
                st.write(result["global_summary"])
                if result.get("translated_global_summary"):
                    st.subheader(f"Translated Summary ({target_lang})")
                    st.write(result["translated_global_summary"])
                st.subheader("Topic Summaries")
                translated_topics = result.get("translated_topics") or []
                for idx, topic in enumerate(result["topics"]):
                    st.markdown(f"**{topic['title']}**")
                    st.write(topic["summary"])
                    if idx < len(translated_topics) and translated_topics[idx]:
                        st.markdown(f"*Translation ({target_lang}):* {translated_topics[idx]}")
                    st.markdown("---")
                if result.get("glossary"):
                    st.subheader("Glossary (EN -> target)")
                    st.write(", ".join(f"{en} => {tgt}" for en, tgt in result["glossary"]))
                if result.get("qe_report"):
                    st.subheader("QE router report")
                    st.json(result["qe_report"])
                st.download_button("Download summary", result["summary_text"], file_name=result["summary_filename"])
            else:
                st.subheader("Generated Summary")
                st.write(result["summary_text"])
                if result.get("translated_summary"):
                    st.subheader(f"Translated Summary ({target_lang})")
                    st.write(result["translated_summary"])
                if result.get("glossary"):
                    st.subheader("Glossary (EN -> target)")
                    st.write(", ".join(f"{en} => {tgt}" for en, tgt in result["glossary"]))
                if result.get("qe_report"):
                    st.subheader("QE router report")
                    st.json(result["qe_report"])
                st.download_button("Download summary", result["summary_text"], file_name=result["summary_filename"])
        else:
            st.error("Processing failed; see messages above.")
