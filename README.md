# 🎓 AI Lecture Notes & Translation Assistant

An end-to-end multimodal AI application that transforms educational lecture videos into comprehensive, structured study notes and Hindi translations using local LLMs.

---

## ✨ Features

- 🎥 **Video Ingestion & Transcription**: Automatically downloads educational videos (or loads local recordings) and converts spoken audio into accurate, punctuated text.
- 📑 **Multimodal Topic Segmentation**: Breaks down long-form lectures into coherent, bite-sized topic sections by analyzing both speech and visual slide transitions.
- 📝 **Comprehensive Study Notes**: Generates high-density study notes with in-depth concept explanations, formal definitions, math formulas, key takeaways, and active recall questions.
- 🌐 **Hindi Translation**: Automatically translates notes into clean, readable Hindi while preserving formulas, code, and core technical terms.
- 💻 **Interactive Web App**: A clean Streamlit interface with live progress tracking and side-by-side English and Hindi study notes.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally (for note generation and translation)
- FFmpeg installed on your system

### 2. Installation
```bash
# Clone the repository
git clone git@github.com:Mauryatan10/Educational-Video-Summarization.git
cd Educational-Video-Summarization

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`, paste any YouTube lecture link (or select a local video), and click **Generate Notes**.

---

## 🛠️ How It Works

```
Video / YouTube URL
       │
       ▼
[1. Speech Recognition] ──► Transcribes lecture audio with timestamps
       │
       ▼
[2. Multimodal Segmentation] ──► Identifies topic boundaries using speech & slides
       │
       ▼
[3. LLM Note Synthesis] ──► Produces comprehensive structured study notes
       │
       ▼
[4. Hindi Translation] ──► Generates bilingual STEM notes with formula protection
       │
       ▼
Interactive Web App (English & Hindi Notes)
```

---

## 📂 Project Structure

```text
├── benchmarks/         # Experimental evaluation results and comparison tables
├── outputs/            # Sample generated lecture notes (Markdown & JSON)
├── scripts/            # Training, evaluation, and data preparation utilities
├── src/
│   ├── app.py          # Streamlit web interface
│   ├── asr/            # Audio processing & speech recognition pipeline
│   ├── segmentation/   # Multimodal discourse topic segmentation
│   ├── pipeline/       # Note generation & translation orchestration
│   └── vision/         # Slide keyframe extraction and visual encoding
└── tests/              # Unit and integration test suite
```
