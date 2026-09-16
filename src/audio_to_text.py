import os
import torch
from faster_whisper import WhisperModel

def transcribe_audio(audio_name, audio_dir="./audios", text_dir="./texts", model_size="small"):
    """
    Convert a single audio file to text and save the transcript.
    """
    os.makedirs(text_dir, exist_ok=True)

    audio_path = os.path.join(audio_dir, audio_name)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    text_path = os.path.join(text_dir, os.path.splitext(audio_name)[0] + ".txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading Whisper model ({model_size}) on {device}...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )
    print("Transcribing... This may take a while for longer files.")
    segments, info = model.transcribe(
        audio_path,
        task="translate",
        vad_filter=True
    )

    full_text = ""
    for segment in segments:
        full_text += segment.text + " "

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text.strip())

    print(f"Transcription saved to: {text_path}")
    return full_text

# Convert ./audios/sample.mp3 -> ./texts/sample.txt
# transcribe_audio("Introduction_to_NLP__NLP_Lecture_1__End_to_End_NLP_Course.mp3")
