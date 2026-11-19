import os
import subprocess

def extract_audio_ffmpeg(video_name, video_dir="./videos", audio_dir="./audios"):
    """
    Extract audio from a single video and save it
    """
    os.makedirs(audio_dir, exist_ok=True)

    video_path = os.path.join(video_dir, video_name)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_name = os.path.splitext(video_name)[0] + ".mp3"
    audio_path = os.path.join(audio_dir, audio_name)

    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
        ], check=True, capture_output=True)
        print(f"Audio extracted and saved to: {audio_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr.decode(errors='ignore') if e.stderr else e}")

# Example usage:
# If the video file is ./video/my_video.mp4
# extract_audio_ffmpeg("Introduction_to_NLP__NLP_Lecture_1__End_to_End_NLP_Course.mp4")
