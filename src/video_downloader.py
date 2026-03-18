from pytubefix import YouTube
import os
from urllib.parse import urlparse, parse_qs
import re

def extract_video_id(url):
    """
    Extract the YouTube video ID from any URL format.
    """
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
    raise ValueError("Invalid YouTube URL")

def download_video(url, output_dir="./videos", filename=None):
    try:
        yt = YouTube(url)

        if(filename==None):
            filename = yt.title.strip().replace(' ', '_')+".mp4"
        print(filename)
        print(f"\nTitle: {yt.title}")
        stream = yt.streams.get_lowest_resolution()
        
        print(stream)

        os.makedirs(output_dir, exist_ok=True)
        output_path = stream.download(output_path=output_dir, filename=filename)
        print(f"Downloaded: {output_path}")

    except Exception as e:
        print(f"Error: {e}")



download_video("https://www.youtube.com/watch?v=ZJ2rmi4wSsk")