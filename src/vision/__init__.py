"""Vision processing module for Multimodal Lecture Discourse Segmentation."""

from src.vision.keyframe_extractor import LectureKeyframeExtractor, KeyframeInfo
from src.vision.slide_encoder import QwenSlideRepresentationEncoder

__all__ = ["LectureKeyframeExtractor", "KeyframeInfo", "QwenSlideRepresentationEncoder"]

