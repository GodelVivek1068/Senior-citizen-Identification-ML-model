"""
utils/helper.py
============================================================
Miscellaneous helper functions used across the project:
    - Safe directory creation
    - Frame cropping with bounds checking
    - Video source resolution / validation
    - Weight file presence checks
============================================================
"""

import logging
import os
from typing import Optional, Union

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_crop(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """
    Crops `frame` to `bbox` = (x1, y1, x2, y2), clamped to frame bounds.
    Returns None if the resulting crop would be empty/invalid.
    """
    if frame is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def resolve_video_capture(
    source_type: str,
    camera_index: int = config.CAMERA_INDEX,
    video_path: str = "",
    rtsp_url: str = "",
) -> cv2.VideoCapture:
    """
    Opens and returns a cv2.VideoCapture for the requested source type.

    Raises
    ------
    RuntimeError with a clear, user-facing message on failure, so the
    calling UI layer can surface it instead of crashing silently.
    """
    if source_type == "webcam":
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"[Error] Could not open webcam at index {camera_index}. "
                f"Check that the camera is connected and not in use by "
                f"another application."
            )
        return cap

    if source_type == "video":
        if not video_path or not os.path.isfile(video_path):
            raise RuntimeError(
                f"[Error] Invalid video path: '{video_path}'. "
                f"Please provide a valid, existing video file."
            )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"[Error] Could not open video file: '{video_path}'. "
                f"The file may be corrupted or in an unsupported format."
            )
        return cap

    if source_type == "rtsp":
        if not rtsp_url:
            raise RuntimeError("[Error] RTSP URL is empty. Please provide a valid RTSP stream URL.")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(
                f"[Error] Could not connect to RTSP stream: '{rtsp_url}'. "
                f"Verify the URL, credentials, and network connectivity."
            )
        return cap

    raise ValueError(f"Unknown source_type '{source_type}'. Expected webcam/video/rtsp.")


def check_required_weights(paths: list) -> list:
    """Returns the subset of `paths` that do NOT exist on disk."""
    return [p for p in paths if not os.path.isfile(p)]


def format_bbox_label(track_id: int, age: int, gender: str, is_senior: bool) -> str:
    status = "SENIOR CITIZEN" if is_senior else ""
    return f"ID:{track_id} {gender} Age:{age} {status}".strip()
