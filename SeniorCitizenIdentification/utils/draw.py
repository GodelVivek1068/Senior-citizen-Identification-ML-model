"""
utils/draw.py
============================================================
Rendering helpers - draws bounding boxes, ID / age / gender /
senior-citizen labels, FPS counter and timestamp overlay onto
the video frame.
============================================================
"""

from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color_bg=config.COLOR_TEXT_BG):
    """Draws a filled-background text label anchored at (x, y) (top-left of text)."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    y = max(y, th + baseline + 2)
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 4, y + baseline), color_bg, -1)
    cv2.putText(
        frame, text, (x + 2, y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.COLOR_TEXT,
        config.FONT_THICKNESS, cv2.LINE_AA,
    )
    return th + baseline + 4  # height consumed, useful for stacking labels


def draw_person_box(
    frame: np.ndarray,
    bbox: tuple,
    track_id: int,
    age: Optional[int],
    gender: Optional[str],
    is_senior: bool,
):
    """
    Draws the full annotation for one tracked person:
        - Green box if senior citizen, blue box otherwise
        - Stacked labels: ID, Gender | Age, SENIOR CITIZEN tag
    """
    x1, y1, x2, y2 = bbox
    color = config.COLOR_SENIOR if is_senior else config.COLOR_NON_SENIOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    lines = [f"ID: {track_id}"]
    if age is not None and age >= 0 and gender not in (None, "Unknown"):
        lines.append(f"{gender} | Age: {age}")
    else:
        lines.append("Face not visible")
    if is_senior:
        lines.append("SENIOR CITIZEN")

    cursor_y = y1
    for line in lines:
        consumed = draw_label(frame, line, x1, cursor_y, color_bg=color if line == "SENIOR CITIZEN" else config.COLOR_TEXT_BG)
        cursor_y += consumed


def draw_overlay(frame: np.ndarray, fps: float, total_visitors: int, senior_count: int):
    """Draws the global HUD overlay: FPS, timestamp, running counters."""
    h, w = frame.shape[:2]
    y = 25

    if config.SHOW_FPS:
        draw_label(frame, f"FPS: {fps:.1f}", 10, y)
        y += 30

    draw_label(frame, f"Total Visitors: {total_visitors}", 10, y)
    y += 30
    draw_label(frame, f"Senior Citizens: {senior_count}", 10, y)
    y += 30

    if config.SHOW_TIMESTAMP:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS)
        draw_label(frame, ts, w - tw - 20, 25)

    return frame
