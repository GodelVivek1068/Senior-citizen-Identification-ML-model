"""
models/detector.py
============================================================
Person detection + multi-object tracking.

Uses Ultralytics YOLO (v8/v10/v11 weights are all compatible -
just point config.YOLO_MODEL_NAME to whichever .pt file you want)
for detection, and Ultralytics' built-in ByteTrack implementation
for assigning persistent tracking IDs across frames.

Public API
----------
PersonDetector.detect_and_track(frame) -> List[Detection]

Each Detection is a simple namedtuple-like object exposing:
    track_id : int
    bbox     : (x1, y1, x2, y2)
    conf     : float
============================================================
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a single tracked person in the current frame."""
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2) in pixel coordinates
    conf: float


class PersonDetector:
    """
    Wraps an Ultralytics YOLO model configured to detect and track
    only the 'person' class, using ByteTrack for identity persistence.
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        conf_threshold: float = config.DETECTION_CONFIDENCE_THRESHOLD,
        device: str = config.DEVICE,
        enable_tracking: bool = config.ENABLE_TRACKING,
    ):
        self.conf_threshold = conf_threshold
        self.enable_tracking = enable_tracking
        self.device = self._resolve_device(device)

        try:
            # If a local weight file exists, Ultralytics will use it directly.
            # If it doesn't exist, Ultralytics will try to auto-download the
            # matching pretrained checkpoint (requires internet access once).
            load_target = model_path if os.path.isfile(model_path) else config.YOLO_MODEL_NAME
            logger.info(f"Loading YOLO model from: {load_target}")
            self.model = YOLO(load_target)
            self.model.to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"[PersonDetector] Failed to load YOLO weights "
                f"('{model_path}'). Ensure the file exists in the weights/ "
                f"folder or that you have internet access for auto-download. "
                f"Original error: {exc}"
            ) from exc

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available - falling back to CPU.")
            return "cpu"
        return device

    def detect_and_track(self, frame) -> List[Detection]:
        """
        Runs detection (+ tracking, if enabled) on a single BGR frame.

        Returns
        -------
        List[Detection]
        """
        if frame is None:
            return []

        try:
            if self.enable_tracking:
                results = self.model.track(
                    frame,
                    persist=True,
                    tracker=config.TRACKER_CONFIG,
                    classes=[config.PERSON_CLASS_ID],
                    conf=self.conf_threshold,
                    imgsz=config.DETECTION_IMG_SIZE,
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    frame,
                    classes=[config.PERSON_CLASS_ID],
                    conf=self.conf_threshold,
                    imgsz=config.DETECTION_IMG_SIZE,
                    verbose=False,
                )
        except Exception as exc:
            logger.error(f"Detection/tracking inference failed: {exc}")
            return []

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        has_ids = self.enable_tracking and boxes.id is not None

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            track_id = int(boxes.id[i].cpu().numpy()) if has_ids else -1

            detections.append(
                Detection(
                    track_id=track_id,
                    bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    conf=conf,
                )
            )

        return detections
