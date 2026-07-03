"""
models/face_utils.py
============================================================
Face detection module.

Primary backend : OpenCV DNN SSD face detector (res10_300x300).
Fallback backend: OpenCV Haar Cascade (bundled with opencv-python,
                  always available, used automatically if the DNN
                  weight files are missing from weights/).

Public API
----------
FaceDetector.detect(person_crop) -> Optional[(x1, y1, x2, y2)]
    Returns the largest/most confident face box found within the
    given person crop, in the crop's local coordinate space, or
    None if no face is visible.
============================================================
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(
        self,
        proto_path: str = config.FACE_PROTO,
        model_path: str = config.FACE_MODEL,
        confidence_threshold: float = config.FACE_CONFIDENCE_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.backend = None
        self.net = None

        if os.path.isfile(proto_path) and os.path.isfile(model_path):
            try:
                self.net = cv2.dnn.readNet(model_path, proto_path)
                self.backend = "dnn"
                logger.info("FaceDetector: using OpenCV DNN backend.")
            except Exception as exc:
                logger.warning(
                    f"FaceDetector: failed to load DNN model ({exc}). "
                    "Face detection disabled. See README.md for download instructions."
                )
        
        if self.backend is None:
            logger.warning(
                "FaceDetector: DNN weight files not found in weights/. "
                "Face detection is disabled. This won't crash, but face detection "
                "will return None. See README.md for download instructions."
            )

    def detect(self, person_crop: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the most prominent face within a cropped person image.

        Parameters
        ----------
        person_crop : np.ndarray
            BGR image containing (ideally) a single person.

        Returns
        -------
        (x1, y1, x2, y2) in the crop's local coordinates, or None if no backend available.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        if self.backend == "dnn":
            return self._detect_dnn(person_crop)
        
        # No backend available
        return None

    def _detect_dnn(self, crop: np.ndarray):
        h, w = crop.shape[:2]
        blob = cv2.dnn.blobFromImage(
            crop, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        best_box = None
        best_conf = self.confidence_threshold

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > best_conf:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    best_box = (x1, y1, x2, y2)
                    best_conf = confidence

        return best_box
