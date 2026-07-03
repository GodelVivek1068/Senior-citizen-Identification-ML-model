"""
models/age_gender_model.py
============================================================
Age & gender estimation.

Two interchangeable backends, selected via config.AGE_GENDER_BACKEND:

  "opencv"      (default)
      Levi-Hassner Caffe models (age_net.caffemodel / gender_net.caffemodel).
      Lightweight, CPU-friendly, no extra installs beyond OpenCV.
      IMPORTANT LIMITATION: this model predicts an AGE BUCKET
      (e.g. "(48-53)"), not an exact integer. We report the bucket's
      midpoint as the "estimated age" for display/logging purposes.
      This is a genuine accuracy trade-off - see README.md.

  "insightface" (optional, more accurate)
      Uses the `insightface` buffalo_l model which performs true
      numeric age regression + gender classification from a face
      crop. Requires `pip install insightface onnxruntime`.
      Imported lazily so the base project runs without it installed.

Public API
----------
AgeGenderEstimator.predict(face_crop) -> (age: int, gender: str)
============================================================
"""

import logging
from typing import Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class AgeGenderEstimator:
    def __init__(self, backend: str = config.AGE_GENDER_BACKEND):
        self.backend = backend

        if backend == "opencv":
            self._init_opencv_backend()
        elif backend == "insightface":
            self._init_insightface_backend()
        else:
            raise ValueError(
                f"Unknown AGE_GENDER_BACKEND '{backend}'. "
                f"Expected 'opencv' or 'insightface'."
            )

    # ------------------------------------------------------------------
    # OpenCV Caffe backend
    # ------------------------------------------------------------------
    def _init_opencv_backend(self):
        import os

        missing = [
            p for p in [
                config.AGE_PROTO, config.AGE_MODEL,
                config.GENDER_PROTO, config.GENDER_MODEL,
            ] if not os.path.isfile(p)
        ]
        if missing:
            raise RuntimeError(
                "[AgeGenderEstimator] Missing required weight files:\n  "
                + "\n  ".join(missing)
                + "\nDownload instructions are in README.md -> Weight Files."
            )

        try:
            self.age_net = cv2.dnn.readNet(config.AGE_MODEL, config.AGE_PROTO)
            self.gender_net = cv2.dnn.readNet(config.GENDER_MODEL, config.GENDER_PROTO)
        except Exception as exc:
            raise RuntimeError(
                f"[AgeGenderEstimator] Failed to load OpenCV DNN age/gender "
                f"models: {exc}"
            ) from exc

    def _predict_opencv(self, face_crop: np.ndarray) -> Tuple[int, str]:
        blob = cv2.dnn.blobFromImage(
            face_crop, 1.0, (227, 227),
            config.MODEL_MEAN_VALUES, swapRB=False,
        )

        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = config.GENDER_LIST[int(gender_preds[0].argmax())]

        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        bucket_idx = int(age_preds[0].argmax())
        age = config.AGE_BUCKET_MIDPOINTS[bucket_idx]

        return age, gender

    # ------------------------------------------------------------------
    # InsightFace backend (optional, lazy import)
    # ------------------------------------------------------------------
    def _init_insightface_backend(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "[AgeGenderEstimator] backend='insightface' requires the "
                "`insightface` and `onnxruntime` packages. Install with:\n"
                "    pip install insightface onnxruntime\n"
                "Or switch config.AGE_GENDER_BACKEND back to 'opencv'."
            ) from exc

        self._insightface_app = FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self._insightface_app.prepare(ctx_id=-1, det_size=(320, 320))

    def _predict_insightface(self, face_crop: np.ndarray) -> Tuple[int, str]:
        faces = self._insightface_app.get(face_crop)
        if not faces:
            # No face re-detected inside the (already-cropped) region.
            return -1, "Unknown"

        face = max(faces, key=lambda f: f.det_score)
        age = int(round(face.age))
        gender = "Male" if int(face.gender) == 1 else "Female"
        return age, gender

    # ------------------------------------------------------------------
    # Unified public method
    # ------------------------------------------------------------------
    def predict(self, face_crop: np.ndarray) -> Tuple[int, str]:
        """
        Parameters
        ----------
        face_crop : np.ndarray
            BGR cropped face image.

        Returns
        -------
        (age, gender) : (int, str)
            age is the model's best estimate (bucket midpoint for the
            'opencv' backend, or a direct regression value for
            'insightface'). gender is one of "Male" / "Female".
        """
        if face_crop is None or face_crop.size == 0:
            return -1, "Unknown"

        try:
            if self.backend == "opencv":
                return self._predict_opencv(face_crop)
            return self._predict_insightface(face_crop)
        except Exception as exc:
            logger.error(f"Age/Gender prediction failed: {exc}")
            return -1, "Unknown"
