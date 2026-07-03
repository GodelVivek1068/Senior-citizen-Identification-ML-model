"""
config.py
============================================================
Central configuration for the Senior Citizen Identification
System. Every tunable parameter used across the project is
defined here so nothing is hard-coded deep inside the
pipeline. Import this module wherever settings are needed:

    import config
    print(config.AGE_THRESHOLD)
============================================================
"""

import os

# ------------------------------------------------------------------
# BASE PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# INPUT SOURCE
# ------------------------------------------------------------------
# "webcam" | "video" | "rtsp"
DEFAULT_SOURCE_TYPE = "webcam"
CAMERA_INDEX = 0
DEFAULT_VIDEO_PATH = ""          # populated at runtime for uploaded videos
DEFAULT_RTSP_URL = ""            # e.g. rtsp://user:pass@ip:554/stream1

# ------------------------------------------------------------------
# PERSON DETECTION (YOLO)
# ------------------------------------------------------------------
# Any Ultralytics-compatible weight name/path works.
# "yolov8n.pt" auto-downloads on first run if not present in WEIGHTS_DIR.
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, YOLO_MODEL_NAME)

PERSON_CLASS_ID = 0              # COCO class id for "person"
DETECTION_CONFIDENCE_THRESHOLD = 0.45
DETECTION_IMG_SIZE = 640
DEVICE = "cpu"                   # "cpu" | "cuda" | "0" (gpu index) - auto-detected in detector.py if left as "auto"

# ------------------------------------------------------------------
# TRACKING (ByteTrack via Ultralytics)
# ------------------------------------------------------------------
ENABLE_TRACKING = True
TRACKER_CONFIG = "bytetrack.yaml"   # shipped with ultralytics
# Seconds a track can be "lost" (no detection) before we consider the
# person to have left the scene. Used for local track bookkeeping.
TRACK_LOST_GRACE_SECONDS = 3

# ------------------------------------------------------------------
# FACE DETECTION
# ------------------------------------------------------------------
# Primary backend: OpenCV DNN SSD (res10). Falls back to Haar cascade
# automatically if the DNN weight files are not found in WEIGHTS_DIR.
FACE_PROTO = os.path.join(WEIGHTS_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(WEIGHTS_DIR, "opencv_face_detector_uint8.pb")
FACE_CONFIDENCE_THRESHOLD = 0.6
FACE_PADDING = 15  # pixels of padding added around detected face box

# ------------------------------------------------------------------
# AGE / GENDER ESTIMATION
# ------------------------------------------------------------------
# "opencv"      -> Caffe Levi-Hassner age/gender nets (default, lightweight)
# "insightface" -> buffalo_l model, numeric age regression (optional, heavier)
AGE_GENDER_BACKEND = "insightface"

AGE_PROTO = os.path.join(WEIGHTS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(WEIGHTS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(WEIGHTS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(WEIGHTS_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# The classic Caffe age model predicts age BUCKETS, not an exact integer.
# We map each bucket to its midpoint for display purposes. This is a
# known accuracy limitation - documented in README.md.
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_BUCKET_MIDPOINTS = [1, 5, 10, 18, 28, 40, 50, 70]

GENDER_LIST = ["Male", "Female"]

# ------------------------------------------------------------------
# SENIOR CITIZEN LOGIC
# ------------------------------------------------------------------
AGE_THRESHOLD = 60               # age > this value => senior citizen

# ------------------------------------------------------------------
# DATA LOGGING
# ------------------------------------------------------------------
SAVE_CSV = True
CSV_FILENAME = "senior_citizen_log.csv"
CSV_PATH = os.path.join(OUTPUT_DIR, CSV_FILENAME)

CSV_COLUMNS = [
    "Serial Number",
    "Tracking ID",
    "Estimated Age",
    "Gender",
    "Senior Citizen (Yes/No)",
    "Date",
    "Time",
    "Timestamp",
]

# If a tracking ID disappears and reappears after this many seconds,
# treat it as a brand-new visit and log it again.
REAPPEAR_TIMEOUT_SECONDS = 300  # 5 minutes

# ------------------------------------------------------------------
# VIDEO OUTPUT
# ------------------------------------------------------------------
SAVE_VIDEO = False
OUTPUT_VIDEO_FILENAME = "output_video.mp4"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_FILENAME)
OUTPUT_VIDEO_FOURCC = "mp4v"
OUTPUT_VIDEO_FPS = 20

# ------------------------------------------------------------------
# DISPLAY / DRAWING
# ------------------------------------------------------------------
COLOR_SENIOR = (0, 200, 0)       # Green (BGR) - senior citizen
COLOR_NON_SENIOR = (255, 120, 0)  # Blue-ish (BGR) - non-senior
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
BOX_THICKNESS = 2
FONT_SCALE = 0.55
FONT_THICKNESS = 1

SHOW_FPS = True
SHOW_TIMESTAMP = True

# ------------------------------------------------------------------
# MISC
# ------------------------------------------------------------------
APP_TITLE = "Senior Citizen Identification System"
LOG_LEVEL = "INFO"
