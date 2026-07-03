# 🧓 Senior Citizen Identification System

A real-time computer vision application that detects people from a webcam, video file, or
RTSP stream, estimates their age and gender, flags senior citizens (age > 60), overlays
live annotations, and automatically logs de-duplicated visit records to CSV/Excel — all
through a polished, interactive Streamlit dashboard.

---

## ✨ Features

- **Multi-person detection** using Ultralytics YOLO (v8/v10/v11-compatible weights).
- **Persistent multi-object tracking** via ByteTrack, so each person keeps a stable ID.
- **Face detection** with an OpenCV DNN SSD model, automatically falling back to a
  bundled Haar cascade if DNN weights aren't available — the app never hard-crashes
  due to missing face weights.
- **Age & gender estimation** (OpenCV Caffe models by default; optional InsightFace
  backend for numeric age regression).
- **Automatic senior citizen flagging** — green box + "SENIOR CITIZEN" tag for age > 60,
  blue box otherwise.
- **Smart, de-duplicated CSV/Excel logging** — one row per real visit, with a
  configurable reappearance timeout so the same person isn't logged every frame.
- **Advanced Streamlit UI**: live annotated video, real-time FPS/visitor metrics,
  a searchable log table, gender/hourly-visit charts, and one-click CSV/Excel export.
- **Multiple input sources**: webcam, uploaded video file, or RTSP CCTV stream.
- **Graceful error handling** for missing cameras, missing weights, invalid paths,
  and CSV write failures.

---

## 📁 Folder Structure

```
SeniorCitizenIdentification/
│
├── app.py                     # Streamlit UI + main pipeline
├── config.py                  # Central configuration
├── requirements.txt
├── README.md
│
├── models/
│   ├── detector.py            # YOLO person detection + ByteTrack
│   ├── face_utils.py          # Face detection (DNN / Haar fallback)
│   └── age_gender_model.py    # Age & gender estimation
│
├── utils/
│   ├── csv_logger.py          # De-duplicated CSV/Excel logging
│   ├── draw.py                # Bounding box / label rendering
│   ├── fps.py                 # Rolling FPS counter
│   └── helper.py              # Misc helpers (safe crop, source resolution...)
│
├── output/
│   ├── senior_citizen_log.csv # Auto-generated visit log
│   └── output_video.mp4       # Optional saved processed video
│
├── weights/                   # Place all pretrained model weights here
└── assets/                    # Screenshots, sample media, etc.
```

---

## 🔧 Installation

```bash
# 1. Clone / unzip the project, then move into it
cd SeniorCitizenIdentification

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU users:** the default `requirements.txt` installs CPU-compatible `torch`.
> For CUDA acceleration, install the matching `torch`/`torchvision` build from
> https://pytorch.org/get-started/locally/ *before* installing the rest of the
> requirements, then set `DEVICE = "cuda"` (or `"auto"`) in `config.py`.

---

## 📥 Required Model Weights

The YOLO person-detection weight (`yolov8n.pt`) auto-downloads on first run if it
isn't already in `weights/`. The face/age/gender models must be downloaded manually
(they are not redistributed here). Place all files directly inside `weights/`:

| File | Purpose | Where to get it |
|---|---|---|
| `opencv_face_detector.pbtxt` | Face detector config | OpenCV's official `samples/dnn/face_detector` folder on GitHub |
| `opencv_face_detector_uint8.pb` | Face detector weights | Same as above |
| `age_deploy.prototxt` | Age model config | Search "Age and Gender Classification using CNN" (Levi & Hassner) — widely mirrored, e.g. the LearnOpenCV `AgeGender` sample repo |
| `age_net.caffemodel` | Age model weights | Same as above |
| `gender_deploy.prototxt` | Gender model config | Same as above |
| `gender_net.caffemodel` | Gender model weights | Same as above |

If the face-detector files are missing, the app **automatically falls back** to
OpenCV's bundled Haar cascade (lower accuracy, but the app keeps running). If the
age/gender weight files are missing, the app will raise a clear startup error telling
you exactly which files to add.

**Optional — higher-accuracy age/gender (InsightFace):**
```bash
pip install insightface onnxruntime
```
Then set in `config.py`:
```python
AGE_GENDER_BACKEND = "insightface"
```
This uses the `buffalo_l` model bundle (auto-downloaded by the `insightface` package
on first use) and performs true numeric age regression instead of bucketed age ranges.

> ⚠️ **Accuracy note:** the default OpenCV Caffe age model predicts an **age bucket**
> (e.g. `(48-53)`), not an exact year. The app reports the bucket's midpoint as the
> displayed/logged age. For genuinely precise ages, use the InsightFace backend.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then, in the browser dashboard that opens:

1. Choose an **input source** in the sidebar (Webcam / Video File / RTSP Stream).
2. Adjust detection confidence, age threshold, and logging options as needed.
3. Click **▶ Start**. The annotated live feed, FPS, and visitor counters appear
   immediately; the log table and charts refresh automatically.
4. Click **⏹ Stop** to end the session. The CSV log persists in `output/`.
5. Use the **Download CSV** / **Export as Excel** buttons to grab the log file.

You can also run the pipeline headlessly (no UI) by importing the modules directly,
e.g. in a custom script:

```python
from models.detector import PersonDetector
from models.face_utils import FaceDetector
from models.age_gender_model import AgeGenderEstimator
from utils.csv_logger import CSVLogger
```

---

## 🗂 CSV Log Format

`output/senior_citizen_log.csv`:

| Serial Number | Tracking ID | Estimated Age | Gender | Senior Citizen (Yes/No) | Date | Time | Timestamp |
|---|---|---|---|---|---|---|---|
| 1 | ID-5 | 68 | Male | Yes | 2026-07-03 | 11:45:20 | 2026-07-03 11:45:20 |

A row is written **once per real visit**: the same tracking ID will not create
duplicate rows on every frame, and a person who leaves and returns after the
configured `REAPPEAR_TIMEOUT_SECONDS` (default 5 minutes) is logged again as a
new visit.

---

## 🛠 Configuration Reference (`config.py`)

| Setting | Description |
|---|---|
| `CAMERA_INDEX` | Default webcam device index |
| `DETECTION_CONFIDENCE_THRESHOLD` | Minimum YOLO confidence to keep a detection |
| `AGE_THRESHOLD` | Age above which a person is flagged "senior citizen" |
| `REAPPEAR_TIMEOUT_SECONDS` | Gap after which a returning person is logged again |
| `AGE_GENDER_BACKEND` | `"opencv"` (default) or `"insightface"` |
| `ENABLE_TRACKING` | Toggle ByteTrack on/off |
| `SAVE_CSV` / `SAVE_VIDEO` | Toggle output artifacts |
| `OUTPUT_DIR` / `WEIGHTS_DIR` / `ASSETS_DIR` | Storage locations |

All of these are also exposed live in the Streamlit sidebar, so you rarely need to
edit `config.py` directly — sidebar values override the defaults for that session.

---

## 🖼 Sample Screenshots

_Add screenshots of the running dashboard here, e.g._:
```
assets/screenshot_dashboard.png
assets/screenshot_detection.png
```

---

## 🩹 Troubleshooting

| Problem | Likely Cause / Fix |
|---|---|
| `Could not open webcam at index 0` | Camera is in use by another app, or wrong index — try `1`, `2`, etc. |
| `Missing required weight files` on start | Download the age/gender Caffe files listed above into `weights/` |
| Face boxes rarely detected | Using the Haar fallback — download the DNN face weights for better accuracy |
| Very low FPS on CPU | Use the `yolov8n.pt` (nano) model, reduce `DETECTION_IMG_SIZE`, or enable GPU (`DEVICE="cuda"`) |
| RTSP stream won't connect | Verify URL/credentials, and that the network allows the RTSP port (usually 554) |
| CSV not updating | Confirm **Save CSV Log** is checked in the sidebar before clicking Start |
| `ultralytics` fails to import tracker config | Run `pip install lapx` (required by ByteTrack) |

---

## 🚀 Future Improvements

- Multi-camera grid view (process several RTSP feeds concurrently).
- Face re-identification (embeddings) to reliably recognize the same senior citizen
  across camera restarts, not just within one tracker session.
- Alerting (SMS/email/webhook) when a senior citizen is detected, for staff assistance.
- Dashboard analytics page: daily/weekly visitor trends, peak-hour heatmaps.
- Dockerfile + docker-compose for one-command deployment.
- Edge deployment via ONNX/TensorRT export for higher FPS on embedded devices.

---

## ⚖️ Ethical & Privacy Note

This system processes video of real people in public/semi-public spaces to estimate
demographic attributes. Before deploying it, ensure compliance with local privacy,
surveillance, and data-protection regulations (e.g. signage disclosing camera use,
data retention limits, and lawful basis for processing). Consider anonymizing or
purging the CSV log periodically and restricting access to authorized staff only.
