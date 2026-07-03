"""
app.py
============================================================
Senior Citizen Identification System - Main Application
============================================================
An advanced Streamlit dashboard that ties together:
    - Person detection & tracking   (models/detector.py)
    - Face detection                (models/face_utils.py)
    - Age & gender estimation       (models/age_gender_model.py)
    - Deduplicated CSV/Excel logging(utils/csv_logger.py)
    - Real-time drawing & overlays  (utils/draw.py)
    - FPS measurement               (utils/fps.py)

Run with:
    streamlit run app.py
============================================================
"""

import logging
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from models.age_gender_model import AgeGenderEstimator
from models.detector import PersonDetector
from models.face_utils import FaceDetector
from utils.csv_logger import CSVLogger
from utils.draw import draw_overlay, draw_person_box
from utils.fps import FPSCounter
from utils.helper import resolve_video_capture, safe_crop

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🧓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background: linear-gradient(135deg, #1c2333 0%, #131722 100%);
        border: 1px solid #2b3245;
        border-radius: 12px;
        padding: 12px 8px;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 6px 0 18px 0;
    }
    .app-header h1 {
        font-size: 1.9rem;
        margin: 0;
        background: linear-gradient(90deg, #34d399, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-live { background: #14532d; color: #4ade80; }
    .status-stopped { background: #3f3f46; color: #d4d4d8; }
    div[data-testid="stVideoFrame"], .video-frame-box {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #2b3245;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="app-header">
        <h1>🧓 {config.APP_TITLE}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
DEFAULTS = {
    "running": False,
    "cap": None,
    "video_writer": None,
    "detector": None,
    "face_detector": None,
    "age_gender": None,
    "csv_logger": None,
    "fps_counter": None,
    "total_visitors": set(),
    "senior_visitor_ids": set(),
    "last_error": None,
    "frame_count": 0,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CACHED MODEL LOADERS (heavy objects, load once)
# ============================================================
@st.cache_resource(show_spinner="Loading person detector (YOLO)...")
def load_detector(conf_threshold: float, enable_tracking: bool, device: str):
    return PersonDetector(
        conf_threshold=conf_threshold,
        enable_tracking=enable_tracking,
        device=device,
    )


@st.cache_resource(show_spinner="Loading face detector...")
def load_face_detector(conf_threshold: float):
    return FaceDetector(confidence_threshold=conf_threshold)


@st.cache_resource(show_spinner="Loading age/gender model...")
def load_age_gender(backend: str):
    return AgeGenderEstimator(backend=backend)


# ============================================================
# SIDEBAR - CONTROLS
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Input Source")
    source_type = st.radio(
        "Select source", ["Webcam", "Video File", "RTSP Stream"], horizontal=False
    )

    camera_index = 0
    uploaded_video = None
    rtsp_url = ""

    if source_type == "Webcam":
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=config.CAMERA_INDEX)
    elif source_type == "Video File":
        uploaded_video = st.file_uploader("Upload a video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])
    else:
        rtsp_url = st.text_input("RTSP URL", value=config.DEFAULT_RTSP_URL, placeholder="rtsp://user:pass@ip:554/stream1")

    st.divider()
    st.subheader("Detection Settings")
    conf_threshold = st.slider("Person Detection Confidence", 0.1, 0.95, config.DETECTION_CONFIDENCE_THRESHOLD, 0.05)
    face_conf_threshold = st.slider("Face Detection Confidence", 0.1, 0.95, config.FACE_CONFIDENCE_THRESHOLD, 0.05)
    age_threshold = st.slider("Senior Citizen Age Threshold", 40, 90, config.AGE_THRESHOLD, 1)
    device_choice = st.selectbox("Compute Device", ["cpu", "cuda", "auto"], index=0)

    st.divider()
    st.subheader("Tracking & Logging")
    enable_tracking = st.checkbox("Enable Tracking (ByteTrack)", value=config.ENABLE_TRACKING)
    log_all_visitors = st.checkbox("Log ALL visitors (not just seniors)", value=False)
    reappear_timeout = st.slider("Reappearance Timeout (seconds)", 30, 900, config.REAPPEAR_TIMEOUT_SECONDS, 30)
    save_csv = st.checkbox("Save CSV Log", value=config.SAVE_CSV)
    save_video = st.checkbox("Save Processed Video", value=config.SAVE_VIDEO)

    st.divider()
    col_start, col_stop = st.columns(2)
    start_clicked = col_start.button("▶ Start", use_container_width=True, type="primary")
    stop_clicked = col_stop.button("⏹ Stop", use_container_width=True)

    status_label = "🟢 LIVE" if st.session_state.running else "⚪ STOPPED"
    status_class = "status-live" if st.session_state.running else "status-stopped"
    st.markdown(f'<span class="status-pill {status_class}">{status_label}</span>', unsafe_allow_html=True)


# ============================================================
# START / STOP HANDLERS
# ============================================================
def _release_capture():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    if st.session_state.video_writer is not None:
        st.session_state.video_writer.release()
        st.session_state.video_writer = None


if start_clicked:
    st.session_state.last_error = None
    try:
        # Resolve video source
        video_path = ""
        if source_type == "Video File" and uploaded_video is not None:
            video_path = f"{config.OUTPUT_DIR}/_uploaded_{uploaded_video.name}"
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())

        src_type_map = {"Webcam": "webcam", "Video File": "video", "RTSP Stream": "rtsp"}
        cap = resolve_video_capture(
            source_type=src_type_map[source_type],
            camera_index=camera_index,
            video_path=video_path,
            rtsp_url=rtsp_url,
        )
        st.session_state.cap = cap

        st.session_state.detector = load_detector(conf_threshold, enable_tracking, device_choice)
        st.session_state.face_detector = load_face_detector(face_conf_threshold)
        st.session_state.age_gender = load_age_gender(config.AGE_GENDER_BACKEND)
        st.session_state.csv_logger = CSVLogger(
            reappear_timeout_seconds=reappear_timeout,
            log_all_visitors=log_all_visitors,
        ) if save_csv else None
        st.session_state.fps_counter = FPSCounter()

        if save_video:
            fourcc = cv2.VideoWriter.fourcc(*config.OUTPUT_VIDEO_FOURCC)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            st.session_state.video_writer = cv2.VideoWriter(
                config.OUTPUT_VIDEO_PATH, fourcc, config.OUTPUT_VIDEO_FPS, (frame_w, frame_h)
            )

        st.session_state.running = True
        st.rerun()

    except Exception as exc:
        st.session_state.last_error = str(exc)
        _release_capture()
        st.session_state.running = False

if stop_clicked:
    st.session_state.running = False
    _release_capture()
    st.rerun()

if st.session_state.last_error:
    st.error(st.session_state.last_error)


# ============================================================
# MAIN LAYOUT
# ============================================================
video_col, stats_col = st.columns([2.2, 1])

with video_col:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()
    if not st.session_state.running:
        video_placeholder.info("Configure your source in the sidebar and click **Start** to begin.")

with stats_col:
    st.subheader("📊 Live Metrics")
    m1, m2, m3 = st.columns(3)
    metric_fps = m1.empty()
    metric_total = m2.empty()
    metric_senior = m3.empty()


# ============================================================
# FRAME PROCESSING (single frame) - core pipeline logic
# ============================================================
def process_frame(frame: np.ndarray) -> np.ndarray:
    detector = st.session_state.detector
    face_detector = st.session_state.face_detector
    age_gender = st.session_state.age_gender
    csv_logger = st.session_state.csv_logger

    detections = detector.detect_and_track(frame)

    for det in detections:
        track_id = det.track_id
        if track_id == -1:
            # Tracking disabled - fall back to a synthetic id per detection
            # so downstream logic still works (no dedup guarantees in this mode).
            track_id = hash(det.bbox) % 100000

        if csv_logger is not None:
            csv_logger.update_seen(track_id)

        st.session_state.total_visitors.add(track_id)

        person_crop = safe_crop(frame, det.bbox)
        age, gender, is_senior = -1, "Unknown", False

        if person_crop is not None:
            face_box = face_detector.detect(person_crop)
            if face_box is not None:
                fx1, fy1, fx2, fy2 = face_box
                pad = config.FACE_PADDING
                fx1, fy1 = max(0, fx1 - pad), max(0, fy1 - pad)
                fx2, fy2 = min(person_crop.shape[1], fx2 + pad), min(person_crop.shape[0], fy2 + pad)
                face_crop = person_crop[fy1:fy2, fx1:fx2]

                age, gender = age_gender.predict(face_crop)
                is_senior = age >= 0 and age > age_threshold

                if is_senior:
                    st.session_state.senior_visitor_ids.add(track_id)

                if csv_logger is not None:
                    csv_logger.log_if_new(track_id, age, gender, is_senior)

        draw_person_box(frame, det.bbox, track_id, age if age >= 0 else None, gender, is_senior)

    fps = st.session_state.fps_counter.tick()
    draw_overlay(
        frame, fps,
        total_visitors=len(st.session_state.total_visitors),
        senior_count=len(st.session_state.senior_visitor_ids),
    )

    metric_fps.metric("FPS", f"{fps:.1f}")
    metric_total.metric("Total Visitors", len(st.session_state.total_visitors))
    metric_senior.metric("Senior Citizens", len(st.session_state.senior_visitor_ids))

    if st.session_state.video_writer is not None:
        st.session_state.video_writer.write(frame)

    return frame


# ============================================================
# LIVE VIDEO FRAGMENT (reruns rapidly while `running` is True)
# ============================================================
@st.fragment(run_every=0.03)
def video_fragment():
    if not st.session_state.running:
        return

    cap = st.session_state.cap
    if cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        st.session_state.running = False
        _release_capture()
        video_placeholder.warning("Video stream ended or camera disconnected.")
        return

    try:
        processed = process_frame(frame)
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        st.session_state.frame_count += 1
    except Exception as exc:
        logger.error(f"Frame processing error: {exc}")
        st.session_state.running = False
        _release_capture()
        st.error(f"Processing stopped due to an error: {exc}")


if st.session_state.running:
    video_fragment()


# ============================================================
# STATS / LOG FRAGMENT (refreshes less frequently)
# ============================================================
@st.fragment(run_every=4)
def stats_fragment():
    st.subheader("🗂 Visit Log")

    if st.session_state.csv_logger is None:
        st.caption("Enable **Save CSV Log** in the sidebar to record and view visits here.")
        return

    df = st.session_state.csv_logger.read_log_dataframe()

    tab_table, tab_charts = st.tabs(["Table", "Charts"])

    with tab_table:
        st.dataframe(df, use_container_width=True, height=280)

        col_a, col_b = st.columns(2)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        col_a.download_button(
            "⬇ Download CSV", data=csv_bytes,
            file_name=config.CSV_FILENAME, mime="text/csv",
            use_container_width=True,
        )

        if col_b.button("⬇ Export as Excel", use_container_width=True):
            try:
                path = st.session_state.csv_logger.export_to_excel()
                with open(path, "rb") as f:
                    st.download_button(
                        "Download .xlsx", data=f.read(),
                        file_name=config.CSV_FILENAME.replace(".csv", ".xlsx"),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
            except RuntimeError as exc:
                st.error(str(exc))

    with tab_charts:
        if df.empty:
            st.caption("No data logged yet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                gender_counts = df["Gender"].value_counts().reset_index()
                gender_counts.columns = ["Gender", "Count"]
                fig1 = px.pie(gender_counts, names="Gender", values="Count", title="Gender Distribution", hole=0.45)
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                df["Hour"] = pd.to_datetime(df["Timestamp"]).dt.hour
                hourly = df.groupby("Hour").size().reset_index(name="Visits")
                fig2 = px.bar(hourly, x="Hour", y="Visits", title="Visits by Hour")
                st.plotly_chart(fig2, use_container_width=True)


stats_fragment()

st.divider()
st.caption(
    "Senior Citizen Identification System · Built with YOLO + ByteTrack + OpenCV · "
    "Ensure pretrained weights are placed in weights/ before starting (see README.md)."
)
