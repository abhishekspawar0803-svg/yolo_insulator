import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import subprocess
import shutil

# Must be the first Streamlit command
st.set_page_config(page_title="Insulator Defect Detection", layout="wide")

st.markdown("""
<style>
    .block-container {
        max-width: 99% !important;
        padding-top: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
    }

    section[data-testid="stSidebar"] {
        width: 235px !important;
        min-width: 235px !important;
    }

    div[data-testid="column"] {
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }

    .stImage img {
        border-radius: 10px;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
        margin-top: 0.6rem;
        margin-bottom: 0.25rem;
        padding-top: 0.1rem;
    }

    .subtle-text {
        color: #b8b8b8;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Config ---
MODEL_PATHS = {
    "YOLOv8n": "models\\v8_best.pt",
    "YOLOv11n": "models\\v11_best.pt",
    "YOLOv26n": "models\\v26_best.pt"
}

@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = YOLO(path)
        except Exception:
            st.warning(f"Could not load {name}. Check the file path.")
    return models

models = load_models()

st.markdown('<div class="main-title">⚡ Insulator Defect Detection (YOLO Comparison)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle-text">Comparing YOLOv8n, YOLOv11n, and YOLOv26n on power line insulators.</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Mode", ["Image Comparison", "Video Processing"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

def plot_result_rgb(result):
    plotted = result.plot()
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

def process_image(img_array, model, conf):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, conf=conf, verbose=False)
    return plot_result_rgb(results[0])

def process_video_to_file(input_video_path, model, conf, output_path):
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    temp_raw = output_path.replace(".mp4", "_raw.avi")
    writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        plotted_bgr = results[0].plot()

        # Create writer only after first plotted frame so dimensions match exactly
        if writer is None:
            h, w = plotted_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(temp_raw, fourcc, fps, (w, h), True)

            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not open VideoWriter for {temp_raw}")

        writer.write(plotted_bgr)

    cap.release()

    if writer is not None:
        writer.release()
    else:
        raise RuntimeError("No frames were written to the output video.")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is installed but not available in PATH for this Python process.")

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", temp_raw,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
        
# --- MODE 1: Image Comparison ---
if mode == "Image Comparison":
    st.header("Image Comparison Mode")
    st.write("Upload an image for side-by-side model comparison.")

    uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        st.markdown("### Original Image")
        _, center_col, _ = st.columns([0.15, 4.7, 0.15])
        with center_col:
            st.image(image, use_container_width=True)

        st.divider()

        st.markdown("### Model Inference Comparison")
        col1, col2, col3 = st.columns([1, 1, 1], gap="small")

        frame8 = process_image(img_array, models["YOLOv8n"], conf_thresh) if "YOLOv8n" in models else None
        frame11 = process_image(img_array, models["YOLOv11n"], conf_thresh) if "YOLOv11n" in models else None
        frame26 = process_image(img_array, models["YOLOv26n"], conf_thresh) if "YOLOv26n" in models else None

        with col1:
            st.markdown("### YOLOv8n")
            st.caption("Baseline")
            if frame8 is not None:
                st.image(frame8, use_container_width=True)

        with col2:
            st.markdown("### YOLOv11n")
            st.caption("Top performer")
            if frame11 is not None:
                st.image(frame11, use_container_width=True)

        with col3:
            st.markdown("### YOLOv26n")
            st.caption("Newest model")
            if frame26 is not None:
                st.image(frame26, use_container_width=True)

# --- MODE 2: Video Processing ---
elif mode == "Video Processing":
    st.header("Video Processing Mode")
    st.write("Upload a video, process it once, and compare the rendered outputs.")

    video_file = st.file_uploader("Upload Video...", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()
        input_video_path = tfile.name

        st.markdown("### Original Video")
        _, top_center, _ = st.columns([0.15, 4.7, 0.15])
        with top_center:
            st.video(input_video_path)

        if st.button("Start Processing"):
            st.info("Processing started. Please wait while all 3 models process video...")

            try:
                with st.spinner("Running inference on the uploaded video..."):
                    progress = st.progress(0, text="Preparing video processing...")

                    base_dir = tempfile.mkdtemp()
                    out8 = os.path.join(base_dir, "yolov8n_output.mp4")
                    out11 = os.path.join(base_dir, "yolov11n_output.mp4")
                    out26 = os.path.join(base_dir, "yolov26n_output.mp4")

                    if "YOLOv8n" in models:
                        progress.progress(15, text="Processing YOLOv8n video...")
                        process_video_to_file(input_video_path, models["YOLOv8n"], conf_thresh, out8)

                    if "YOLOv11n" in models:
                        progress.progress(50, text="Processing YOLOv11n video...")
                        process_video_to_file(input_video_path, models["YOLOv11n"], conf_thresh, out11)

                    if "YOLOv26n" in models:
                        progress.progress(85, text="Processing YOLOv26n video...")
                        process_video_to_file(input_video_path, models["YOLOv26n"], conf_thresh, out26)

                    progress.progress(100, text="Processing complete!")

            except Exception as e:
                st.error(f"Video processing failed: {e}")
                st.stop()
            st.divider()
            st.markdown("### Processed Video Comparison")
            col1, col2, col3 = st.columns([1, 1, 1], gap="small")

            with col1:
                st.markdown("### YOLOv8n")
                st.caption("Baseline")
                if os.path.exists(out8):
                    with open(out8, "rb") as f:
                        st.video(f.read())

            with col2:
                st.markdown("### YOLOv11n")
                st.caption("Top performer")
                if os.path.exists(out11):
                    with open(out11, "rb") as f:
                        st.video(f.read())

            with col3:
                st.markdown("### YOLOv26n")
                st.caption("Newest model")
                if os.path.exists(out26):
                    with open(out26, "rb") as f:
                        st.video(f.read())

            st.success("All processed videos are ready.")