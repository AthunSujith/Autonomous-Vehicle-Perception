import streamlit as st
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---------- Helpers ----------
@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)

def robust_text_size(font, text):
    # Work around different Pillow versions
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text) * 6, 11)

def draw_boxes_pil(pil_img: Image.Image, boxes, confs, classes, names, conf_thresh=0.25, draw_labels=True):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box, conf, cls in zip(boxes, confs, classes):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        label = names[int(cls)] if names and int(cls) in names else str(int(cls))
        text = f"{label} {conf:.2f}" if draw_labels else None

        # box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        if text:
            tw, th = robust_text_size(font, text)
            bg_x0 = x1
            bg_y0 = max(0, y1 - th - 6)
            bg_x1 = x1 + tw + 6
            bg_y1 = y1
            draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill="red")
            draw.text((bg_x0 + 3, bg_y0 + 1), text, fill="white", font=font)

    return pil_img

def run_inference_on_image(model, img_bgr, conf_thresh, device, draw_labels):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(source=np.array(img_rgb), imgsz=640, conf=conf_thresh, device=device, verbose=False)
    r = results[0]
    boxes, confs, classes = [], [], []
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
        conf_arr = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, "cpu") else np.array(r.boxes.conf)
        cls_arr = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, "cpu") else np.array(r.boxes.cls)
        for b, c, cl in zip(xyxy, conf_arr, cls_arr):
            boxes.append(b.tolist()); confs.append(float(c)); classes.append(int(cl))
    names = getattr(model, "names", None)
    pil = Image.fromarray(img_rgb)
    pil = draw_boxes_pil(pil, boxes, confs, classes, names, conf_thresh=conf_thresh, draw_labels=draw_labels)
    counts = {}
    for cl in classes:
        key = names[int(cl)] if names and int(cl) in names else str(int(cl))
        counts[key] = counts.get(key, 0) + 1
    return np.array(pil), counts, len(boxes)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Media Mode YOLO Demo", layout="wide")
st.title("Autonomous Vehicle Perception — Demo App")

# Sidebar controls
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Mode", ["Image", "Video file", "Live webcam"], index=0)
weights = st.sidebar.text_input("Weights path (.pt)", value="yolov8n.pt")
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
draw_labels = st.sidebar.checkbox("Draw labels", value=True)
save_output = st.sidebar.checkbox("Save annotated video (video/webcam)", value=False)
output_name = st.sidebar.text_input("Output filename", value="annotated_output.mp4")
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: use a GPU (cuda) if available for faster inference.")

# Load model (show helpful messages)
model = None
if weights:
    try:
        if os.path.exists(weights) or weights.endswith(".pt"):
            model = load_model(weights)
            st.sidebar.success("Model loaded")
        else:
            st.sidebar.warning("Weights path not found — you can still try built-in 'yolov8n.pt' or provide full path.")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

# layout
colL, colR = st.columns([3,1])
preview = colL.empty()
info = colR.empty()

# Image mode
if mode == "Image":
    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
    if uploaded and model:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, counts, nboxes = run_inference_on_image(model, bgr, conf_thresh, device, draw_labels)
        preview.image(annotated, channels="RGB", use_column_width=True)
        info.markdown(f"**Detections:** {nboxes}\n\n**Top classes:**\n" + "\n".join([f"- {k}: {v}" for k,v in sorted(counts.items(), key=lambda x:-x[1])[:6]]))
    elif uploaded and not model:
        st.warning("Upload model weights path or use default 'yolov8n.pt' in the sidebar.")

# Video file mode
elif mode == "Video file":
    vid_upload = st.file_uploader("Upload video (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"])
    start = st.button("Start processing")
    stop = st.button("Stop")
    if "video_running" not in st.session_state:
        st.session_state.video_running = False
    if start:
        if not vid_upload:
            st.error("Please upload a video first.")
        elif not model:
            st.error("Please provide valid weights.")
        else:
            # save upload to temp file
            tmp_path = f"tmp_{int(time.time())}.mp4"
            with open(tmp_path, "wb") as f:
                f.write(vid_upload.read())
            cap = cv2.VideoCapture(tmp_path)
            writer = None
            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                writer = cv2.VideoWriter(output_name, fourcc, 20.0, (w,h))
                st.sidebar.success(f"Writing to {output_name}")

            st.session_state.video_running = True
            frame_idx = 0
            prev = time.time()
            while cap.isOpened() and st.session_state.video_running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                annotated, counts, nboxes = run_inference_on_image(model, frame, conf_thresh, device, draw_labels)
                # show
                preview.image(annotated, channels="RGB", use_column_width=True)
                info.markdown(f"Frame: {frame_idx}\n\nFPS: {1.0/(time.time()-prev):.1f}\n\nBoxes: {nboxes}\n\n" + 
                              "\n".join([f"- {k}: {v}" for k,v in sorted(counts.items(), key=lambda x:-x[1])[:6]]))
                prev = time.time()
                if writer:
                    writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                # let UI breathe
                time.sleep(0.01)
                # allow user to stop via button
                if stop:
                    st.session_state.video_running = False
                    break

            cap.release()
            if writer:
                writer.release()
                st.success(f"Saved annotated video to {output_name}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if stop:
        st.session_state.video_running = False

# Live webcam mode
elif mode == "Live webcam":
    start_cam = st.button("Start webcam")
    stop_cam = st.button("Stop webcam")
    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    if start_cam:
        if not model:
            st.error("Load model weights first.")
        else:
            st.session_state.cam_running = True

    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.cam_running = False
        else:
            writer = None
            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                writer = cv2.VideoWriter(output_name, fourcc, 20.0, (w,h))
                st.sidebar.success(f"Writing to {output_name}")

            frame_idx = 0
            prev = time.time()
            try:
                while st.session_state.cam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed reading frame.")
                        break
                    frame_idx += 1
                    annotated, counts, nboxes = run_inference_on_image(model, frame, conf_thresh, device, draw_labels)
                    now = time.time()
                    fps = 1.0 / (now - prev) if (now - prev) > 0 else 0.0
                    prev = now

                    preview.image(annotated, channels="RGB", use_column_width=True)
                    info.markdown(f"Frame: {frame_idx}\n\nFPS: {fps:.1f}\n\nBoxes: {nboxes}\n\n" +
                                  "\n".join([f"- {k}: {v}" for k,v in sorted(counts.items(), key=lambda x:-x[1])[:6]]))
                    if writer:
                        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    time.sleep(0.02)  # small sleep so UI can update
            finally:
                cap.release()
                if writer:
                    writer.release()
                    st.success(f"Saved annotated video to {output_name}")
    if stop_cam:
        st.session_state.cam_running = False

st.markdown("---")
st.caption("This app lets the user select Image, Video file, or Live Webcam modes. Adjust confidence and device (cuda/cpu). Use a GPU for faster inference.")
