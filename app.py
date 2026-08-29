"""
🐾 Animal Intrusion Detection AI - 24/7 Streamlit Cloud Web Application
Mobile & Desktop friendly. Allows farmers, clients, and friends to identify animals in fields.
"""

import os
import time
from typing import List, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Animal Intrusion Detection AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, mobile-friendly card styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; height: 3rem; background: linear-gradient(90deg, #dc2626, #ea580c); color: white; border: none; }
    .stMetric { background-color: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "best.onnx"
DEFAULT_CLASSES = ["buffalo", "elephant", "rhino", "zebra", "deer", "wild_boar", "monkey", "cow", "human"]
CLASS_COLORS = [
    (235, 120, 30),   # Orange
    (30, 30, 220),    # Red
    (220, 50, 50),    # Blue
    (50, 180, 50),    # Green
    (200, 200, 30),   # Yellow
    (180, 50, 200),   # Purple
]


# ==============================================================================
# 2. MODEL LOADER (CACHED FOR INSTANT SPEED)
# ==============================================================================
@st.cache_resource
def load_onnx_model():
    """Loads ONNX Runtime session in memory once and reuses across all users."""
    model_file = MODEL_PATH
    if not os.path.exists(model_file):
        for p in ["models/best.onnx", "../models/best.onnx", "best.onnx"]:
            if os.path.exists(p):
                model_file = p
                break

    if not os.path.exists(model_file):
        return None, None, None

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_file, sess_options=sess_options, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


# ==============================================================================
# 3. PREPROCESSING & POSTPROCESSING
# ==============================================================================
def letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640)):
    """Resize & pad image preserving aspect ratio."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, (r, r), (dw, dh)


def decode_yolo_onnx(output_tensor, conf_threshold, iou_threshold, original_shape, ratio, pad):
    """Parses raw YOLO11/YOLOv8 ONNX tensor and applies NMS."""
    preds = np.squeeze(output_tensor)
    if preds.ndim != 2:
        return []
    if preds.shape[0] < preds.shape[1] and preds.shape[0] <= 100:
        preds = preds.transpose()

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = (cx - w / 2.0 - pad[0]) / ratio[0]
    y1 = (cy - h / 2.0 - pad[1]) / ratio[1]
    x2 = (cx + w / 2.0 - pad[0]) / ratio[0]
    y2 = (cy + h / 2.0 - pad[1]) / ratio[1]

    orig_h, orig_w = original_shape[:2]
    x1 = np.clip(x1, 0, orig_w - 1)
    y1 = np.clip(y1, 0, orig_h - 1)
    x2 = np.clip(x2, 0, orig_w - 1)
    y2 = np.clip(y2, 0, orig_h - 1)

    nms_boxes = [[int(x1[i]), int(y1[i]), max(1, int(x2[i] - x1[i])), max(1, int(y2[i] - y1[i]))] for i in range(len(x1))]
    scores_list = [float(c) for c in confidences]

    final_dets = []
    for cls in np.unique(class_ids):
        cls_indices = [i for i, c in enumerate(class_ids) if c == cls]
        try:
            indices = cv2.dnn.NMSBoxes([nms_boxes[i] for i in cls_indices], [scores_list[i] for i in cls_indices], float(conf_threshold), float(iou_threshold))
        except Exception:
            indices = []

        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten().tolist()
            for idx in indices:
                orig_idx = cls_indices[idx]
                final_dets.append({
                    "box": [int(x1[orig_idx]), int(y1[orig_idx]), int(x2[orig_idx]), int(y2[orig_idx])],
                    "class_id": int(class_ids[orig_idx]),
                    "confidence": float(confidences[orig_idx])
                })

    final_dets.sort(key=lambda d: d["confidence"], reverse=True)
    return final_dets


# ==============================================================================
# 4. STREAMLIT USER INTERFACE
# ==============================================================================
st.markdown('<div class="main-title">🐾 Animal Intrusion Detection AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time wildlife identification & automated intrusion warning alerts.</div>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection Controls")
    conf_thresh = st.slider("Confidence Sensitivity", min_value=0.10, max_value=0.95, value=0.25, step=0.05,
                            help="Lower values detect distant animals; higher values require strong certainty.")
    
    target_intruders = st.multiselect(
        "🚨 High-Threat Intruder Animals (Trigger Red Alert):",
        options=["Elephant", "Wild_Boar", "Buffalo", "Rhino", "Zebra", "Deer", "Monkey", "Cow", "Human"],
        default=["Elephant", "Wild_Boar", "Buffalo", "Rhino"]
    )
    
    st.markdown("---")
    st.markdown("💡 **Tip**: On mobile phones, choose *'Take Live Camera Photo'* to snap photos directly from the field.")

# Input options
input_mode = st.radio("Choose How to Input Image:", ["📁 Upload Photo from Device", "📸 Take Photo with Camera"], horizontal=True)

image_input = None
if input_mode == "📁 Upload Photo from Device":
    uploaded = st.file_uploader("Upload an animal photo (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_input = Image.open(uploaded).convert("RGB")
else:
    captured = st.camera_input("Snap a photo of the animal...")
    if captured:
        image_input = Image.open(captured).convert("RGB")

# Run Detection
if image_input is not None:
    session, input_name, output_name = load_onnx_model()
    
    if session is None:
        st.error("⚠️ Model file `best.onnx` not found in this repository. Please make sure `best.onnx` is uploaded.")
    else:
        img_np = np.array(image_input)
        orig_h, orig_w = img_np.shape[:2]

        # 1. Preprocess
        letterboxed, ratio, pad = letterbox(img_np, (640, 640))
        blob = letterboxed.transpose(2, 0, 1)
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # 2. Inference
        t0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: blob})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Postprocess
        detections = decode_yolo_onnx(outputs[0], conf_thresh, 0.45, (orig_h, orig_w), ratio, pad)

        # 4. Annotate
        annotated = img_np.copy()
        has_intrusion = False
        threat_list = []
        species_counts = {}

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cid = det["class_id"]
            conf = det["confidence"]
            cname = DEFAULT_CLASSES[cid] if 0 <= cid < len(DEFAULT_CLASSES) else f"class_{cid}"
            
            species_counts[cname] = species_counts.get(cname, 0) + 1
            is_threat = cname.capitalize() in target_intruders

            if is_threat:
                has_intrusion = True
                threat_list.append(f"{cname.upper()} ({conf * 100:.1f}%)")
                color = (220, 20, 20)  # Red Alert
                box_thickness = 3
            else:
                color = CLASS_COLORS[cid % len(CLASS_COLORS)]
                box_thickness = 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
            label = f"⚠️ {cname.upper()} {conf*100:.1f}%" if is_threat else f"{cname} {conf*100:.1f}%"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), bl = cv2.getTextSize(label, font, 0.55, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - bl - 4)), (min(orig_w, x1 + tw + 6), y1), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - bl - 1), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Display Columns
        col_img, col_info = st.columns([1.3, 1])

        with col_img:
            st.image(annotated, caption="Detection Result", use_container_width=True)

        with col_info:
            if has_intrusion:
                st.error(f"🚨 **INTRUSION ALERT CONFIRMED!**\n\n**Threat Species:** {', '.join(threat_list)}")
            elif len(detections) > 0:
                st.success("✅ **Animals Detected (Safe / Non-Intruder)**")
            else:
                st.info("🔍 **No Animals Detected.** Try lowering the sensitivity slider on the left.")

            st.markdown("### 📊 Detection Summary")
            m1, m2 = st.columns(2)
            m1.metric("Inference Time", f"{latency_ms:.1f} ms")
            m2.metric("Total Animals", len(detections))

            if species_counts:
                st.markdown("**Spotted Species Breakdown:**")
                for sp, count in species_counts.items():
                    st.write(f"- 🐾 **{sp.capitalize()}**: `{count} detected`")

st.markdown("---")
st.caption("Animal Intrusion Detection AI • Hosted 24/7 on Streamlit Cloud • Powered by YOLO11 & ONNX Runtime")
