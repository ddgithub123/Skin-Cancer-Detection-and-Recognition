"""
ui/page_demo.py  —  Interactive AI Learning Lab
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

from model import has_image, get_image, preprocess_display, is_binary, get_task


def render():
    st.title("🎮 Interactive AI Learning Lab")
    st.write(
        "Pull back the curtain on how deep learning processes dermatoscopic images."
    )

    if not has_image():
        st.info("👈 Upload an image in the sidebar to enable interactive demos.")

    tabs = st.tabs([
        "🔍 Edge Detection",
        "🌈 Color Channels",
        "📐 Resolution",
        "🎬 Pipeline Walkthrough",
        "⚖️ Confusion Matrix" if is_binary() else "🌳 Class Hierarchy",
    ])

    with tabs[0]:
        _edge_detection_tab()
    with tabs[1]:
        _color_channels_tab()
    with tabs[2]:
        _resolution_tab()
    with tabs[3]:
        _pipeline_tab()
    with tabs[4]:
        _binary_confusion_tab() if is_binary() else _multiclass_hierarchy_tab()


# ──────────────────────────────────────────────

def _get_gray():
    if not has_image():
        return None
    return np.array(get_image().convert("L"))


def _get_rgb():
    if not has_image():
        return None
    return preprocess_display(get_image())


# ──────────────────────────────────────────────

def _edge_detection_tab():
    st.subheader("Feature Extraction: Edge & Texture Kernels")
    st.write(
        "Convolutional filters detect low-level features by sliding a small "
        "math window (kernel) over the image."
    )

    gray = _get_gray()
    if gray is None:
        st.info("Upload an image to see edge detection.")
        return

    kernel = st.radio(
        "Select kernel",
        ["Sobel (Edges)", "Laplacian (Sharp Detail)", "Gaussian Blur (Denoise)"],
        horizontal=True,
    )

    if "Sobel" in kernel:
        result = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        caption = "Sobel gradient — highlights boundary regions"
    elif "Laplacian" in kernel:
        result = cv2.Laplacian(gray, cv2.CV_64F)
        caption = "Laplacian — detects rapid intensity changes"
    else:
        result = cv2.GaussianBlur(gray.astype(float), (0, 0), 2.0)
        caption = "Gaussian blur — suppresses high-frequency noise"

    result = np.abs(result)
    result = (result / (result.max() + 1e-5) * 255).astype(np.uint8)

    col1, col2 = st.columns(2)
    col1.image(gray, caption="Grayscale input", use_container_width=True)
    col2.image(result, caption=caption, use_container_width=True)

    st.info(
        "💡 The model learns thousands of such filters automatically during "
        "training — no manual kernel design required."
    )


def _color_channels_tab():
    st.subheader("Color Channels & Biological Significance")
    st.write("Different color channels encode different tissue information:")

    rgb = _get_rgb()
    if rgb is None:
        st.info("Upload an image to analyse color channels.")
        return

    channel_info = {
        "Red":   ("Hemoglobin, blood vessels, vascular lesions",   0, "#FF5252"),
        "Green": ("Best lesion contrast, border detection",        1, "#69F0AE"),
        "Blue":  ("Melanin density, deep pigment networks",        2, "#40C4FF"),
    }

    channel = st.select_slider("Select channel", options=list(channel_info.keys()))
    desc, idx, color = channel_info[channel]

    overlay = np.zeros_like(rgb)
    overlay[:, :, idx] = rgb[:, :, idx]

    col1, col2 = st.columns(2)
    col1.image(rgb, caption="Original RGB", use_container_width=True)
    col2.image(overlay, caption=f"{channel} channel only", use_container_width=True)
    st.markdown(f"**{channel} channel** — {desc}")


def _resolution_tab():
    st.subheader("Resolution & Information Loss")
    st.write(
        "Models resize all images to a fixed size (224×224). "
        "See how lesion detail degrades as resolution decreases:"
    )

    pil = get_image() if has_image() else None
    if pil is None:
        st.info("Upload an image to see resolution scaling.")
        return

    res = st.select_slider("Target resolution", options=[224, 128, 64, 32, 16])
    resized = pil.resize((res, res)).resize((224, 224), Image.NEAREST)
    st.image(np.array(resized), caption=f"{res}×{res} pixels (upscaled for display)")

    if res <= 32:
        st.warning(
            "At this resolution, fine border irregularities are lost — "
            "the model cannot reliably distinguish melanoma from nevus."
        )


def _pipeline_tab():
    st.subheader("AI Reasoning Pipeline — Step by Step")

    steps = [
        ("Raw pixel matrix",     "The image is a 3D array: 224 × 224 × 3 = 150,528 numbers."),
        ("Backbone preprocessing","Pixel values are normalised using the backbone-specific function\n"
                                  "(e.g. EfficientNet uses its own mean/std, not plain /255)."),
        ("Early conv layers",    "Kernels detect edges, gradients, and color blobs (low-level)."),
        ("Middle layers",        "Features combine into shapes, textures, pigment patterns."),
        ("Deep layers",          "High-level concepts: 'blue-white veil', 'atypical network'."),
        ("Global Average Pool",  "Spatial dimensions collapse: (7,7,1280) → (1280,). "
                                 "Translation invariance is achieved."),
        ("Classification head",  "Dense layer maps (1280,) → 1 (binary) or 7 (multiclass) logits."),
        ("Activation",           "Sigmoid → [0,1] probability (binary), "
                                 "or Softmax → probability distribution (multiclass)."),
        ("Grad-CAM",             "Gradients flow back to the last conv layer to reveal "
                                 "which spatial regions drove the prediction."),
    ]

    if st.button("▶ Run walkthrough"):
        prog = st.progress(0)
        step_box = st.empty()
        desc_box = st.empty()
        for i, (title, desc) in enumerate(steps):
            prog.progress((i + 1) / len(steps))
            step_box.markdown(f"### Step {i+1}: {title}")
            desc_box.info(desc)
            time.sleep(2.5)
        step_box.markdown("### ✅ Pipeline complete!")
        desc_box.success("The model has produced its prediction and Grad-CAM explanation.")


def _binary_confusion_tab():
    st.subheader("Confusion Matrix & Threshold Mechanics")
    st.write("Move the threshold slider to see how patients are classified:")

    threshold = st.slider("Decision threshold τ", 0.0, 1.0, 0.5)

    patients = [
        (0.95,"M"),(0.85,"M"),(0.75,"M"),(0.65,"M"),(0.55,"M"),(0.45,"M"),(0.35,"M"),
        (0.05,"B"),(0.15,"B"),(0.25,"B"),(0.35,"B"),(0.45,"B"),(0.55,"B"),(0.65,"B"),
        (0.12,"B"),(0.18,"B"),(0.88,"M"),(0.92,"M"),(0.48,"B"),(0.52,"M"),
    ]

    tp, fp, tn, fn = [], [], [], []
    for p, gt in patients:
        pred = "M" if p >= threshold else "B"
        if gt == "M" and pred == "M": tp.append(p)
        elif gt == "B" and pred == "M": fp.append(p)
        elif gt == "B" and pred == "B": tn.append(p)
        else: fn.append(p)

    c1, c2 = st.columns(2)
    c1.success(f"✅ True Negatives (correct benign): {len(tn)}")
    c2.warning(f"⚠️ False Positives (false alarm): {len(fp)}")
    c3, c4 = st.columns(2)
    c3.error(f"🚨 False Negatives (missed cancer): {len(fn)}")
    c4.success(f"✅ True Positives (caught cancer): {len(tp)}")

    total_mel = len(tp) + len(fn)
    if total_mel > 0:
        sens = len(tp) / total_mel
        st.metric("Sensitivity (Recall)", f"{sens:.0%}")


def _multiclass_hierarchy_tab():
    st.subheader("Skin Lesion Classification Hierarchy")

    st.graphviz_chart("""
    digraph H {
        rankdir=TB;
        node [shape=box, style=filled, fontname="Arial", fontsize=11];

        Root [label="Skin Lesion", fillcolor="#37474f", fontcolor=white];

        M   [label="Malignant", fillcolor="#b71c1c", fontcolor=white];
        B   [label="Benign", fillcolor="#1b5e20", fontcolor=white];

        mel [label="Melanoma (mel)", fillcolor="#e53935", fontcolor=white];
        bcc [label="Basal Cell Carcinoma (bcc)", fillcolor="#e57373", fontcolor=white];
        akiec [label="Actinic Keratosis (akiec)", fillcolor="#ef9a9a", fontcolor=white];

        nv  [label="Melanocytic Nevi (nv)", fillcolor="#43a047", fontcolor=white];
        bkl [label="Seborrheic Keratosis (bkl)", fillcolor="#66bb6a", fontcolor=white];
        df  [label="Dermatofibroma (df)", fillcolor="#a5d6a7", fontcolor=black];
        vasc [label="Vascular Lesions (vasc)", fillcolor="#c8e6c9", fontcolor=black];

        Root -> M; Root -> B;
        M -> mel; M -> bcc; M -> akiec;
        B -> nv; B -> bkl; B -> df; B -> vasc;
    }
    """)

    st.info(
        "The 7-class model treats all 7 types independently via softmax. "
        "A clinician might further group akiec as 'pre-malignant' — "
        "the model doesn't encode this hierarchy explicitly, "
        "but it can be enforced post-hoc via probability grouping."
    )