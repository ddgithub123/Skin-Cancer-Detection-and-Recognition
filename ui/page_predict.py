"""
ui/page_predict.py
──────────────────
Live Prediction page.

Educational elements added:
  • Logits → Softmax visualisation for multiclass
  • Binary threshold slider that actually re-evaluates the prediction
  • Feature map grid (early conv layer outputs)
  • Grad-CAM overlay
  • ABCDE clinical checklist
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from model import (
    has_image, get_image, get_task, is_binary,
    ensure_prediction, get_prediction, get_heatmap,
    get_feature_maps, get_logits_and_softmax,
    overlay_heatmap, preprocess_display,
    model_available,
)


def render():
    task = get_task()
    st.title(f"🧪 Live {'Binary' if is_binary() else 'Multiclass'} Prediction")

    if not has_image():
        st.info("👈 Upload an image in the sidebar to begin.")
        return

    # ── Run inference (cached) ─────────────────
    with st.spinner("Running analysis…"):
        ensure_prediction()

    result = get_prediction()
    heatmap = get_heatmap()

    if result is None:
        st.error("Prediction failed. Check logs.")
        return

    if result.is_demo:
        st.warning(
            "⚠️ **Demo mode** — model file not found. "
            "Results are illustrative only."
        )

    pil_img = get_image()
    img_arr = preprocess_display(pil_img)   # uint8 RGB (224,224,3)

    # ─────────────────────────────────────────────
    # Layout: original | grad-cam | result panel
    # ─────────────────────────────────────────────
    col_img, col_cam, col_res = st.columns([1, 1, 1])

    with col_img:
        st.subheader("Input Image")
        st.image(img_arr, use_container_width=True)

    with col_cam:
        st.subheader("Grad-CAM Attention")
        if heatmap is not None:
            overlay = overlay_heatmap(img_arr, heatmap, alpha=0.45)
            st.image(overlay, use_container_width=True)
            st.caption("Red = high model attention | Blue = low attention")
        else:
            st.info("Grad-CAM unavailable (model not loaded).")

    with col_res:
        st.subheader("Prediction")
        if is_binary():
            _binary_result_panel(result)
        else:
            _multiclass_result_panel(result)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # Educational tabs
    # ─────────────────────────────────────────────
    tabs = st.tabs([
        "🔬 Feature Maps",
        "📊 Logits → Softmax",
        "⚖️ Threshold Mechanics",
        "🕵️ ABCDE Checklist",
    ])

    with tabs[0]:
        _feature_maps_tab(task, pil_img)

    with tabs[1]:
        _logits_tab(task, pil_img)

    with tabs[2]:
        _threshold_tab(result)

    with tabs[3]:
        _abcde_tab(pil_img)


# ──────────────────────────────────────────────
# Result panels
# ──────────────────────────────────────────────

def _binary_result_panel(result):
    prob = result.probability or 0.0
    label = result.predicted_class
    color = "🔴" if label == "Melanoma" else "🟢"
    st.metric("Verdict", f"{color} {label}")
    st.metric("Melanoma probability", f"{prob:.1%}")
    st.progress(prob)


def _multiclass_result_panel(result):
    st.metric("Top Prediction", result.predicted_class)
    if result.probabilities:
        import pandas as pd
        df = (
            pd.DataFrame.from_dict(result.probabilities, orient="index", columns=["Probability"])
            .sort_values("Probability", ascending=False)
        )
        st.bar_chart(df)


# ──────────────────────────────────────────────
# Educational tabs
# ──────────────────────────────────────────────

def _feature_maps_tab(task: str, pil_img: Image.Image):
    st.subheader("Early Convolutional Feature Maps")
    st.write(
        "The very first convolutional layers detect low-level patterns: "
        "edges, color gradients, and textures. Each square below is one "
        "filter's activation map — brighter = stronger response."
    )

    if not model_available(task):
        st.info("Load the model to see real feature maps.")
        _show_placeholder_feature_maps()
        return

    with st.spinner("Extracting feature maps…"):
        maps = get_feature_maps(task, pil_img, layer_index=2)

    if not maps:
        st.warning("Could not extract feature maps from this model.")
        return

    cols = st.columns(4)
    for i, fm in enumerate(maps[:16]):
        with cols[i % 4]:
            fm_u8 = np.uint8(255 * fm)
            fm_rgb = cv2.applyColorMap(fm_u8, cv2.COLORMAP_VIRIDIS)
            st.image(cv2.cvtColor(fm_rgb, cv2.COLOR_BGR2RGB),
                     caption=f"Filter {i+1}", use_container_width=True)

    st.info(
        "💡 **Transfer Learning insight**: These filters were originally "
        "learned on ImageNet (cats, cars, furniture). The model reuses them "
        "because edges and textures are universal — then fine-tunes its deeper "
        "layers specifically for skin lesion patterns."
    )


def _show_placeholder_feature_maps():
    """Synthetic placeholder grids when model is absent."""
    cols = st.columns(4)
    rng = np.random.default_rng(42)
    for i in range(8):
        fake = rng.random((28, 28)).astype(np.float32)
        fake_u8 = np.uint8(255 * fake)
        fake_rgb = cv2.applyColorMap(fake_u8, cv2.COLORMAP_VIRIDIS)
        with cols[i % 4]:
            st.image(cv2.cvtColor(fake_rgb, cv2.COLOR_BGR2RGB),
                     caption=f"Filter {i+1} (demo)", use_container_width=True)


def _logits_tab(task: str, pil_img: Image.Image):
    st.subheader("Logits → Softmax → Probability")

    st.write("""
    The model's final Dense layer outputs raw scores called **logits** — 
    unbounded real numbers. Softmax converts them to a probability 
    distribution that sums to 1.
    """)

    st.latex(r"\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}")

    if task == "binary":
        st.write("""
        In **binary classification**, the model uses a single sigmoid neuron instead:
        """)
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        st.info(
            "A logit of 0 → 50% probability | logit of +2 → 88% | logit of −2 → 12%"
        )
        return

    if not model_available(task):
        st.info("Load the multiclass model to see real logits.")
        _demo_logits_bar()
        return

    with st.spinner("Computing logits…"):
        data = get_logits_and_softmax(task, pil_img)

    if data is None:
        st.warning("Logits not available.")
        return

    import pandas as pd
    labels  = data["labels"]
    logits  = data["logits"]
    softmax = data["softmax"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw logits** (before softmax)")
        df_logits = pd.DataFrame({"Logit": logits}, index=labels)
        st.bar_chart(df_logits)
    with col2:
        st.markdown("**Softmax probabilities**")
        df_soft = pd.DataFrame({"Probability": softmax}, index=labels)
        st.bar_chart(df_soft)

    st.success(
        "Notice how softmax amplifies the largest logit and suppresses others. "
        "A difference of ~2 logit units can dominate the output."
    )


def _demo_logits_bar():
    import pandas as pd
    demo_logits  = np.array([-1.2, 0.3, -0.5, -2.1, 1.4, 2.8, -1.7])
    import tensorflow as tf
    demo_softmax = tf.nn.softmax(demo_logits).numpy()
    labels = ["akiec","bcc","bkl","df","mel","nv","vasc"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Demo logits**")
        st.bar_chart(pd.DataFrame({"Logit": demo_logits}, index=labels))
    with col2:
        st.markdown("**Demo softmax**")
        st.bar_chart(pd.DataFrame({"Probability": demo_softmax}, index=labels))


def _threshold_tab(result):
    st.subheader("Binary Decision Threshold Mechanics")

    st.write("""
    A binary model outputs a single probability *p* for Melanoma.  
    The **threshold** τ determines the clinical decision:
    """)
    st.latex(r"\hat{y} = \begin{cases} \text{Melanoma} & p \geq \tau \\ \text{Non-Melanoma} & p < \tau \end{cases}")

    if not is_binary():
        st.info("Switch to Binary Classification mode to use this control.")
        return

    prob = result.probability if result.probability is not None else 0.18
    threshold = st.slider(
        "Decision threshold τ",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    )

    decision = "🔴 Melanoma" if prob >= threshold else "🟢 Non-Melanoma"

    col1, col2, col3 = st.columns(3)
    col1.metric("Model output p", f"{prob:.3f}")
    col2.metric("Threshold τ", f"{threshold:.2f}")
    col3.metric("Decision", decision)

    st.markdown("---")
    sens = (1 - threshold) * 100
    spec = threshold * 100
    st.progress(sens / 100, text=f"Sensitivity (catch rate): {sens:.0f}%")
    st.progress(spec / 100, text=f"Specificity (precision):  {spec:.0f}%")

    if threshold < 0.3:
        st.error("⚠️ Very low threshold: catches nearly all melanoma but generates many false alarms.")
    elif threshold > 0.7:
        st.warning("⚠️ High threshold: fewer false alarms, but may miss early-stage melanoma.")
    else:
        st.success("⚖️ Balanced threshold — typical for clinical screening.")

    st.info(
        "**Clinical note**: In cancer screening, sensitivity (not missing real cancer) "
        "is usually prioritised over specificity. A threshold of 0.3–0.4 is common "
        "for melanoma to ensure missed cases are minimised."
    )


def _abcde_tab(pil_img: Image.Image):
    st.subheader("🕵️ ABCDE Clinical Checklist")
    st.write(
        "Dermatologists use the **ABCDE criteria** to evaluate suspicious lesions. "
        "Mark what you observe in the uploaded image:"
    )

    st.image(pil_img, width=220)

    signs = {
        "A — Asymmetry": "One half doesn't match the other.",
        "B — Border": "Ragged, notched, or blurred edges.",
        "C — Color": "Multiple shades of brown, black, red, or white.",
        "D — Diameter": "Larger than 6 mm (pencil-eraser size).",
        "E — Evolving": "Changing in size, shape, or color over time.",
    }

    count = 0
    for label, desc in signs.items():
        checked = st.checkbox(f"**{label}** — {desc}")
        if checked:
            count += 1

    if count == 0:
        st.info("No signs marked yet.")
    elif count <= 1:
        st.success(f"✅ {count} sign found — likely benign. Monitor periodically.")
    elif count <= 3:
        st.warning(f"⚠️ {count} signs found — consider dermatologist review.")
    else:
        st.error(f"🚨 {count} signs found — urgent dermatologist referral recommended.")

    st.caption(
        "This checklist is for educational purposes only. "
        "Always consult a qualified dermatologist for diagnosis."
    )