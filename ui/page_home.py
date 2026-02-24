"""
ui/page_home.py
"""
import streamlit as st
from model import get_config, get_task, is_binary, model_available


def render():
    st.markdown("""
    <div style="background:rgba(30,79,163,0.15);padding:40px;border-radius:20px;
    border:1px solid rgba(78,168,255,0.3);">
    <h1 style="color:#4ea8ff;font-size:42px;">AI-Powered Skin Cancer Diagnosis</h1>
    <p style="font-size:18px;color:#dbeafe;">
    A deep learning system for dermatoscopic image analysis using EfficientNetB0 
    transfer learning and Grad-CAM explainability.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🧠 System Overview")

    cfg = get_config(get_task())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔄 Pipeline
        - Image upload → backbone preprocessing
        - EfficientNetB0 feature extraction
        - Task-specific classification head
        - Grad-CAM explainability (auto layer detection)
        - Threshold-aware binary decision
        """)
    with col2:
        st.markdown(f"""
        ### ⚙ Current Configuration
        - **Task**: `{get_task()}`
        - **Backbone**: `{cfg.backbone}`
        - **Classes**: `{cfg.num_classes}`
        - **Output activation**: `{"Sigmoid" if is_binary() else "Softmax"}`
        - **Model loaded**: `{"✅" if model_available(get_task()) else "⚠️ file not found"}`
        """)

    st.markdown("## 📊 Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "HAM10000")
    c2.metric("Images", "10,015")
    c3.metric("Classes", "7 types")
    c4.metric("Imbalance", "67% Nevus")

    st.markdown("## 🏗 Architecture")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="#1e4fa3", fontcolor="white"];
        Input -> EfficientNetB0 -> GlobalAvgPool -> Dropout -> Dense -> Output;
    }
    """)

    st.markdown("## 🔑 Key Design Decisions")
    st.info("""
    - **No hardcoded layer names** — Grad-CAM auto-detects the last Conv2D
    - **Backbone-specific preprocessing** — EfficientNet ≠ ResNet ≠ plain /255
    - **Single image upload** — stored globally, reused across all pages
    - **Threshold awareness** — binary decision adjustable at inference time
    - **Educational mode** — feature maps, logits, softmax all accessible
    """)

    st.caption(
        "For research and educational purposes only. "
        "Not a substitute for professional medical diagnosis."
    )