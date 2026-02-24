"""
app.py  —  SkinAI Streamlit entry point
────────────────────────────────────────
Architectural rules enforced here:
  • session state initialised ONCE via app_controller.init_state()
  • task selected ONCE in sidebar → propagated globally
  • image uploaded ONCE (sidebar widget) → stored in session_state
  • pages are thin view modules imported from ui/
  • no inference logic, no preprocessing, no file I/O in this file
"""

import streamlit as st
from model import init_state, set_task, get_task, set_image, has_image, set_page, get_page

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SkinAI – Dermatoscopic Analysis",
    page_icon="🧬",
    layout="wide",
)

# ──────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f1c2e, #132c4c);
    color: #ffffff;
}
section[data-testid="stSidebar"] {
    background: #0c1a2b;
    border-right: 1px solid #1f3b5c;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
h1, h2, h3 { color: #4ea8ff; }
div[data-testid="metric-container"] {
    background: #162c47;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #1f3b5c;
}
.stButton>button {
    background-color: #1e4fa3; color: white;
    border-radius: 8px; border: none;
}
.stButton>button:hover { background-color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state bootstrap
# ──────────────────────────────────────────────
init_state()

# ──────────────────────────────────────────────
# Sidebar — global controls (defined ONCE)
# ──────────────────────────────────────────────
st.sidebar.markdown("## 🧬 SkinAI")
st.sidebar.markdown("### Clinical ML Explorer")
st.sidebar.markdown("---")

# Navigation
for label in ["Home", "Learn", "Demo", "Live Prediction", "Performance Metrics"]:
    if st.sidebar.button(label, use_container_width=True):
        set_page(label)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Classification Task")

task_choice = st.sidebar.radio(
    "Select task",
    ["Binary (Melanoma vs Others)", "Multiclass (7 Lesion Types)"],
    index=0 if get_task() == "binary" else 1,
    label_visibility="collapsed",
)
set_task("binary" if "Binary" in task_choice else "multiclass")

# ── Single global image uploader ─────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🖼 Upload Image")
uploaded = st.sidebar.file_uploader(
    "Upload a dermatoscopic image",
    type=["jpg", "jpeg", "png"],
    key="global_uploader",
)
if uploaded is not None:
    from PIL import Image
    pil_img = Image.open(uploaded)
    set_image(pil_img, name=uploaded.name)
    st.sidebar.success(f"✅ {uploaded.name}")

if has_image():
    from model import get_image
    st.sidebar.image(get_image(), caption="Current image", use_container_width=True)
else:
    st.sidebar.info("No image loaded yet.")

# ──────────────────────────────────────────────
# Page routing
# ──────────────────────────────────────────────
page = get_page()

if page == "Home":
    from ui.page_home import render
    render()

elif page == "Learn":
    from ui.page_learn import render
    render()

elif page == "Demo":
    from ui.page_demo import render
    render()

elif page == "Live Prediction":
    from ui.page_predict import render
    render()

elif page == "Performance Metrics":
    from ui.page_metrics import render
    render()