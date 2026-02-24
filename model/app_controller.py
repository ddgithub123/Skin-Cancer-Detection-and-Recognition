"""
app_controller.py
─────────────────
Centralised Streamlit session-state management.

Rules:
  1. The image is uploaded ONCE (on any page) and stored in session_state.
  2. The task (binary / multiclass) is selected ONCE in the sidebar and persists.
  3. Changing the task invalidates the cached PredictionResult and Grad-CAM heatmap.
  4. All pages read state from here — no local uploaders, no local task selectors.

Public API:
    init_state()                -> None   (call at top of app.py)
    set_task(task_key)          -> None
    set_image(pil_image, name)  -> None
    get_task()                  -> str    "binary" | "multiclass"
    get_image()                 -> PIL.Image | None
    get_image_name()            -> str | None
    get_prediction()            -> PredictionResult | None
    set_prediction(result)      -> None
    clear_prediction()          -> None
    get_heatmap()               -> np.ndarray | None
    set_heatmap(arr)            -> None
    clear_heatmap()             -> None
    has_image()                 -> bool
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from PIL import Image
import streamlit as st


# ──────────────────────────────────────────────
# Keys
# ──────────────────────────────────────────────

_KEY_TASK        = "ac_task"
_KEY_IMAGE       = "ac_image"          # PIL.Image
_KEY_IMAGE_NAME  = "ac_image_name"
_KEY_PREDICTION  = "ac_prediction"     # PredictionResult | None
_KEY_HEATMAP     = "ac_heatmap"        # np.ndarray | None
_KEY_PAGE        = "ac_page"


# ──────────────────────────────────────────────
# Initialisation
# ──────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        _KEY_TASK:       "binary",
        _KEY_IMAGE:      None,
        _KEY_IMAGE_NAME: None,
        _KEY_PREDICTION: None,
        _KEY_HEATMAP:    None,
        _KEY_PAGE:       "Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ──────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────

def set_task(task_key: str) -> None:
    """task_key must be 'binary' or 'multiclass'."""
    if task_key not in ("binary", "multiclass"):
        raise ValueError(f"Invalid task: {task_key!r}")
    if st.session_state.get(_KEY_TASK) != task_key:
        st.session_state[_KEY_TASK] = task_key
        # Invalidate cached inference results
        clear_prediction()
        clear_heatmap()


def get_task() -> str:
    return st.session_state.get(_KEY_TASK, "binary")


def is_binary() -> bool:
    return get_task() == "binary"


# ──────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────

def set_image(pil_image: Image.Image, name: str = "uploaded") -> None:
    """Store a PIL image. Clears prediction/heatmap because image changed."""
    st.session_state[_KEY_IMAGE]      = pil_image.convert("RGB")
    st.session_state[_KEY_IMAGE_NAME] = name
    clear_prediction()
    clear_heatmap()


def get_image() -> Optional[Image.Image]:
    return st.session_state.get(_KEY_IMAGE)


def get_image_name() -> Optional[str]:
    return st.session_state.get(_KEY_IMAGE_NAME)


def has_image() -> bool:
    return st.session_state.get(_KEY_IMAGE) is not None


# ──────────────────────────────────────────────
# Prediction cache
# ──────────────────────────────────────────────

def set_prediction(result) -> None:
    st.session_state[_KEY_PREDICTION] = result


def get_prediction():
    return st.session_state.get(_KEY_PREDICTION)


def clear_prediction() -> None:
    st.session_state[_KEY_PREDICTION] = None


# ──────────────────────────────────────────────
# Heatmap cache
# ──────────────────────────────────────────────

def set_heatmap(arr: np.ndarray) -> None:
    st.session_state[_KEY_HEATMAP] = arr


def get_heatmap() -> Optional[np.ndarray]:
    return st.session_state.get(_KEY_HEATMAP)


def clear_heatmap() -> None:
    st.session_state[_KEY_HEATMAP] = None


# ──────────────────────────────────────────────
# Page routing
# ──────────────────────────────────────────────

def set_page(name: str) -> None:
    st.session_state[_KEY_PAGE] = name


def get_page() -> str:
    return st.session_state.get(_KEY_PAGE, "Home")


# ──────────────────────────────────────────────
# Convenience: run inference + gradcam if needed
# ──────────────────────────────────────────────

def ensure_prediction() -> None:
    """
    Run prediction and Grad-CAM if the cached result is stale.
    No-op if cache is current.
    """
    if get_prediction() is not None:
        return
    if not has_image():
        return

    from model.predictor import predict
    from model.gradcam import compute_gradcam
    from model.preprocessing import preprocess
    from model.model_manager import get_model, get_config, model_available

    task  = get_task()
    image = get_image()

    result = predict(task, image)
    set_prediction(result)

    print("Model available:", model_available(task))

    if model_available(task):
        cfg   = get_config(task)
        model = get_model(task)
        img_batch = preprocess(image, cfg.backbone)
        try:
            heatmap = compute_gradcam(
                model,
                img_batch,
                class_index=result.class_index
            )
            set_heatmap(heatmap)
        except Exception as e:
            import traceback
            traceback.print_exc()
            clear_heatmap()