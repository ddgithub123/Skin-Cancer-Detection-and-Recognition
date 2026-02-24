# Model package initialization
from .inference import predict_binary, predict_multiclass, generate_gradcam, overlay_gradcam, HAS_TF, MULTICLASS_LABELS


"""
model package public API
"""
from .model_manager  import get_model, get_config, has_tf, model_available, MODEL_REGISTRY
from .preprocessing  import preprocess, preprocess_display, load_image
from .predictor      import predict, get_feature_maps, get_logits_and_softmax, PredictionResult
from .gradcam        import compute_gradcam, overlay_heatmap, find_last_conv_layer
from .app_controller import (
    init_state, set_task, get_task, is_binary,
    set_image, get_image, get_image_name, has_image,
    set_prediction, get_prediction, clear_prediction,
    set_heatmap, get_heatmap, clear_heatmap,
    set_page, get_page, ensure_prediction,
)

__all__ = [
    # model_manager
    "get_model", "get_config", "has_tf", "model_available", "MODEL_REGISTRY",
    # preprocessing
    "preprocess", "preprocess_display", "load_image",
    # predictor
    "predict", "get_feature_maps", "get_logits_and_softmax", "PredictionResult",
    # gradcam
    "compute_gradcam", "overlay_heatmap", "find_last_conv_layer",
    # app_controller
    "init_state", "set_task", "get_task", "is_binary",
    "set_image", "get_image", "get_image_name", "has_image",
    "set_prediction", "get_prediction", "clear_prediction",
    "set_heatmap", "get_heatmap", "clear_heatmap",
    "set_page", "get_page", "ensure_prediction",
]