"""
ui/page_learn.py
────────────────
Educational walkthrough of the ML pipeline.

Sections:
  1. Data distribution & class imbalance
  2. Preprocessing pipeline (backbone-specific)
  3. Transfer learning — frozen vs trainable layers
  4. Architecture graph (accurate per selected task)
  5. Decision logic (sigmoid vs softmax with math)
  6. Grad-CAM explainability
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from model import (
    get_task, is_binary, has_image, get_image,
    ensure_prediction, get_heatmap, get_prediction,
    overlay_heatmap, preprocess_display,
    get_config, model_available,
)


def render():
    st.title("🧠 ML Pipeline Walkthrough")
    st.write(
        "A structured tour of how dermatoscopic image classification works — "
        "from raw pixels to clinical prediction."
    )

    tabs = st.tabs([
        "1️⃣ Data",
        "2️⃣ Preprocessing",
        "3️⃣ Transfer Learning",
        "4️⃣ Architecture",
        "5️⃣ Decision Logic",
        "6️⃣ Grad-CAM",
    ])

    with tabs[0]:
        _data_tab()
    with tabs[1]:
        _preprocessing_tab()
    with tabs[2]:
        _transfer_learning_tab()
    with tabs[3]:
        _architecture_tab()
    with tabs[4]:
        _decision_logic_tab()
    with tabs[5]:
        _gradcam_tab()


# ──────────────────────────────────────────────
# Tab 1 — Data
# ──────────────────────────────────────────────

from model import get_task

def _data_tab():
    st.header("Dataset: HAM10000")

    task = get_task()

    # Full 7-class counts
    full_counts = {
        "nv": 6705,
        "mel": 1113,
        "bkl": 1099,
        "bcc": 514,
        "akiec": 327,
        "vasc": 142,
        "df": 115,
    }

    if task == "binary":

        cancer = full_counts["mel"] + full_counts["bcc"] + full_counts["akiec"]
        non_cancer = (
            full_counts["nv"]
            + full_counts["bkl"]
            + full_counts["df"]
            + full_counts["vasc"]
        )

        class_counts = {
            "Cancer": cancer,
            "Non-Cancer": non_cancer,
        }

    else:
        class_counts = {
            "nv (Nevus)": 6705,
            "mel (Melanoma)": 1113,
            "bkl (Seborrheic keratosis)": 1099,
            "bcc (Basal cell carcinoma)": 514,
            "akiec (Actinic keratosis)": 327,
            "vasc (Vascular lesions)": 142,
            "df (Dermatofibroma)": 115,
        }

    df = pd.DataFrame({"Images": class_counts})

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))

        wedges, texts, autotexts = ax.pie(
            df["Images"],
            autopct="%1.0f%%",
            startangle=90,
            pctdistance=0.75
        )

        ax.set_title("Class Distribution")

        # Move labels to legend outside
        ax.legend(
            wedges,
            df.index,
            title="Classes",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )

        st.pyplot(fig)
        plt.close()

    # Task-specific explanation
    if task == "binary":
        st.warning(
            "⚠️ Binary setup merges melanoma, BCC and AKIEC into 'Cancer'. "
            "This simplifies the task but still suffers from imbalance."
        )
    else:
        st.warning(
            "⚠️ Severe class imbalance: Nevus makes up 67% of the dataset."
        )
# ──────────────────────────────────────────────
# Tab 2 — Preprocessing
# ──────────────────────────────────────────────

def _preprocessing_tab():
    st.header("Backbone-Specific Preprocessing")

    st.write("""
    Different ImageNet-pretrained backbones were trained with different 
    normalisation schemes. Using the wrong one degrades performance significantly.
    """)

    data = {
        "Backbone": ["EfficientNetB0-B7", "ResNet50 / ResNet101", "VGG16 / VGG19", "InceptionV3"],
        "Normalisation": [
            "Scale to [0, 255] then apply per-channel shift (tf built-in)",
            "Zero-centre per ImageNet channel means (BGR order)",
            "Zero-centre per ImageNet channel means (BGR order)",
            "Scale to [−1, 1]",
        ],
        "Input range": ["[−1, 1] approx", "[~−123, ~131]", "[~−103, ~152]", "[−1, 1]"],
    }
    st.table(pd.DataFrame(data))

    st.error(
        "🐛 **Bug in original code**: `preprocess.py` used plain `/ 255.0` "
        "for both models. EfficientNet requires "
        "`tf.keras.applications.efficientnet.preprocess_input`. "
        "This was silently degrading accuracy."
    )

    if has_image():
        pil_img = get_image()
        img = preprocess_display(pil_img)

        st.subheader("Augmentation preview")
        cols = st.columns(4)
        augmentations = [
            ("Original", img),
            ("Horizontal flip", img[:, ::-1, :]),
            ("Vertical flip", img[::-1, :, :]),
            ("Brightness +20%", np.clip((img.astype(float) * 1.2), 0, 255).astype(np.uint8)),
        ]
        for (label, aug), col in zip(augmentations, cols):
            with col:
                st.image(aug, caption=label, use_container_width=True)
    else:
        st.info("Upload an image in the sidebar to see augmentation previews.")






# ──────────────────────────────────────────────
# Tab 3 — Transfer Learning
# ──────────────────────────────────────────────

def _transfer_learning_tab():
    st.header("Transfer Learning: Frozen vs Trainable Layers")

    st.write("""
    Instead of training from scratch on 10,015 images (far too few for a deep CNN),
    we start with a network pre-trained on **ImageNet** — 1.2 million images 
    across 1,000 categories.
    """)

    st.graphviz_chart("""
    digraph TL {
        rankdir=LR;
        node [shape=box, style=filled, fontname="Arial", fontsize=11];
        
        subgraph cluster_frozen {
            label="🔒 Frozen (ImageNet weights preserved)";
            style=filled; color="#1a3a5c";
            fontcolor=white;
            A [label="Conv Block 1\\n(edges)", fillcolor="#0d47a1", fontcolor=white];
            B [label="Conv Block 2\\n(shapes)", fillcolor="#1565c0", fontcolor=white];
            C [label="Conv Block 3\\n(patterns)", fillcolor="#1976d2", fontcolor=white];
        }
        
        subgraph cluster_fine {
            label="🔓 Fine-tuned (skin lesion domain)";
            style=filled; color="#1a3a1a";
            fontcolor=white;
            D [label="Conv Block 4\\n(complex)", fillcolor="#1b5e20", fontcolor=white];
            E [label="Conv Block 5\\n(abstract)", fillcolor="#2e7d32", fontcolor=white];
        }
        
        subgraph cluster_head {
            label="🆕 New Classification Head";
            style=filled; color="#5a1a0a";
            fontcolor=white;
            F [label="Global Avg Pool", fillcolor="#bf360c", fontcolor=white];
            G [label="Dropout 0.3", fillcolor="#d84315", fontcolor=white];
            H [label="Dense → Output", fillcolor="#e64a19", fontcolor=white];
        }
        
        A -> B -> C -> D -> E -> F -> G -> H;
    }
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Why freeze early layers?**
        
        Early conv layers detect universal features — edges, 
        corners, gradients — that apply equally to skin and 
        to cats or furniture. Re-learning them wastes compute 
        and risks overfitting on small datasets.
        """)
    with col2:
        st.info("""
        **Why fine-tune later layers?**
        
        Later layers encode domain-specific abstractions 
        (pigment networks, telangiectasia). These need 
        adaptation to the dermatology domain. Fine-tuning 
        with a small learning rate adjusts them without 
        destroying the general representations.
        """)

    st.code("""
# Example: freeze backbone, train head only first
base = tf.keras.applications.EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False          # 🔒 freeze all backbone weights

# Add classification head
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # binary
model = tf.keras.Model(base.input, output)

# Phase 2: unfreeze top N layers for fine-tuning
base.trainable = True
for layer in base.layers[:-20]:   # keep bottom frozen
    layer.trainable = False
""", language="python")


# ──────────────────────────────────────────────
# Tab 4 — Architecture
# ──────────────────────────────────────────────

def _architecture_tab():
    task = get_task()
    cfg = get_config(task)

    st.header(f"Model Architecture — {task.title()} Mode")
    st.write(f"**Backbone**: `{cfg.backbone}` | **Output classes**: `{cfg.num_classes}`")

    activation = "Sigmoid (1 neuron)" if is_binary() else f"Softmax ({cfg.num_classes} neurons)"

    st.graphviz_chart(f"""
    digraph G {{
        rankdir=LR;
        node [shape=box, style=filled, fontname="Arial", fontsize=11];

        Input [label="Input\\n224×224×3", fillcolor="#0d47a1", fontcolor=white];
        BN [label="{cfg.backbone.upper()}\\n(ImageNet pretrained)", fillcolor="#1565c0", fontcolor=white];
        GAP [label="GlobalAvgPool2D\\n→ (1280,)", fillcolor="#1976d2", fontcolor=white];
        Drop [label="Dropout 0.3", fillcolor="#1e88e5", fontcolor=white];
        Dense [label="Dense\\n(activation)", fillcolor="#2196f3", fontcolor=white];
        Output [label="{activation}", fillcolor="#42a5f5", fontcolor=white];

        Input -> BN -> GAP -> Drop -> Dense -> Output;
    }}
    """)

    st.markdown("---")
    st.subheader("Parameter count (approximate)")
    params = {
        "efficientnetb0": ("4.0M backbone", "~0.5K head", "~4M total"),
        "resnet50":        ("23.5M backbone", "~0.5K head", "~23.5M total"),
        "efficientnetb3":  ("10.8M backbone", "~0.5K head", "~10.8M total"),
    }
    bb = cfg.backbone.lower()
    p = params.get(bb, ("Unknown", "Unknown", "Unknown"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Backbone params", p[0])
    col2.metric("Head params", p[1])
    col3.metric("Total", p[2])


# ──────────────────────────────────────────────
# Tab 5 — Decision Logic
# ──────────────────────────────────────────────

def _decision_logic_tab():
    st.header("Decision Logic: Logits → Prediction")

    if is_binary():
        st.subheader("Binary Classification — Sigmoid")
        st.latex(r"p = \sigma(z) = \frac{1}{1 + e^{-z}}")
        st.write("The model outputs a single score *z* (logit). Sigmoid squashes it to [0,1].")

        z_vals = np.linspace(-6, 6, 200)
        sig_vals = 1 / (1 + np.exp(-z_vals))
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(z_vals, sig_vals, color="#4ea8ff", linewidth=2)
        ax.axhline(0.5, color="red", linestyle="--", label="τ = 0.5 (default threshold)")
        ax.axvline(0,   color="gray", linestyle=":", alpha=0.5)
        ax.fill_between(z_vals, sig_vals, 0.5,
                         where=sig_vals >= 0.5, alpha=0.15, color="red", label="Melanoma region")
        ax.fill_between(z_vals, sig_vals, 0.5,
                         where=sig_vals < 0.5, alpha=0.15, color="green", label="Non-melanoma region")
        ax.set_xlabel("Logit z"); ax.set_ylabel("Probability p")
        ax.set_title("Sigmoid Decision Function")
        ax.legend(fontsize=9)
        st.pyplot(fig)
        plt.close()

    else:
        st.subheader("Multiclass Classification — Softmax")
        st.latex(r"p_i = \frac{e^{z_i}}{\sum_{j=1}^{7} e^{z_j}}")

        st.write("Adjust the logits below to see how softmax responds:")
        demo_logits = []
        labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        col_l, col_r = st.columns([1, 2])
        with col_l:
            for lb in labels:
                v = st.slider(lb, -4.0, 4.0, 0.0, step=0.1, key=f"logit_{lb}")
                demo_logits.append(v)

        z = np.array(demo_logits)
        e_z = np.exp(z - z.max())   # numerically stable
        softmax = e_z / e_z.sum()

        with col_r:
            import pandas as pd
            df = pd.DataFrame({"Logit": z, "Softmax prob": softmax}, index=labels)
            st.bar_chart(df[["Softmax prob"]])
            st.write(f"**Predicted class**: {labels[np.argmax(softmax)]} ({softmax.max():.1%})")


# ──────────────────────────────────────────────
# Tab 6 — Grad-CAM
# ──────────────────────────────────────────────

def _gradcam_tab():
    st.header("Grad-CAM Explainability")

    st.write("""
    Gradient-weighted Class Activation Mapping (**Grad-CAM**) answers:
    *"Which pixels drove the model toward this prediction?"*
    """)

    st.latex(r"L^c_{Grad-CAM} = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right)")
    st.caption(r"α_k = global-average-pooled gradient of class score c w.r.t. feature map A^k")

    st.write("""
    Steps:
    1. Forward pass → record feature maps of the last conv layer
    2. Backward pass → compute gradients of the target class score w.r.t. those maps
    3. Pool gradients spatially (average over H×W) to get importance weights α
    4. Weighted sum of feature maps, followed by ReLU
    5. Resize to input resolution and overlay on the original image
    """)

    if not has_image():
        st.info("Upload an image in the sidebar to generate a Grad-CAM.")
        return

    ensure_prediction()
    heatmap = get_heatmap()
    pil_img = get_image()
    img_arr = preprocess_display(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_arr, caption="Original", use_container_width=True)
    with col2:
        if heatmap is not None:
            overlay = overlay_heatmap(img_arr, heatmap, alpha=0.45)
            st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.info("Grad-CAM requires the model to be loaded.")

    st.info("""
    **Architecture note**: This implementation dynamically detects the last 
    Conv2D / DepthwiseConv2D layer — no layer names are hardcoded. 
    It works for EfficientNet, ResNet, MobileNet, and any other backbone.
    """)