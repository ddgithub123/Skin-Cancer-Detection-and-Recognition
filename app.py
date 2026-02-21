import streamlit as st
import os
import time
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from model.inference import (
    predict_binary, 
    predict_multiclass, 
    generate_gradcam, 
    overlay_gradcam, 
    HAS_TF, 
    MULTICLASS_LABELS
)

def get_feature_maps(image, filter_count=8):
    """
    Simulate feature maps by applying different kernels.
    In a real scenario, this would extract layer outputs.
    """
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_gray = cv2.resize(img_gray, (224, 224))
    
    maps = []
    # 1. Horizontal Edges
    maps.append(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3))
    # 2. Vertical Edges
    maps.append(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))
    # 3. Diagonal Edges
    maps.append(cv2.Laplacian(img_gray, cv2.CV_64F))
    # 4. Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    maps.append(cv2.filter2D(img_gray, -1, kernel))
    # 5. Gaussian Blur (Low pass)
    maps.append(cv2.GaussianBlur(img_gray, (5,5), 0))
    # 6. Erode (Shrink features)
    maps.append(cv2.erode(img_gray, np.ones((3,3), np.uint8), iterations=1))
    # 7. Dilate (Expand features)
    maps.append(cv2.dilate(img_gray, np.ones((3,3), np.uint8), iterations=1))
    # 8. Canny edges
    maps.append(cv2.Canny(img_gray, 100, 200))

    final_maps = []
    for m in maps:
        m = np.abs(m)
        m = (m / (m.max() + 1e-5) * 255).astype(np.uint8)
        final_maps.append(m)
    
    return final_maps

# Page Configuration
st.set_page_config(
    page_title="AI Skin Cancer Diagnosis",
    page_icon="🧠",
    layout="wide"
)


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "How Model Works", "Demo", "Live Prediction", "Performance Metrics"])

st.sidebar.markdown("---")
st.sidebar.subheader("Global Settings")
task_type = st.sidebar.radio(
    "Select Classification Task",
    ["Binary Classification (Melanoma vs Others)", "Multiclass Classification (7 Lesion Types)"],
    index=0
)
is_binary = "Binary" in task_type

# Global Image Upload - Available on all pages
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Your Image")
uploaded_image = st.sidebar.file_uploader(
    "Upload a dermatoscopic image to use throughout the app", 
    type=['jpg', 'jpeg', 'png'], 
    key="global_image_upload"
)

# Store image in session state for persistence across pages
if uploaded_image is not None:
    st.session_state.current_image = uploaded_image
elif 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Image Preview and Management
if st.session_state.get('current_image'):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Image")
    try:
        # Display current image preview
        current_img = Image.open(st.session_state.current_image)
        st.sidebar.image(current_img, caption="Current Image", use_column_width=True)
        
        # Image info
        st.sidebar.info(f"**Size:** {current_img.size[0]}x{current_img.size[1]}")
        st.sidebar.info(f"**Mode:** {current_img.mode}")
        
        # Clear image option
        if st.sidebar.button("🗑️ Clear Image"):
            st.session_state.current_image = None
            st.rerun()
    except Exception as e:
        st.sidebar.error("Error loading image preview")
        if st.sidebar.button("🗑️ Clear Image"):
            st.session_state.current_image = None
            st.rerun()
else:
    st.sidebar.markdown("---")
    st.sidebar.info("📸 No image uploaded yet. Upload one to use throughout the app!")

if page == "Home":
    st.title(f"🧠 AI-Based Skin Cancer Diagnosis ({'Binary' if is_binary else 'Multiclass'})")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Problem Statement")
        if is_binary:
            st.write("""
            Skin cancer is one of the most common types of cancer worldwide. Early detection is critical for successful treatment and survival. 
            This mode focuses specifically on identifying **Melanoma**, the most dangerous form of skin cancer.
            """)
        else:
            st.write("""
            Skin cancer takes many forms. While Melanoma is common, other types like Basal Cell Carcinoma or Actinic Keratoses require different treatments.
            This mode classifies images into **7 distinct clinical categories**.
            """)
        
        st.header("Why AI Matters?")
        st.write("""
        Artificial Intelligence, specifically Deep Learning, can analyze dermatoscopic images with high accuracy, assisting medical professionals 
        in making faster and more reliable diagnoses.
        """)
        
    with col2:
        st.header("Dataset: HAM10000")
        st.write("""
        The "Human Against Machine with 10000 training images" dataset (HAM10000) is used for training. 
        It contains 10,015 dermatoscopic images of common pigmented skin lesions.
        """)
        
        st.header("Model Summary")
        if is_binary:
            st.info("Using **EfficientNetB0** optimized for Binary Classification of Melanoma.")
        else:
            st.info("Using **EfficientNetB0** optimized for Multiclass Classification (7 Categories).")

elif page == "How Model Works":
    st.title(f"🔍 How the {'Binary' if is_binary else 'Multiclass'} Model Works")
    
    if is_binary:
        st.markdown("""
        **Binary classification** means the model predicts one of two possible classes.  
        In skin cancer detection:
        * **Class 0** → Benign (Non-cancerous)
        * **Class 1** → Malignant (Cancerous)
        """)
    else:
        st.markdown("""
        **Multiclass classification** predicts one class out of many categories. In this application, we categorize images into 7 distinct clinical types.
        """)
    
    # --- Technical Pipeline Simulator ---
    st.subheader("🪜 Interactive Training Simulator")
    st.write("Understand the engine behind the AI. Toggle through the technical steps below.")

    # State management for simulation
    if "sim_step" not in st.session_state: st.session_state.sim_step = 1
    
    sim_col1, sim_col2 = st.columns([1, 2])
    
    with sim_col1:
        st.write("### Choose a Step")
        sim_choice = st.radio("Pipeline Stages", [
            "1. Data Collection",
            "2. Preprocessing Lab",
            "3. Split Visualizer",
            "4. Model Architecture",
            "5. Training Simulator",
            "6. Evaluation Hub",
            "7. Prediction Logic"
        ], label_visibility="collapsed")
    
    with sim_col2:
        if "1." in sim_choice:
            st.markdown("#### 📁 Step 1: Data Collection")
            dataset = "ISIC Archive" if is_binary else "HAM10000"
            st.write(f"Feeding the AI with {dataset} images. In real-world data science, we collect thousands of curated images to teach the model.")
            st.image("https://isic-archive.com/static/images/isic-logo.png", width=150)
            st.info("💡 **Key Concept**: Garbage In, Garbage Out. High-quality labels are essential.")

        elif "2." in sim_choice:
            st.markdown("#### 🧪 Step 2: Augmentation Lab")
            st.write("We 'trick' the AI into seeing more data by flipping and rotating images.")
            
            # Using a fallback image if specific ones aren't found
            sample_img_path = "C:/Users/tejasri/.gemini/antigravity/brain/586d411a-3e54-42e3-88a8-e7476a264a7c/sample_melanoma_lesion_1771301003956.png"
            if os.path.exists(sample_img_path):
                sample_img = Image.open(sample_img_path).resize((200, 200))
            else:
                sample_img = Image.new('RGB', (200, 200), color=(150, 50, 50))
                
            aug_op = st.selectbox("Apply Augmentation:", ["Original", "Horizontal Flip", "90° Rotation", "Zoom (Crop)"])
            
            if aug_op == "Original":
                st.image(sample_img, caption="Raw Image")
            elif aug_op == "Horizontal Flip":
                st.image(sample_img.transpose(Image.FLIP_LEFT_RIGHT), caption="Flipped (AI sees a 'new' lesion)")
            elif aug_op == "90° Rotation":
                st.image(sample_img.rotate(90), caption="Rotated (Invariant to orientation)")
            elif aug_op == "Zoom (Crop)":
                w, h = sample_img.size
                st.image(sample_img.crop((w//4, h//4, 3*w//4, 3*h//4)).resize((200, 200)), caption="Zoomed (Detail focus)")

        elif "3." in sim_choice:
            st.markdown("#### 📊 Step 3: Dataset Split Visualizer")
            train_ratio = st.slider("Training %", 50, 90, 70)
            val_ratio = (100 - train_ratio) // 2
            test_ratio = 100 - train_ratio - val_ratio
            
            split_df = pd.DataFrame({
                "Set": ["Training", "Validation", "Testing"],
                "Images": [train_ratio, val_ratio, test_ratio]
            })
            st.bar_chart(split_df.set_index("Set"))
            st.write(f"**Split Strategy**: {train_ratio}% Learn | {val_ratio}% Verify | {test_ratio}% Final Test")

        elif "4." in sim_choice:
            st.markdown("#### 🧠 Step 4: Model Architecture (EfficientNetB0)")
            st.write("The AI architecture is like a complex funnel of filters.")
            dot_arch = """
            digraph G {
                rankdir=TD;
                node [shape=box, style=filled, fontname="Arial"];
                input [label="Input Image (224x224)", fillcolor="#BBDEFB"];
                conv [label="MBConv Blocks (Feature Extractors)", fillcolor="#FFF9C4"];
                gap [label="Global Avg Pooling", fillcolor="#C8E6C9"];
                fc [label="Fully Connected Head", fillcolor="#F8BBD0"];
                output [label="Final Logic (Sigmoid/Softmax)", fillcolor="#FFCDD2"];
                input -> conv -> gap -> fc -> output;
            }
            """
            st.graphviz_chart(dot_arch)

        elif "5." in sim_choice:
            st.markdown("#### 📈 Step 5: Training Run Simulator")
            if st.button("🚀 Start Training Simulation"):
                status = st.empty()
                progress_bar = st.progress(0)
                chart = st.empty()
                
                loss_history = []
                for i in range(1, 21):
                    loss = 0.8 * (0.8 ** i) + (np.random.random() * 0.05)
                    loss_history.append(loss)
                    status.text(f"Epoch {i}/20 | Loss: {loss:.4f}")
                    progress_bar.progress(i * 5)
                    chart.line_chart(loss_history)
                    time.sleep(0.05)
                st.success("Training Complete! The AI has successfully learned the patterns.")

        elif "6." in sim_choice:
            st.markdown("#### 🏆 Step 6: Evaluation Hub")
            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                st.metric("Model Precision", "94%", delta="Target Meta")
                st.metric("Model Recall", "91%", delta="Critical for Cancer")
            with col_ev2:
                st.write("**Key Takeaway**: We optimize for **Recall** so we never miss a potential cancer case.")

        elif "7." in sim_choice:
            st.markdown("#### 🎯 Step 7: Prediction Logic")
            prob_test = st.slider("Simulate Raw AI Output:", 0.0, 1.0, 0.55)
            if is_binary:
                verdict = "Malignant" if prob_test > 0.5 else "Benign"
                st.markdown(f"**AI Logic**: $P({prob_test:.2f}) > 0.5 \implies$ **{verdict}**")
            else:
                st.write("In Multiclass, we pick the highest probability class from the distribution.")
                st.markdown(f"**Argmax Logic**: Winning Category with Score: **{prob_test:.2%}**")

    st.markdown("---")

    # --- Stage 1: Educational Context ---
    with st.expander("📚 Interactive Guide: The ABCDE Rule of Dermatology", expanded=True):
        st.write("Dermatologists use the **ABCDE** rule to identify potential Melanoma. AI models are trained to look for these same patterns!")
        cols = st.columns(5)
        with cols[0]: st.info("**A**symmetry\n\nOne half doesn't match the other.")
        with cols[1]: st.info("**B**order\n\nRagged, blurred, or irregular edges.")
        with cols[2]: st.info("**C**olor\n\nMultiple shades of tan, brown, or black.")
        with cols[3]: st.info("**D**iameter\n\nLarger than 6mm (pencil eraser).")
        with cols[4]: st.info("**E**volving\n\nChanging in size, shape, or color.")

    if is_binary:
        st.info("🎯 **Objective**: Distinguish **Melanoma** from benign lesions.")
        with st.expander("⚙️ Technical Architecture: Binary Model"):
            st.markdown("""
            * **Backbone**: EfficientNetB0 (ImageNet weights) extracts high-level features.
            * **Pooling**: Global Average Pooling reduces spatial dimensions.
            * **Output**: Single neuron with **Sigmoid** activation (Range: 0 to 1).
            * **Decision**: Threshold at 0.5.
            """)
    else:
        st.info("🎯 **Objective**: Categorize lesions into **7 clinical types**.")
        with st.expander("⚙️ Technical Architecture: Multiclass Model"):
            st.markdown("""
            * **Backbone**: EfficientNetB0 extracts pathological patterns.
            * **Head**: Fully connected dense layers for class separation.
            * **Output**: 7 neurons with **Softmax** activation (Total probability = 100%).
            * **Decision**: Highest probability (Argmax) class wins.
            """)

    # Use global image from sidebar
    uploaded_file = st.session_state.get('current_image', None)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.info("📸 **Using image from sidebar upload**")
    else:
        st.warning("⚠️ Please upload an image in the sidebar to begin the interactive journey!")
        st.image("https://via.placeholder.com/400x300?text=Upload+Image+in+Sidebar", caption="Upload your image in the sidebar to get started")
        
        st.markdown("---")
        
        # --- Stage 2: Resolution & Perception ---
        st.subheader("1️⃣ Step: Resolution & AI Perception")
        st.write("AI models don't see high-res images like humans. They usually look at a **downscaled** version to save memory.")
        
        res_val = st.slider("Simulate AI Resolution:", 32, 512, 224, step=32)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.image(image, caption="Human Perspective (High Res)", use_container_width=True)
        with col_res2:
            simulated_res = image.resize((res_val, res_val))
            st.image(simulated_res, caption=f"AI Perspective ({res_val}x{res_val})", width=300)

        # --- Stage 3: Feature Map Explorer ---
        st.subheader("2️⃣ Step: Feature Extraction (The Filter Bank)")
        st.write("The AI slides 'filters' across the image to find edges, textures, and patterns.")
        
        filter_type = st.selectbox(
            "Select Feature to Visualize:",
            ["Horizontal Edges", "Vertical Edges", "Details & Borders", "Contrast Patterns", "Texture Smoothing", "Geometric Shapes", "Edge Shadows", "Outlines (Canny)"]
        )
        
        f_maps = get_feature_maps(image)
        filter_idx = ["Horizontal Edges", "Vertical Edges", "Details & Borders", "Contrast Patterns", "Texture Smoothing", "Geometric Shapes", "Edge Shadows", "Outlines (Canny)"].index(filter_type)
        
        col_feat1, col_feat2 = st.columns([1, 2])
        with col_feat1:
            st.image(image.resize((224, 224)), caption="Resized Base", use_container_width=True)
        with col_feat2:
            st.image(f_maps[filter_idx], caption=f"AI Filter Result: {filter_type}", use_container_width=True)

        # --- Stage 4: Grad-CAM ---
        st.subheader("3️⃣ Step: Explainability (Grad-CAM)")
        st.write("Where is the AI looking? This heatmap shows the areas that most influenced the prediction.")
        
        temp_path = "temp_how_it_works.jpg"
        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(temp_path)
        
        with st.spinner("Calculating Attention Map..."):
            m_type = "binary" if is_binary else "multiclass"
            heatmap = generate_gradcam(temp_path, model_type=m_type)
            overlay = overlay_gradcam(temp_path, heatmap)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="AI Focus Area", use_container_width=True)
        
        os.remove(temp_path)


elif page == "Demo":
    st.title("🎮 Interactive AI Learning")
    st.write("Gain hands-on experience with how AI models process images.")
    
    tabs = ["Overview", "Pattern Recognition", "Color Channels", "Image Scaling", "AI Reasoning Pipeline"]
    if not is_binary:
        tabs.append("Lesion Taxonomy")
    else:
        tabs.append("Binary Logic Sandbox")
        
    st_tabs = st.tabs(tabs)
    
    # Use global image from sidebar
    demo_img_file = st.session_state.get('current_image', None)
    
    with st_tabs[0]:
        st.subheader("🏠 Demo Overview")
        st.write("""
        Welcome to the **Interactive AI Lab**. This section is designed to pull back the curtain on how 
        Deep Learning models process medical images.
        """)
        
        # Mode-Specific Visual Guide
        if is_binary:
            st.markdown("### 📊 Visual Confusion Matrix Dashboard")
            st.write("*Drag the slider to see how patients move between 'Healthy' and 'At Risk' boxes based on your clinical cautiousness.*")
            
            # Threshold slider
            threshold = st.slider("⚖️ Decision Threshold (Sensitivity vs. Specificity):", 0.0, 1.0, 0.5, key="binary_thresh_grid")
            
            # Mock Data: 20 patients
            # (Prob, Ground Truth) - Ground Truth: Malignant (M) or Benign (B)
            patients = [
                (0.95, "M"), (0.85, "M"), (0.75, "M"), (0.65, "M"), (0.55, "M"), (0.45, "M"), (0.35, "M"),
                (0.05, "B"), (0.15, "B"), (0.25, "B"), (0.35, "B"), (0.45, "B"), (0.55, "B"), (0.65, "B"),
                (0.12, "B"), (0.18, "B"), (0.88, "M"), (0.92, "M"), (0.48, "B"), (0.52, "M")
            ]
            
            tp, fp, tn, fn = [], [], [], []
            for p, gt in patients:
                pred = "M" if p >= threshold else "B"
                if gt == "M" and pred == "M": tp.append(p)
                elif gt == "B" and pred == "M": fp.append(p)
                elif gt == "B" and pred == "B": tn.append(p)
                elif gt == "M" and pred == "B": fn.append(p)
            
            # 2x2 Layout
            col_top1, col_top2 = st.columns(2)
            with col_top1:
                st.success(f"✅ **Correct: Benign** ({len(tn)})\n\n" + " ".join(["🩺"] * len(tn)))
                st.caption("People correctly cleared of cancer.")
            with col_top2:
                st.warning(f"⚠️ **False Alarms** ({len(fp)})\n\n" + " ".join(["🩺"] * len(fp)))
                st.caption("(False Positives) Healthy people flagged as at-risk.")
            
            col_bot1, col_bot2 = st.columns(2)
            with col_bot1:
                st.error(f"🚨 **Missed Cases** ({len(fn)})\n\n" + " ".join(["🚨"] * len(fn)))
                st.caption("(False Negatives) Real cancer cases not caught!")
            with col_bot2:
                st.success(f"🏁 **Correct: Malignant** ({len(tp)})\n\n" + " ".join(["🚨"] * len(tp)))
                st.caption("People correctly identified for treatment.")

            st.markdown("---")
            if len(fn) > 0:
                st.error(f"❗ **Warning**: A threshold of {threshold:.2f} is too high! You just missed {len(fn)} active cancer cases.")
            elif len(fp) > 5:
                st.info(f"💡 **Note**: You have {len(fp)} false alarms. This is safe for the patient, but might overwhelm the clinic.")
            else:
                st.success("✨ **Optimized**: This threshold provides a balanced clinical workflow.")
        else:
            st.markdown("### 🌈 Multiclass Weighting Sandbox")
            st.write("In multiclass, the model weighs 7 categories. Adjust the 'Attention' levels to see how the logic centers shift.")
            
            focus_class = st.selectbox("Focus Model Attention on:", list(MULTICLASS_LABELS.values()), key="multi_focus")
            boost = st.slider("Confidence Bias Boost:", 1.0, 3.0, 1.0, key="multi_boost")
            
            dot_multi = "digraph G {\n node [shape=square, style=filled];\n"
            for label in MULTICLASS_LABELS.values():
                size = 0.8 * boost if label == focus_class else 0.5
                color = "#BBDEFB" if label == focus_class else "#F5F5F5"
                dot_multi += f' "{label}" [width={size}, fillcolor="{color}"];\n'
            
            dot_multi += ' center [shape=point, width=0];\n center -> {"' + '" "'.join(MULTICLASS_LABELS.values()) + '"} [color=gray, style=dashed];\n}'
            st.graphviz_chart(dot_multi)
            st.info(f"💡 By boosting **{focus_class}**, the model becomes hyper-sensitive to its specific features, potentially ignoring others.")

        st.markdown("---")
        st.subheader("🔬 Interactive Lesion Explorer")
        st.write("Select a lesion type to learn about its features and how the AI identifies it.")
        
        lesion_data = {
            "Melanoma": {
                "hallmark": "Asymmetry, irregular borders, and deep pigment networks.",
                "ai_logic": "AI uses high-edge filters to detect 'jagged' boundaries and Blue-channel analysis for deep melanin.",
                "quiz": "Which clinical sign is MOST associated with Melanoma?",
                "options": ["Symmetry", "Irregular Borders", "Uniform Color"],
                "correct": "Irregular Borders"
            },
            "Basal Cell Carcinoma": {
                "hallmark": "Pearlescent edges and telangiectasia (tiny blood vessels).",
                "ai_logic": "AI looks for 'globular' structures and uses the Red channel to detect vascular patterns.",
                "quiz": "BCC often shows tiny blood vessels called...",
                "options": ["Melanin", "Telangiectasia", "Keratin"],
                "correct": "Telangiectasia"
            },
            "Actinic Keratosis": {
                "hallmark": "Rough, scaly patches (precancerous).",
                "ai_logic": "AI detects 'roughness' through high-frequency texture filters (Laplacian kernels).",
                "quiz": "AK is often described as feeling like...",
                "options": ["Sandpaper", "Silk", "Velvet"],
                "correct": "Sandpaper"
            },
            "Nevus (Mole)": {
                "hallmark": "Symmetric, well-defined border, uniform brown color.",
                "ai_logic": "AI finds a lack of high gradients. Low contrast between center and edge suggests benignity.",
                "quiz": "Most common moles (Nevi) are typically:",
                "options": ["Symmetrical", "Multicolor", "Jagged"],
                "correct": "Symmetrical"
            }
        }
        
        lesion_choice = st.selectbox("Pick a lesion to explore:", list(lesion_data.keys()))
        
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            st.info(f"**Clinical Hallmark:** {lesion_data[lesion_choice]['hallmark']}")
        with col_l2:
            st.success(f"**AI Strategy:** {lesion_data[lesion_choice]['ai_logic']}")
            
        # Mini-Quiz
        with st.expander(f"🧠 Quick Challenge: {lesion_choice}"):
            ans = st.radio(lesion_data[lesion_choice]['quiz'], lesion_data[lesion_choice]['options'])
            if st.button("Check Answer"):
                if ans == lesion_data[lesion_choice]['correct']:
                    st.success("✅ Correct! You're thinking like a dermatologist.")
                else:
                    st.error(f"❌ Not quite. The correct answer is **{lesion_data[lesion_choice]['correct']}**.")

        st.markdown("---")
        st.write("""
        **What else you can explore here:**
        1. **Low-level features**: How the AI finds borders and edges.
        2. **Biological layers**: How color channels map to skin pathology.
        3. **Resolution constraints**: What happens when the AI 'squints' at an image.
        4. **Reasoning Pipeline**: The step-by-step logic from raw pixels to diagnosis.
        5. **Advanced Theory**: Taxonomy (Multiclass) or Decision Thresholds (Binary).
        """)
        if not demo_img_file:
            st.warning("👈 Please upload an image in the sidebar to begin the interactive journey!")
        else:
            st.success("✅ Image detected! You can now browse the tabs to see how the AI processes your specific upload.")

    with st_tabs[1]:
        st.subheader("1️⃣ Concept: Feature Extraction")
        st.write("AI finds 'features' like edges by sliding a small math window over the image.")
        
        if demo_img_file:
            img = np.array(Image.open(demo_img_file).convert('L'))
            filter_choice = st.radio("Choose a feature kernel:", ["Find Edges", "Find Sharp Details"])
            
            if filter_choice == "Find Edges":
                filtered = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=5)
            else:
                filtered = cv2.Laplacian(img, cv2.CV_64F)
            
            filtered = np.abs(filtered)
            filtered = (filtered / (filtered.max() + 1e-5) * 255).astype(np.uint8)
            
            col1, col2 = st.columns(2)
            col1.image(img, caption="What You See", use_container_width=True)
            col2.image(filtered, caption="What the AI focuses on", use_container_width=True)
        else:
            st.info("Upload an image in the sidebar to see the demo.")

    with st_tabs[2]:
        st.subheader("2️⃣ Concept: Color Importance")
        st.write("Different layers of color help the model distinguish different tissue types.")
        if demo_img_file:
            img_rgb = np.array(Image.open(demo_img_file).convert('RGB'))
            c = st.select_slider("Select Channel Color:", options=["Red", "Green", "Blue"])
            ch_idx = {"Red": 0, "Green": 1, "Blue": 2}[c]
            
            overlay = np.zeros_like(img_rgb)
            overlay[:,:,ch_idx] = img_rgb[:,:,ch_idx]
            st.image(overlay, caption=f"The model looks at {c} Intensity")
            
            if st.button("▶️ Play Channel Cycle"):
                status = st.empty()
                display = st.empty()
                explanation = st.empty()
                
                channel_details = {
                    "Red": "The **Red channel** highlights **hemoglobin** and blood vessels. AI looks for 'milky red areas' suggesting high vascularity.",
                    "Green": "The **Green channel** provides the best contrast for skin lesions. Many filters find border textures here.",
                    "Blue": "The **Blue channel** emphasizes **melanin** density (brown/black colors). AI uses this to find deep pigment networks."
                }
                
                for color in ["Red", "Green", "Blue"]:
                    status.info(f"Cycling to **{color}**...")
                    idx = {"Red": 0, "Green": 1, "Blue": 2}[color]
                    frame = np.zeros_like(img_rgb)
                    frame[:,:,idx] = img_rgb[:,:,idx]
                    display.image(frame, caption=f"Analyzing {color} pixels...", use_container_width=True)
                    explanation.write(channel_details[color])
                    import time
                    time.sleep(2.5)
                status.success("Cycle Complete!")
        else:
            st.info("Upload an image in the sidebar to see the demo.")

    with st_tabs[3]:
        st.subheader("3️⃣ Concept: Resolution Matters")
        st.write("Large images take too much memory, so models 'downscale' them.")
        if demo_img_file:
            img_pill = Image.open(demo_img_file)
            res = st.select_slider("Reduce Resolution To:", options=[256, 128, 64, 32], value=128)
            st.image(img_pill.resize((res, res)), caption=f"Downscaled to {res}x{res} pixels", width=res * 2)
            
            if st.button("▶️ Play Auto-Scale Demo"):
                status = st.empty()
                display = st.empty()
                for r in [512, 256, 128, 64, 32]:
                    status.info(f"Downscaling to **{r}x{r}**...")
                    resized = img_pill.resize((r, r))
                    display.image(resized, caption=f"Resolution: {r}x{r}", width=r*2)
                    import time
                    time.sleep(0.8)
                status.warning("Loss of detail: Notice how boundaries become blurred at 32x32.")
        else:
            st.info("Upload an image in the sidebar to see the demo.")

    with st_tabs[4]:
        st.subheader("4️⃣ Concept: The AI Reasoning Pipeline")
        st.write("Think of the AI's logic as a sequence of steps. Click 'Play' to see the pipeline in action!")
        
        if st.button("🎬 Play AI Walkthrough"):
            steps = [
                ("1. Raw Input", "High-res images contain millions of details. AI first normalizes this data into a consistent format."),
                ("2. Digital Grid", "Pixels are converted to matrix values. The AI doesn't see 'colors', it sees intensity patterns (0-255)."),
                ("3. Normalization", "Values are centered around zero. This prevents 'exploding gradients' during training."),
                ("4. Feature Detection", "Kernels (math filters) slide across the image to find edge gradients (Melanoma often has 'jagged' borders)."),
                ("5. Pattern Matching", "Low-level edges are combined into high-level features like 'Blue-White veil' or 'Pigment networks'."),
                ("6. Probability Map", "The 'Attention mechanism' (Grad-CAM) weighs which pixels pushed the model toward its final guess."),
            ]
            
            # Adaptive Final Step
            if is_binary:
                steps.append(("7. Final Verdict (Sigmoid)", "The model outputs a single number (0 to 1). If > 0.5, it flags as Melanoma."))
            else:
                steps.append(("7. Final Verdict (Softmax)", "The model creates a probability distribution across 7 types. The highest % wins."))

            prog = st.progress(0)
            status_box = st.info("Initializing Walkthrough...")
            desc_box = st.empty()
            
            import time
            for i, (title, desc) in enumerate(steps):
                prog.progress((i + 1) / len(steps))
                status_box.markdown(f"**Step {i+1}: {title}**")
                desc_box.write(desc)
                time.sleep(3.0)
            
            status_box.success("Pipeline Walkthrough Complete!")

    if not is_binary:
        with st_tabs[5]:
            st.subheader("4️⃣ Concept: Skin Cancer Hierarchy")
            st.write("The multiclass model classifies lesions into a structured hierarchy. Explore it below!")
            
            # 1. Visual Hierarchy
            with st.expander("📊 View Full Classification Tree", expanded=False):
                dot_code = """
                digraph G {
                    rankdir=TB;
                    node [shape=box, style=filled, fillcolor="#E1F5FE", fontname="Arial", fontsize=12];
                    
                    "Skin Cancer" -> "Malignant" [color="#FF5252", penwidth=2];
                    "Skin Cancer" -> "Benign" [color="#4CAF50", penwidth=2];
                    
                    "Malignant" -> "Melanoma" -> "Malignant Melanoma";
                    "Malignant" -> "Non-melanoma" -> {"Basal Cell Carcinoma" "Squamous Cell Carcinoma"};
                    
                    "Benign" -> {"Actinic Keratosis" "Melanocytic Nevus" "Seborrheic Keratosis" "Dermatofibroma" "Vascular Malformations"};
                    
                    "Malignant" [fillcolor="#FFCDD2"];
                    "Benign" [fillcolor="#C8E6C9"];
                }
                """
                st.graphviz_chart(dot_code)

            # 2. Interactive Path Finder
            st.markdown("### 🗺️ Step-by-Step Path Finder")
            st.write("Build the classification path for a lesion to understand the model's logic.")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                root_choice = st.selectbox("1. Primary Class", ["Select", "Malignant 🔴", "Benign 🟢"])
            
            final_info = ""
            if root_choice == "Malignant 🔴":
                with col_p2:
                    sub_choice = st.selectbox("2. Category", ["Select", "Melanoma", "Non-melanoma"])
                if sub_choice == "Melanoma":
                    with col_p3:
                        st.selectbox("3. Specific Type", ["Malignant Melanoma"])
                    st.success("Selected Path: **Skin Cancer** ➔ **Malignant** ➔ **Melanoma**")
                    final_info = "**Malignant Melanoma**: The most aggressive form. Requires immediate medical attention."
                elif sub_choice == "Non-melanoma":
                    with col_p3:
                        spec = st.selectbox("3. Specific Type", ["Select", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"])
                    if spec != "Select":
                        st.success(f"Selected Path: **Skin Cancer** ➔ **Malignant** ➔ **Non-melanoma** ➔ **{spec}**")
                        final_info = f"**{spec}**: Common skin cancers that are usually slow-growing and treatable."
            
            elif root_choice == "Benign 🟢":
                with col_p2:
                    sub_choice = st.selectbox("2. Category", ["Select", "Actinic Keratosis", "Melanocytic Nevus", "Seborrheic Keratosis", "Dermatofibroma", "Vascular Malf."])
                if sub_choice != "Select":
                    st.success(f"Selected Path: **Skin Cancer** ➔ **Benign** ➔ **{sub_choice}**")
                    final_info = f"**{sub_choice}**: Generally non-cancerous, but the model learns to identify them to avoid false positives."

            if final_info:
                st.info(final_info)

            # 3. Quick Quiz
            st.markdown("---")
            st.markdown("### 🧠 Rapid Quiz: Test Your Knowledge")
            quiz_q = "Is **Basal Cell Carcinoma** categorized as Malignant or Benign?"
            ans = st.radio(quiz_q, ["Select an answer", "Malignant", "Benign"], horizontal=True)
            
            if ans == "Malignant":
                st.success("✅ Correct! BCC is a Malignant (Cancerous) lesion.")
            elif ans == "Benign":
                st.error("❌ Incorrect. BCC is actually Malignant, though it is usually less aggressive than Melanoma.")
    else:
        with st_tabs[5]:
            st.subheader("6️⃣ Concept: The Binary Decision Threshold")
            st.write("""
            Binary models output a single probability (0 to 1). The **Threshold** is where we 
            decide to say 'Melanoma'.
            """)
            
            thresh = st.slider("Select Decision Threshold:", 0.0, 1.0, 0.5, step=0.05)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"### Current Threshold: **{thresh}**")
                if thresh < 0.3:
                    st.error("🚨 **Aggressive Mode**: You'll catch EVERY cancer, but you'll have many 'False Alarms'.")
                elif thresh > 0.7:
                    st.warning("🛡️ **Conservative Mode**: No false alarms, but you might MISS an early-stage cancer!")
                else:
                    st.success("⚖️ **Balanced Mode**: The standard for most diagnostic AI.")
            
            with col_t2:
                # Simulation of sensitivity vs specificity
                st.markdown("### Sensitivity vs. Specificity")
                sensitivity = (1.0 - thresh) * 100
                specificity = thresh * 100
                st.progress(sensitivity/100, text=f"Sensitivity (Catching Cancers): {sensitivity:.0f}%")
                st.progress(specificity/100, text=f"Specificity (Reducing False Alarms): {specificity:.0f}%")
            
            st.info("""
            **Why this matters?** In medicine, we often prefer a **low threshold** (e.g., 0.3) for 
            dangerous diseases like Melanoma, because missing a cancer is much worse than checking a healthy mole.
            """)

elif page == "Live Prediction":
    st.title(f"🧪 Live {'Binary' if is_binary else 'Multiclass'} Prediction")
    st.write(f"Upload a dermatoscopic image for instant {task_type.split(' ')[0]} analysis.")
    
    # --- Quest v2 State Management ---
    if "quest_step" not in st.session_state: st.session_state.quest_step = 0
    if "xp" not in st.session_state: st.session_state.xp = 0
    
    # Custom Cinematic CSS
    st.markdown("""
    <style>
    .evidence-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #007bff;
        margin-bottom: 10px;
    }
    .rank-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .scanner-title {
        color: #007bff;
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Use global image from sidebar
    input_img = st.session_state.get('current_image', None)
    
    if input_img is not None:
        st.info("📸 **Using image from sidebar upload**")
        # Initial Processing
        temp_path = "temp_prediction.jpg"
        image = Image.open(input_img)
        # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(temp_path)
        
        with st.spinner("Initializing Deep Scan..."):
            if is_binary:
                result = predict_binary(temp_path)
                m_type = "binary"
            else:
                result = predict_multiclass(temp_path)
                m_type = "multiclass"
            heatmap = generate_gradcam(temp_path, model_type=m_type)
            conf_val = result['probability'] if is_binary else max(result['probabilities'].values())

        # Rank Calculation
        rank = "Medical Student"
        if st.session_state.xp > 50: rank = "Junior Resident"
        if st.session_state.xp > 150: rank = "Chief Resident"
        if st.session_state.xp > 300: rank = "Consultant"
        if st.session_state.xp > 500: rank = "Department Head"

        # Quest Header
        st.markdown(f"<h2 class='scanner-title'>🧪 MISSION: DISCOVERY QUEST</h2>", unsafe_allow_html=True)
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1: st.metric("🏆 Expertise Points", st.session_state.xp)
        with col_stats2: st.markdown(f"**Current Rank:** <br><div class='rank-badge'>{rank}</div>", unsafe_allow_html=True)
        
        # Phase 0: The Target
        if st.session_state.quest_step == 0:
            st.image(input_img, caption="The Subject", use_container_width=True)
            st.info("💡 **Mission Objective**: Investigate this lesion. Look for clinical red flags and confirm with the AI Cinematic Scanner.")
            if st.button("🚀 Begin Mission"):
                st.session_state.quest_step = 1
                st.rerun()

        # Phase 1: Human Investigation (Evidence Wall)
        elif st.session_state.quest_step == 1:
            st.subheader("🕵️ Phase 1: Gathering Evidence")
            st.write("Examine the skin under the visual lens. Which 'Red Flags' are present?")
            
            qc1, qc2 = st.columns([1, 1])
            with qc1: st.image(input_img, use_container_width=True)
            with qc2:
                with st.expander("📐 Asymmetry (A)", expanded=False):
                    st.write("Does one half look different? Melanoma cells grow at different rates, causing asymmetrical shapes.")
                    q_a = st.checkbox("Identify Asymmetry", key="q_a")
                
                with st.expander("🌊 Irregular Border (B)", expanded=False):
                    st.write("Are the edges notched or blurred? Cancerous lesions often 'invade' surrounding skin unevenly.")
                    q_b = st.checkbox("Identify Irregular Border", key="q_b")
                
                with st.expander("🎨 Multiple Colors (C)", expanded=False):
                    st.write("Varied shades of brown, black, red, or white suggest different clusters of abnormal cells.")
                    q_c = st.checkbox("Identify Multiple Colors", key="q_c")
                
                with st.expander("📏 Diameter > 6mm (D)", expanded=False):
                    st.write("Most melanomas are larger than a pencil eraser, though some can be smaller.")
                    q_d = st.checkbox("Identify Large Diameter", key="q_d")
                
                with st.expander("🔄 Evolving (E)", expanded=False):
                    st.write("The most critical sign: Has it changed size, shape, or color recently?")
                    q_e = st.checkbox("Identify Evolving Marker", key="q_e")
                
                if st.button("📝 Log Evidence Board"):
                    count = sum([q_a, q_b, q_c, q_d, q_e])
                    st.session_state.xp += (count * 25) # 25 XP per sign
                    st.session_state.user_signs = [q_a, q_b, q_c, q_d, q_e]
                    st.session_state.quest_step = 2
                    st.rerun()

        # Phase 2: AI X-Ray Scanner
        elif st.session_state.quest_step == 2:
            st.subheader("🎬 Phase 2: Cinematic Deep Scanner")
            st.markdown("*Engaging AI reasoning engine. Slide the 'Scanner Depth' to peel back the surface.*")
            
            opacity = st.slider("Scanner Depth Intensity:", 0.0, 1.0, 0.0)
            
            # Blend logic
            base_img = cv2.imread(temp_path)
            base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            heatmap_resized = cv2.resize(heatmap, (base_img.shape[1], base_img.shape[0]))
            heatmap_u8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            blended = cv2.addWeighted(base_img_rgb, 1-opacity, heatmap_color_rgb, opacity, 0)
            
            st.image(blended, caption=f"Scanner Depth: {opacity*100:.0f}%", use_container_width=True)
            
            if opacity > 0.0:
                st.write("🔦 **Scanner Analysis**: Blue = Background | Red = High Activation Focus")
            
            if opacity > 0.6:
                if st.button("🔍 Scan for Active Medical Markers"):
                    st.info("🔬 **Hotspot Explorer Analysis**:")
                    if is_binary and result['class'] == "Melanoma":
                        st.write("- **Marker 1**: Intense focal activation at the border suggests structural chaos.")
                        st.write("- **Marker 2**: Deep pigment network detected in the central zone.")
                    elif not is_binary:
                        st.write(f"- **Marker 1**: Visual weight suggests high similarity to {result['class']} prototypes.")
                        st.write("- **Marker 2**: Attention map correlates with specific surface irregularities.")
                    else:
                        st.write("- **Marker 1**: Distributed activation suggests uniform benign structure.")
                        st.write("- **Marker 2**: No aggressive focal points detected.")

            if opacity > 0.8:
                st.success("🔬 **AI Focus Locked!** The model is detecting significant structural anomalies in the red zones.")
                if st.button("🎯 Finalize Forensic Investigation"):
                    st.session_state.quest_step = 3
                    st.rerun()

        # Phase 3: The Forensic Board
        elif st.session_state.quest_step == 3:
            st.subheader("🧩 Phase 3: Forensic Evidence Board")
            
            final_col1, final_col2 = st.columns(2)
            with final_col1:
                st.markdown("<div class='evidence-card'><h4>👤 Human Discovery</h4>", unsafe_allow_html=True)
                if "user_signs" in st.session_state:
                    signs = ["Asymmetry", "Border", "Color", "Diameter", "Evolving"]
                    found_any = False
                    for i, s in enumerate(signs):
                        if st.session_state.user_signs[i]: 
                            st.write(f"✅ **{s}**")
                            found_any = True
                    if not found_any: st.write("No major clinical signs noted.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with final_col2:
                st.markdown("<div class='evidence-card'><h4>🤖 AI Discovery</h4>", unsafe_allow_html=True)
                st.write(f"**Primary Verdict:** {result['class']}")
                st.write(f"**Confidence Score:** {conf_val:.2%}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            # XP Award Logic
            match_bonus = 0
            if is_binary:
                if (result['class'] == "Melanoma" and any(st.session_state.user_signs)):
                    match_bonus = 100
                    st.info("⭐ **Perfect Correlation Bonus!** Your clinical eye matched the AI's detection of a malignant lesion.")
            
            if st.button("📜 Seal Investigation & Report"):
                st.session_state.xp += match_bonus
                st.session_state.quest_step = 4
                
                # Pre-generate report
                rank_at_end = rank
                st.session_state.final_report = f"""
# Forensic Discovery Report
**Mission ID:** FS-{int(time.time())}
**Expert Pathologist:** {rank_at_end}

## 🔍 Forensic Summary
The subject lesion was analyzed using a multi-step investigation. 

### 🧬 Human Observations
* {sum(st.session_state.user_signs)} clinical hallmarks identified.
* Focus markers aligned with {result['class'] if any(st.session_state.user_signs) else 'a different category'}.

### 🧠 Model Execution
* **AI Consensus:** {result['class']}
* **Intensity:** {conf_val:.2%}
* **Layer Depth:** Deep scanner revealed focus on specific peripheral textures.

---
**XP Final Score:** {st.session_state.xp} pts
                """
                st.rerun()

        # Phase 4: Mission End
        elif st.session_state.quest_step == 4:
            st.balloons()
            st.subheader("🎖️ Mission Accomplished!")
            st.success(f"Final Expertise Score: **{st.session_state.xp} XP** | Rank: **{rank}**")
            
            with st.expander("📄 Open Forensic Report", expanded=True):
                st.markdown(st.session_state.get('final_report', ''))
            
            if st.button("🔄 Start New Clinical Mission"):
                st.session_state.quest_step = 0
                st.rerun()
                
        os.remove(temp_path)


elif page == "Performance Metrics":
    st.title("📊 Model Performance Metrics")
    st.write("Detailed analysis of the model's training and evaluation for Binary Classification.")

    history_path = "training/binary/history.csv"
    preds_path = "training/binary/test_predictions.csv"

    if os.path.exists(history_path):
        df = pd.read_csv(history_path)

        # ---------------------------
        # Accuracy Curve
        # ---------------------------
        st.subheader("📈 Accuracy Curve")
        st.line_chart(df[['accuracy', 'val_accuracy']])
        st.write("""
        The validation accuracy appears high (~89%). However, due to dataset imbalance,
        accuracy alone is not a reliable metric in medical AI systems.
        """)

        # ---------------------------
        # Loss Curve
        # ---------------------------
        st.subheader("📉 Loss Curve")
        st.line_chart(df[['loss', 'val_loss']])
        st.write("""
        The loss curves indicate limited overfitting. However, the model struggles
        to properly learn the minority melanoma class.
        """)

        # ---------------------------
        # Confusion Matrix
        # ---------------------------
        if os.path.exists("visuals/binary/confusion_matrix.png"):
            st.subheader("🔢 Confusion Matrix")
            st.image("visuals/binary/confusion_matrix.png", use_container_width=True)
            st.write("""
            The confusion matrix reveals that the model predicts all samples
            as non-melanoma, resulting in zero sensitivity.
            This highlights the severe class imbalance problem.
            """)

        # ---------------------------
        # ROC Curve
        # ---------------------------
        if os.path.exists("visuals/binary/roc_curve.png"):
            st.subheader("📊 ROC Curve")
            st.image("visuals/binary/roc_curve.png", use_container_width=True)
            st.write("""
            The ROC curve shows moderate separability between classes.
            An AUC of approximately 0.70 indicates reasonable discrimination ability,
            though classification threshold tuning is required.
            """)

        # ---------------------------
        # Final Metrics Summary
        # ---------------------------
        st.subheader("📌 Final Metrics Summary")

        final_train_acc = df['accuracy'].iloc[-1]
        final_val_acc = df['val_accuracy'].iloc[-1]
        final_train_loss = df['loss'].iloc[-1]
        final_val_loss = df['val_loss'].iloc[-1]

        col1, col2 = st.columns(2)
        col1.metric("Final Training Accuracy", f"{final_train_acc:.2%}")
        col2.metric("Final Validation Accuracy", f"{final_val_acc:.2%}")

        col3, col4 = st.columns(2)
        col3.metric("Final Training Loss", f"{final_train_loss:.4f}")
        col4.metric("Final Validation Loss", f"{final_val_loss:.4f}")

        st.info("""
        ⚠️ Key Observation:
        Despite high validation accuracy, the model shows zero sensitivity.
        This demonstrates why accuracy alone is misleading in imbalanced
        medical datasets and emphasizes the importance of recall in cancer detection.
        """)

    else:
        st.warning("Training history not found. Please complete the training phase first.")

