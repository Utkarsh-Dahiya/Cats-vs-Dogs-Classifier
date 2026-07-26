"""
🐱 Cats vs Dogs Classifier — Streamlit Web App 🐶
==================================================
A premium, modern web interface for the Cats vs Dogs image classification project.

Run locally with:
    streamlit run app.py

Expected model files in the same directory (produced by the training notebook):
    - cats_vs_dogs_cnn.keras            (CNN trained from scratch, input size 128x128)
    - cats_vs_dogs_mobilenetv2.keras    (MobileNetV2 transfer-learning model, input size 160x160)

Both saved models already contain their own Rescaling / preprocess_input layers internally,
so this app only needs to resize the uploaded image — no manual normalization required.
"""

import io
import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ["Cat", "Dog"]
CLASS_EMOJIS = {"Cat": "🐱", "Dog": "🐶"}
CLASS_COLORS = {
    "Cat": ("#FA709A", "#FEE140"),   # warm sunset gradient
    "Dog": ("#4FACFE", "#00F2FE"),   # cool ocean gradient
}

# Model registry: everything the UI needs to know about each available model.
MODEL_REGISTRY = {
    "MobileNetV2 (Transfer Learning)": {
        "path": "cats_vs_dogs_mobilenetv2.keras",
        "img_size": (160, 160),
        "description": "Pre-trained on ImageNet, fine-tuned on Cats vs Dogs. "
                        "Higher accuracy, recommended default.",
        "params_note": "~2.3M trainable params (after fine-tuning top layers)",
        "badge": "⚡ Recommended",
    },
    "CNN (Built From Scratch)": {
        "path": "cats_vs_dogs_cnn.keras",
        "img_size": (128, 128),
        "description": "A custom convolutional network trained from random weights. "
                        "Lighter, fully transparent architecture.",
        "params_note": "3 conv blocks + dense head",
        "badge": "🧩 Educational",
    },
}

# =============================================================================
# CUSTOM CSS — PREMIUM GLASSMORPHIC STYLING
# =============================================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Animated mesh gradient backdrop */
        .stApp {
            background: linear-gradient(-45deg, #ede7ff, #e0f7fa, #ffe8f3, #eaf3ff);
            background-size: 400% 400%;
            animation: gradientShift 18s ease infinite;
        }
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Floating decorative paw prints */
        .paw-field {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }
        .paw {
            position: absolute;
            font-size: 2.2rem;
            opacity: 0.08;
            animation: floatPaw 12s ease-in-out infinite;
        }
        @keyframes floatPaw {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50%      { transform: translateY(-30px) rotate(12deg); }
        }

        /* Hero header with gradient text + glow */
        .hero {
            background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%);
            padding: 2.6rem 2rem;
            border-radius: 24px;
            color: white;
            text-align: center;
            margin-bottom: 1.6rem;
            box-shadow: 0 20px 45px rgba(124, 58, 237, 0.35);
            position: relative;
            z-index: 1;
            overflow: hidden;
        }
        .hero::before {
            content: "";
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 60%);
            animation: shimmer 6s linear infinite;
        }
        @keyframes shimmer {
            0%   { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hero h1 {
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
            position: relative;
            z-index: 1;
            text-shadow: 0 4px 18px rgba(0,0,0,0.15);
        }
        .hero p {
            font-size: 1.1rem;
            opacity: 0.95;
            margin: 0;
            position: relative;
            z-index: 1;
        }

        /* Hero stat pills */
        .hero-badges {
            display: flex;
            justify-content: center;
            gap: 0.7rem;
            margin-top: 1.2rem;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }
        .hero-badge {
            background: rgba(255,255,255,0.18);
            backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,0.35);
            padding: 0.4rem 1rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Glassmorphic section cards */
        .section-card {
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(14px);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 8px 30px rgba(31, 38, 135, 0.08);
            border: 1px solid rgba(255,255,255,0.5);
            margin-bottom: 1.4rem;
            position: relative;
            z-index: 1;
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .section-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 14px 38px rgba(31, 38, 135, 0.14);
        }
        .section-card h3 {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
        }

        /* Result card with bounce + gradient */
        .result-card {
            border-radius: 22px;
            padding: 2rem 1.5rem;
            text-align: center;
            color: white;
            box-shadow: 0 16px 34px rgba(0,0,0,0.18);
            animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        @keyframes popIn {
            0%   { opacity: 0; transform: scale(0.85) translateY(10px); }
            100% { opacity: 1; transform: scale(1) translateY(0); }
        }
        .result-card .emoji-big {
            font-size: 3.4rem;
            display: inline-block;
            animation: bounce 1.4s ease infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50%      { transform: translateY(-8px); }
        }
        .result-card h2 {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            margin: 0.3rem 0 0.2rem 0;
        }
        .result-label {
            font-size: 0.95rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            opacity: 0.85;
        }

        /* Circular confidence ring */
        .confidence-ring {
            width: 130px; height: 130px;
            border-radius: 50%;
            margin: 1.1rem auto 0.6rem auto;
            display: flex; align-items: center; justify-content: center;
            box-shadow: inset 0 0 0 6px rgba(255,255,255,0.15);
        }
        .confidence-ring-inner {
            width: 102px; height: 102px;
            border-radius: 50%;
            background: rgba(255,255,255,0.97);
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
        }
        .confidence-value {
            font-family: 'Poppins', sans-serif;
            font-size: 1.5rem;
            font-weight: 800;
            color: #333;
        }
        .confidence-caption {
            font-size: 0.65rem;
            color: #888;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        /* Model badge chip in sidebar */
        .model-chip {
            display: inline-block;
            background: linear-gradient(135deg, #7C3AED, #06B6D4);
            color: white;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }

        /* File uploader dropzone polish (best-effort selector) */
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 16px !important;
            border: 2px dashed #a78bfa !important;
            background: rgba(124, 58, 237, 0.04) !important;
        }

        /* Footer */
        .app-footer {
            text-align: center;
            padding: 1.4rem 0 0.4rem 0;
            color: #8a8fa3;
            font-size: 0.85rem;
            border-top: 1px solid rgba(0,0,0,0.06);
            margin-top: 2rem;
            position: relative;
            z-index: 1;
        }
        .app-footer a { color: #7C3AED; text-decoration: none; font-weight: 600; }

        /* Sidebar polish */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0,0,0,0.06);
        }
    </style>

    <div class="paw-field">
        <span class="paw" style="top:8%; left:5%; animation-delay:0s;">🐾</span>
        <span class="paw" style="top:22%; left:85%; animation-delay:2s;">🐾</span>
        <span class="paw" style="top:65%; left:12%; animation-delay:4s;">🐾</span>
        <span class="paw" style="top:78%; left:90%; animation-delay:1s;">🐾</span>
        <span class="paw" style="top:45%; left:50%; animation-delay:3s;">🐾</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL LOADING (cached so it only happens once per session)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load a saved Keras model from disk. Cached across reruns for speed."""
    return tf.keras.models.load_model(model_path)


def preprocess_image(pil_image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Convert a PIL image into the (1, H, W, 3) float32 array the model expects.
    Note: normalization (Rescaling / preprocess_input) is already baked into the
    saved model architecture, so we only resize here.
    """
    img = pil_image.convert("RGB").resize(target_size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(array, axis=0)


def predict(model, img_array: np.ndarray) -> tuple[str, float]:
    """Run inference and return (predicted_label, confidence_percent)."""
    raw_prediction = float(model.predict(img_array, verbose=0)[0][0])
    predicted_idx = int(raw_prediction > 0.5)
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = raw_prediction if predicted_idx == 1 else 1 - raw_prediction
    return predicted_label, confidence * 100


# =============================================================================
# SIDEBAR — MODEL INFORMATION
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")

    selected_model_name = st.selectbox(
        "Choose a model",
        options=list(MODEL_REGISTRY.keys()),
        help="Switch between the from-scratch CNN and the MobileNetV2 transfer-learning model.",
    )
    model_info = MODEL_REGISTRY[selected_model_name]

    st.markdown(f'<span class="model-chip">{model_info["badge"]}</span>', unsafe_allow_html=True)

    st.markdown("### 📋 Model Details")
    st.markdown(
        f"""
        - **Architecture:** {selected_model_name}
        - **Input size:** {model_info['img_size'][0]} × {model_info['img_size'][1]}
        - **Details:** {model_info['params_note']}
        """
    )
    st.info(model_info["description"])

    st.markdown("---")
    st.markdown("### 🧠 About This Project")
    st.markdown(
        """
        This app classifies images as **Cat 🐱** or **Dog 🐶** using a deep learning
        model trained in TensorFlow/Keras — comparing a custom CNN against
        transfer learning with MobileNetV2.
        """
    )

    st.markdown("---")
    st.caption(f"TensorFlow version: `{tf.__version__}`")


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🐾 Cats vs Dogs Classifier</h1>
        <p>Upload a photo and let a deep learning model decide — cat or dog?</p>
        <div class="hero-badges">
            <div class="hero-badge">🧠 Deep Learning</div>
            <div class="hero-badge">⚡ TensorFlow / Keras</div>
            <div class="hero-badge">🎯 Instant Prediction</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# PROJECT DESCRIPTION
# =============================================================================
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### 📌 About
        This demo is powered by a **Convolutional Neural Network** and a
        **MobileNetV2 transfer-learning model**, both trained on the Microsoft
        Cats vs Dogs dataset. Pick a model in the sidebar, upload a clear photo
        of a cat or a dog below, and the app will predict the class along with
        a confidence score.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# MAIN LAYOUT — UPLOAD + RESULTS
# =============================================================================
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Best results with a clear, well-lit photo of a single cat or dog.",
    )

    pil_image = None
    if uploaded_file is not None:
        try:
            pil_image = Image.open(io.BytesIO(uploaded_file.getvalue()))
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        except UnidentifiedImageError:
            st.error("⚠️ That file doesn't look like a valid image. Please upload a JPG or PNG.")
        except Exception as exc:  # noqa: BLE001 - surface any unexpected read error to the user
            st.error(f"⚠️ Couldn't read that file: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Prediction")

    if uploaded_file is None:
        st.markdown(
            "👋 Upload an image on the left to see the model's prediction here."
        )

    elif pil_image is not None:
        try:
            with st.spinner("🧠 Analyzing image..."):
                model = load_model(model_info["path"])
                img_array = preprocess_image(pil_image, model_info["img_size"])
                time.sleep(0.3)  # small delay so the spinner is visible for tiny models
                label, confidence = predict(model, img_array)

            start_color, end_color = CLASS_COLORS[label]
            emoji = CLASS_EMOJIS[label]

            st.markdown(
                f"""
                <div class="result-card" style="background: linear-gradient(135deg, {start_color} 0%, {end_color} 100%);">
                    <div class="result-label">Prediction</div>
                    <div class="emoji-big">{emoji}</div>
                    <h2>{label}</h2>
                    <div class="confidence-ring" style="background: conic-gradient(white {confidence:.1f}%, rgba(255,255,255,0.25) {confidence:.1f}% 100%);">
                        <div class="confidence-ring-inner">
                            <div class="confidence-value">{confidence:.1f}%</div>
                            <div class="confidence-caption">Confidence</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(f"Model used: {selected_model_name}")

            # 🎉 A little delight for a highly confident prediction
            if confidence >= 90:
                st.balloons()

        except FileNotFoundError:
            st.error(
                f"⚠️ Model file `{model_info['path']}` was not found. "
                "Make sure it's saved in the same folder as this app."
            )
        except Exception as exc:  # noqa: BLE001 - fail gracefully, never crash the UI
            st.error(f"⚠️ Something went wrong during prediction: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    """
    <div class="app-footer">
        Built with ❤️ using TensorFlow &amp; Streamlit ·
        Cats vs Dogs Classification Project ·
        <a href="https://github.com/" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
