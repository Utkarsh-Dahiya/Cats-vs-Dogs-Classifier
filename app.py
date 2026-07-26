"""
🐱 Cats vs Dogs Classifier — Streamlit Web App 🐶
==================================================
A clean, modern web interface for the Cats vs Dogs image classification project.

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

# Model registry: everything the UI needs to know about each available model.
MODEL_REGISTRY = {
    "MobileNetV2 (Transfer Learning)": {
        "path": "cats_vs_dogs_mobilenetv2.keras",
        "img_size": (160, 160),
        "description": "Pre-trained on ImageNet, fine-tuned on Cats vs Dogs. "
                        "Higher accuracy, recommended default.",
        "params_note": "~2.3M trainable params (after fine-tuning top layers)",
    },
    "CNN (Built From Scratch)": {
        "path": "cats_vs_dogs_cnn.keras",
        "img_size": (128, 128),
        "description": "A custom convolutional network trained from random weights. "
                        "Lighter, fully transparent architecture.",
        "params_note": "~3 conv blocks + dense head",
    },
}

# =============================================================================
# CUSTOM CSS — MODERN, CLEAN STYLING
# =============================================================================
st.markdown(
    """
    <style>
        /* Overall page */
        .stApp {
            background: linear-gradient(180deg, #fafbff 0%, #f3f5fb 100%);
        }

        /* Hide default Streamlit chrome for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Hero header */
        .hero {
            background: linear-gradient(135deg, #6C5CE7 0%, #00B8D9 100%);
            padding: 2.2rem 2rem;
            border-radius: 18px;
            color: white;
            text-align: center;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(108, 92, 231, 0.25);
        }
        .hero h1 {
            font-size: 2.4rem;
            margin-bottom: 0.3rem;
            font-weight: 800;
        }
        .hero p {
            font-size: 1.05rem;
            opacity: 0.92;
            margin: 0;
        }

        /* Section card */
        .section-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem 1.7rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06);
            margin-bottom: 1.4rem;
            border: 1px solid #eef0f7;
        }

        /* Result cards */
        .result-card {
            border-radius: 18px;
            padding: 1.8rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 24px rgba(0,0,0,0.12);
            animation: fadeIn 0.5s ease-in-out;
        }
        .result-card.dog {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }
        .result-card.cat {
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }
        .result-card h2 {
            font-size: 2.6rem;
            margin: 0.2rem 0;
        }
        .result-card .confidence {
            font-size: 1.15rem;
            font-weight: 600;
            opacity: 0.95;
        }

        /* Confidence bar */
        .conf-bar-bg {
            background: rgba(255,255,255,0.35);
            border-radius: 10px;
            height: 14px;
            width: 100%;
            margin-top: 0.8rem;
            overflow: hidden;
        }
        .conf-bar-fill {
            background: white;
            height: 100%;
            border-radius: 10px;
        }

        /* Footer */
        .app-footer {
            text-align: center;
            padding: 1.2rem 0 0.4rem 0;
            color: #8a8fa3;
            font-size: 0.85rem;
            border-top: 1px solid #e5e7ef;
            margin-top: 2rem;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(8px);}
            to {opacity: 1; transform: translateY(0);}
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #eef0f7;
        }
    </style>
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

            css_class = "dog" if label == "Dog" else "cat"
            emoji = CLASS_EMOJIS[label]

            st.markdown(
                f"""
                <div class="result-card {css_class}">
                    <div style="font-size:1rem; opacity:0.9;">Prediction</div>
                    <h2>{emoji} {label}</h2>
                    <div class="confidence">Confidence: {confidence:.2f}%</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{confidence:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(f"Model used: {selected_model_name}")

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