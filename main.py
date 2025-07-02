import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import preprocess_image, make_gradcam_heatmap, superimpose_heatmap
import base64

# Page setup
st.set_page_config(page_title="🫁 Pneumonia Detector", layout="wide")

# Load models
model_cnn = load_model("cnn_v3.keras")
model_mobilenet = load_model("mobilenetv2.keras")
model_effnet = load_model("efficientnetb0_v3.keras")

models = {
    "Simple CNN": model_cnn,
    "MobileNetV2": model_mobilenet,
    "EfficientNetB0": model_effnet
}

class_names = ['NORMAL', 'PNEUMONIA']

# Header
st.markdown(
    """
    <div style='text-align:center; padding: 10px 0;'>
        <h1 style='color:#3C3C3C;'>🧠 Pneumonia Detection using Deep Learning Ensemble</h1>
        <p style='font-size:18px;'>Upload a chest X-ray to get predictions from 3 models and interpret it using Grad-CAM visualization.</p>
    </div>
    """, unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image.resize((400, 400)), caption="🖼️ Original Chest X-ray", use_container_width=True)
    with col2:
        try:
            heatmap = make_gradcam_heatmap(img_array, model_effnet, last_conv_layer_name="top_conv")
            cam_image = superimpose_heatmap(heatmap, image)
            st.image(Image.fromarray(cam_image).resize((400, 400)), caption="🔥 Grad-CAM (EfficientNetB0)", use_container_width=True)
        except Exception as e:
            st.error(f"Grad-CAM error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;'>🔍 Individual Model Predictions</h2>", unsafe_allow_html=True)
    probs, weights = [], []
    pred_cols = st.columns(3)

    for (name, model), col in zip(models.items(), pred_cols):
        prob = model.predict(img_array, verbose=0)[0][0]
        probs.append(prob)
        confidence = max(prob, 1 - prob)
        weights.append(confidence)

        with col:
            st.markdown(
                f"""
                <div style='background-color: #f7f7f7; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
                    <h4>{name}</h4>
                    <p style='font-size: 16px;'>🔴 <strong>PNEUMONIA:</strong> {prob * 100:.2f}%</p>
                    <p style='font-size: 16px;'>🟢 <strong>NORMAL:</strong> {(1 - prob) * 100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Ensemble Prediction - Confidence Weighted
    probs = np.array(probs)
    weights = np.array(weights)
    normalized_weights = weights / np.sum(weights)
    weighted_avg_prob = np.sum(probs * normalized_weights)
    final_class = "PNEUMONIA" if weighted_avg_prob > 0.5 else "NORMAL"

    # Final prediction box
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background-color: #e6f0ff; padding: 25px; border-radius: 15px; border: 2px solid #99ccff;
                    text-align: center; margin: auto; width: 70%; box-shadow: 3px 3px 10px rgba(0,0,0,0.1);'>
            <h2>🧠 Final Ensemble Prediction: 
                <span style='color: {"red" if final_class == "PNEUMONIA" else "green"};'>{final_class}</span></h2>
            <p style='font-size: 18px;'>🔴 PNEUMONIA Probability: <strong>{weighted_avg_prob * 100:.2f}%</strong></p>
            <p style='font-size: 18px;'>🟢 NORMAL Probability: <strong>{(1 - weighted_avg_prob) * 100:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Optional: Expandable section
    with st.expander("🧠 How the Ensemble Works"):
        st.write("""
        Each model outputs a probability score for the image. Instead of averaging directly,
        we assign **higher weight to models that are more confident**, using a dynamic confidence-based weighted voting system.
        This improves prediction robustness.
        """)

else:
    st.info("👆 Please upload a Chest X-ray to get started.")

    
    
        

  
