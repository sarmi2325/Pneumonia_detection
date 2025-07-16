import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import preprocess_image, make_gradcam_heatmap, superimpose_heatmap

# Page setup
st.set_page_config(page_title="Pneumonia Detector", layout="wide")

# Load models
model_mobilenet = load_model("mobilenet_distilled.keras")
model_effnet = load_model("efficientnetb0_v3.keras")

models = {
    "MobileNetV2": model_mobilenet,
    "EfficientNetB0": model_effnet
}

gradcam_layers = {
    "MobileNetV2": "block_13_expand",
    "EfficientNetB0": "top_conv"
}

class_names = ['NORMAL', 'PNEUMONIA']

# UI Header
st.markdown("""
<div style='text-align:center; padding: 10px 0;'>
    <h1>Pneumonia Detection using Deep Learning Ensemble</h1>
    <p style='font-size:18px;'>Upload a chest X-ray to get predictions from 2 optimized models and visualize key regions with Grad-CAM.</p>
</div>
""", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    st.markdown("<hr>", unsafe_allow_html=True)
   

    # Grad-CAM
    st.markdown("<h2 style='text-align:center;'>Grad-CAM Visualizations</h2>", unsafe_allow_html=True)
    gradcam_cols = st.columns(2)
    for (name, model), col in zip(models.items(), gradcam_cols):
        with col:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=gradcam_layers[name])
                cam_image = superimpose_heatmap(heatmap, image)
                st.image(Image.fromarray(cam_image).resize((300, 300)), caption=f"Grad-CAM ({name})", use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Grad-CAM error for {name}: {e}")

    # Predictions
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Model Predictions</h2>", unsafe_allow_html=True)

    probs, weights = [], []
    pred_cols = st.columns(2)
    for (name, model), col in zip(models.items(), pred_cols):
        prob = model.predict(img_array, verbose=0)[0][0]
        probs.append(prob)
        confidence = max(prob, 1 - prob)
        weights.append(confidence)

        with col:
            st.markdown(f"""
            <div style='background-color: #ffffff; padding: 15px; border-radius: 12px; text-align: center;
                        border: 1px solid #ccc; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); color: #000000;'>
                <h4>{name}</h4>
                <p style='font-size: 16px;'>🔴 <strong>PNEUMONIA:</strong> {prob * 100:.2f}%</p>
                <p style='font-size: 16px;'>🟢 <strong>NORMAL:</strong> {(1 - prob) * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # Ensemble Voting
    probs = np.array(probs)
    weights = np.array(weights)
    normalized_weights = weights / np.sum(weights)
    weighted_avg_prob = np.sum(probs * normalized_weights)
    final_class = "PNEUMONIA" if weighted_avg_prob > 0.5 else "NORMAL"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color: #e6f0ff; padding: 25px; border-radius: 15px; border: 2px solid #99ccff;
                text-align: center; margin: auto; width: 70%; box-shadow: 3px 3px 10px rgba(0,0,0,0.1); color: #000000;'>
        <h2>Final Ensemble Prediction: 
            <span style='color: {"red" if final_class == "PNEUMONIA" else "green"};'>{final_class}</span></h2>
        <p style='font-size: 18px;'>🔴 PNEUMONIA Probability: <strong>{weighted_avg_prob * 100:.2f}%</strong></p>
        <p style='font-size: 18px;'>🟢 NORMAL Probability: <strong>{(1 - weighted_avg_prob) * 100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Please upload a Chest X-ray to begin prediction.")

    
    
        

  
