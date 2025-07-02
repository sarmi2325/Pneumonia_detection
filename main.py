import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import preprocess_image, make_gradcam_heatmap, superimpose_heatmap

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

st.set_page_config(page_title="Pneumonia Detection", layout="wide")

# Title and description
st.markdown(
    """
    <h1 style='text-align: center;'>Pneumonia Detection with Model Ensemble & Grad-CAM</h1>
    <p style='text-align: center; font-size: 18px;'>Upload a chest X-ray to view predictions from 3 models and visualize Grad-CAM attention.</p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Display images side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h4 >Original Chest X-ray</h4>", unsafe_allow_html=True)
        st.image(image.resize((400, 400)), use_container_width=False)
    
    with col2:
        st.markdown("<h4 >Grad-CAM (EfficientNetB0)</h4>", unsafe_allow_html=True)
        try:
            heatmap = make_gradcam_heatmap(img_array, model_effnet, last_conv_layer_name="top_conv")
            cam_image = superimpose_heatmap(heatmap, image)
            st.image(Image.fromarray(cam_image).resize((400, 400)), use_container_width=False)
        except Exception as e:
            st.error(f"Could not generate Grad-CAM: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Individual model predictions
    st.markdown("<h2 style='text-align: center;'>Individual Model Predictions</h2>", unsafe_allow_html=True)

    probs = []
    weights = []
    pred_cols = st.columns(3)
    
    # Display individual model predictions and calculate weights
    for (name, model), col in zip(models.items(), pred_cols):
        prob = model.predict(img_array, verbose=0)[0][0]
        probs.append(prob)
        
        confidence = max(prob, 1 - prob)  # Confidence weight
        weights.append(confidence)
    
        with col:
            st.markdown(
                f"""
                <div style='text-align: center; font-size: 20px;'>
                    <strong>{name}</strong><br>
                    <span style='color: red;'>PNEUMONIA: {prob * 100:.2f}%</span><br>
                    <span style='color: green;'>NORMAL: {(1 - prob) * 100:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Final Ensemble Prediction: Weighted Average using Confidence
    probs = np.array(probs)
    weights = np.array(weights)
    normalized_weights = weights / np.sum(weights)
    weighted_avg_prob = np.sum(probs * normalized_weights)
    final_class = "PNEUMONIA" if weighted_avg_prob > 0.5 else "NORMAL"
    
    st.markdown(
        f"""
        <div style='background-color: #f0f0f0; padding: 25px; border: 2px solid black; border-radius: 12px;
                    width: 60%; margin: 40px auto; text-align: center;'>
            <h2 style='color: black;'>Final Ensemble Prediction: 
            <span style='color: {"red" if final_class == "PNEUMONIA" else "green"};'>{final_class}</span></h2>
            <p style='font-size: 18px; color: red;'>PNEUMONIA Probability: {weighted_avg_prob * 100:.2f}%</p>
            <p style='font-size: 18px; color: green;'>NORMAL Probability: {(1 - weighted_avg_prob) * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
        

  
