# 🫁 Pneumonia Detection using Ensemble of Deep Learning Models

A Streamlit-based web application that uses an ensemble of three deep learning models — **Simple CNN**, **MobileNetV2**, and **EfficientNetB0** — to predict **Pneumonia** from Chest X-ray images. The app also includes **Grad-CAM visualizations** to explain model decisions.

---

## 🎯 Objective

To develop a reliable and explainable AI system that can:
- Automatically detect Pneumonia from chest X-ray images.
- Combine multiple deep learning models to improve prediction robustness.
- Visualize the decision-making process using Grad-CAM heatmaps.

---

## 🧠 Models Used

### ✅ Simple CNN
- A custom Convolutional Neural Network built from scratch.
- **Techniques:** Dropout, Data Augmentation, L2 Regularization, EarlyStopping.
- **Best Accuracy:** 94.20%

### ✅ MobileNetV2
- Pretrained MobileNetV2 with fine-tuning (last 10 layers unfrozen).
- **Techniques:** Transfer Learning, Data Augmentation, EarlyStopping.
- **Best Accuracy:** 95.22%

### ✅ EfficientNetB0
- Pretrained EfficientNetB0 with fine-tuning (last 20 layers unfrozen).
- **Techniques:** Transfer Learning, Data Augmentation, ReduceLROnPlateau.
- **Best Accuracy:** 94.71%

---

## ⚙️ Techniques Used

| Technique              | Purpose                                            |
|------------------------|-----------------------------------------------------|
| Data Augmentation      | Reduce overfitting, improve generalization          |
| Transfer Learning      | Use pretrained weights for faster convergence       |
| Fine-Tuning            | Unfreeze layers to specialize model on dataset      |
| Dropout & L2 Reg       | Combat overfitting in Simple CNN                    |
| EarlyStopping          | Stop training early to prevent overfitting          |
| ReduceLROnPlateau      | Adjust learning rate dynamically for better minima  |
| Ensemble (Weighted Avg)| Robust, accurate final prediction                   |

---

## 📊 Evaluation Metrics

| Model              | Class     | Precision | Recall | F1-Score | AUC    |
| ------------------ | --------- | --------- | ------ | -------- | ------ |
| **Simple CNN**     | NORMAL    | 0.95      | 0.94   | 0.94     |        |
|                    | PNEUMONIA | 0.94      | 0.95   | 0.94     | 0.9848 |
| **MobileNetV2**    | NORMAL    | 0.96      | 0.95   | 0.95     |        |
|                    | PNEUMONIA | 0.95      | 0.96   | 0.95     | 0.9925 |
| **EfficientNetB0** | NORMAL    | 0.94      | 0.96   | 0.95     |        |
|                    | PNEUMONIA | 0.95      | 0.94   | 0.95     | 0.9874 |


| Model          | Accuracy           | Precision   | Recall      | F1-Score    | AUC    | Notes                                                                                                           |
| -------------- | ------------------ | ----------- | ----------- | ----------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Simple CNN     | 94.20%             | 0.94        | 0.94        | 0.94        | 0.9848 | Overfitting initially, improved with L2 & dropout                                                               |
| MobileNetV2    | 95.22%             | 0.95        | 0.95        | 0.95        | 0.9925 | Best individual performer                                                                                       |
| EfficientNetB0 | 94.71%             | 0.95        | 0.95        | 0.95        | 0.9874 | Stable and efficient                                                                                            |
| **Ensemble**   | **\~95.7% (est.)** | 0.95 (est.) | 0.95 (est.) | 0.95 (est.) | \~0.99 | Ensemble improves generalization and robustness. Exact metrics depend on testing on combined prediction output. |


---

## Grad-CAM Visualization

The model highlights regions of the chest X-ray that contributed most to the final prediction using Grad-CAM, offering transparency and aiding clinical trust.

---

## 🚀 Deployment

This project is deployed using **Streamlit**, allowing users to:
- Upload X-ray images.
- View predictions from all 3 models and ensemble output.
- See class probabilities.
- Visualize Grad-CAM for each model.

---

## 🧪 How Ensemble Works

Each model predicts the probability of Pneumonia. The final prediction is a **weighted average** of all models:

\[
\text{Prediction} = 0.2 \times \text{SimpleCNN} + 0.3 \times \text{MobileNetV2} + 0.5 \times \text{EfficientNetB0}
\]

Weights were selected based on validation performance.

---

## 🔍 Screenshot

![App Screenshot](images/demo_screenshot.png)

