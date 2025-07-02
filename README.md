#  Pneumonia Detection using Ensemble of Deep Learning Models

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


Prediction = Dynamic Confidence-Based Voting to avoid misleading prediction

Weights were selected based on validation performance.

---

## Samples Test

| Sample | Ground Truth | Predicted | **Simple CNN** <br> Pneumonia / Normal | **MobileNetV2** <br> Pneumonia / Normal | **EfficientNetB0** <br> Pneumonia / Normal | **Final Ensemble** | Pneumonia / Normal |
| ------ | ------------ | --------- | -------------------------------------- | --------------------------------------- | ------------------------------------------ | ------------------ | ------------------ |
| 1      | NORMAL       | NORMAL    | 3.69% / 96.31%                         | 17.02% / 82.98%                         | 10.13% / 89.87%                            | **NORMAL**         | 10.28% / 89.72%    |
| 2      | NORMAL       | NORMAL    | 0.10% / 99.90%                         | 0.17% / 99.83%                          | 0.31% / 99.69%                             | **NORMAL**         | 0.19% / 99.81%     |
| 3      | NORMAL       | NORMAL    | 7.64% / 92.36%                         | 9.99% / 90.01%                          | 3.89% / 96.11%                             | **NORMAL**         | 7.17% / 92.83%     |
| 4      | NORMAL       | NORMAL    | 13.42% / 86.58%                        | 0.01% / 99.99%                          | 0.28% / 99.72%                             | **NORMAL**         | 4.57% / 95.43%     |
| 5      | NORMAL       | NORMAL    | 23.50% / 76.50%                        | 0.17% / 99.83%                          | 0.10% / 99.90%                             | **NORMAL**         | 7.92% / 92.08%     |
| 6      | PNEUMONIA    | PNEUMONIA | 99.83% / 0.17%                         | 99.77% / 0.23%                          | 99.71% / 0.29%                             | **PNEUMONIA**      | 99.77% / 0.23%     |
| 7      | PNEUMONIA    | PNEUMONIA | 99.30% / 0.70%                         | 99.94% / 0.06%                          | 96.42% / 3.58%                             | **PNEUMONIA**      | 98.55% / 1.45%     |
| 8      | PNEUMONIA    | PNEUMONIA | 70.51% / 29.49%                        | 95.77% / 4.23%                          | 96.86% / 3.14%                             | **PNEUMONIA**      | 87.71% / 12.29%    |
| 9      | PNEUMONIA    | PNEUMONIA | 61.57% / 38.43%                        | 92.21% / 7.79%                          | 90.54% / 9.46%                             | **PNEUMONIA**      | 81.44% / 18.56%    |
| 10     | PNEUMONIA    | PNEUMONIA | 91.46% / 8.54%                         | 95.06% / 4.94%                          | 95.84% / 4.16%                             | **PNEUMONIA**      | 94.12% / 5.88%     |


## 🔍 Demo

You can check out the live deployed app here:[Demo](https://pneumoniadetection-er8mvnqy9s9ejsr2g6wqjk.streamlit.app/)


