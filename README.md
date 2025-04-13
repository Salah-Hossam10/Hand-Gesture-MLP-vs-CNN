# 🖐️ Hand Gesture Recognition: MLP vs CNN

A comparative study between **Multilayer Perceptron (MLP)** using hand landmarks and a **Convolutional Neural Network (CNN)** using raw gesture images for hand gesture classification.

---

## 📌 Overview

This project aims to evaluate two different approaches for recognizing static hand gestures:

1. **MLP with MediaPipe Landmarks** – Uses 3D hand landmarks extracted via [MediaPipe](https://google.github.io/mediapipe/solutions/hands).
2. **CNN on Raw Images** – Trains a deep convolutional network on raw gesture images.

---

## 🧠 Models & Techniques

### 1. 🔷 MLP (Using MediaPipe Landmarks)
- **Input:** 21 hand landmarks (x, y, z) → 63 features
- **Framework:** scikit-learn / TensorFlow (dense layers)
- **Advantages:** Lightweight, fast, and interpretable
- **Dependencies:** Accurate landmark extraction via MediaPipe

### 2. 🔶 CNN (Using Raw Images)
- **Input:** Preprocessed hand gesture images
- **Architecture:** 3 Conv layers → MaxPooling → Flatten → Dense layers
- **Advantages:** Learns spatial features directly from pixels
- **Challenges:** Requires larger datasets and compute resources

---

## 📁 Project Structure

📦 Hand-Gesture-Recognition ├── cnn_image_model/ # CNN-based pipeline │ ├── preprocess_images.py │ ├── train_cnn.py ├── mediapipe_landmark_model/ # MLP-based pipeline │ ├── extract_landmarks.py │ ├── train_mlp.py ├── dataset/ │ ├── images/ │ ├── landmarks.csv ├── results/ │ ├── mlp_accuracy.png │ ├── cnn_accuracy.png ├── requirements.txt └── README.md

yaml
Copy
Edit

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **MLP** | 92% | 91% | 92% | 91% |
| **CNN** | 96% | 95% | 96% | 95% |

✅ **CNN outperforms MLP**, but MLP still provides strong results with faster training and less compute.

---

## 🚀 Getting Started

### Clone the repo
```bash
git clone https://github.com/yourusername/hand-gesture-mlp-vs-cnn.git
cd hand-gesture-mlp-vs-cnn
Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
▶️ How to Run
Train MLP model (using landmarks)
bash
Copy
Edit
cd mediapipe_landmark_model
python train_mlp.py
Train CNN model (using images)
bash
Copy
Edit
cd cnn_image_model
python train_cnn.py
📂 Dataset
You can use:

Your own webcam-captured dataset

Public datasets like:

Rock-Paper-Scissors (Kaggle)

Hand Gesture Recognition Database (UCI)

📌 Future Improvements
⏺️ Real-time gesture prediction

🧠 LSTM for dynamic gesture sequences

🔁 Data augmentation for robustness

🤝 Integrate with sign language recognition systems

👨‍💻 Author
Salah Hossam
Machine Learning & Embedded Systems Engineer
