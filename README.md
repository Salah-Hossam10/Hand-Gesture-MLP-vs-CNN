🖐️ Hand Gesture Recognition: MLP vs CNN
This project compares the performance of Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN) models in classifying hand gestures. It explores two approaches:

Using MediaPipe to extract 3D hand landmarks for training traditional ML models (MLP).

Training a CNN directly on raw hand gesture images.

🔍 Objective
To evaluate and compare:

The effectiveness of feature-based learning (using hand landmarks with MLP).

The power of image-based deep learning (using CNN).

🗂️ Project Structure
graphql
Copy
Edit
📁 Hand-Gesture-Recognition
├── 📁 mediapipe_landmark_model  # MLP pipeline using landmarks
│   ├── extract_landmarks.py
│   ├── train_mlp.py
├── 📁 cnn_image_model           # CNN pipeline using images
│   ├── preprocess_images.py
│   ├── train_cnn.py
├── 📁 dataset
│   ├── images/
│   ├── landmarks.csv
├── 📊 results
│   ├── mlp_accuracy.png
│   ├── cnn_accuracy.png
├── requirements.txt
└── README.md
🧠 Models
1. MLP (Multilayer Perceptron)
Input: 21 hand landmarks (x, y, z coordinates) extracted using MediaPipe Hands.

Architecture: 3 Dense layers with ReLU and dropout.

Pros: Lightweight and fast inference.

Cons: Dependent on accurate landmark detection.

2. CNN (Convolutional Neural Network)
Input: Preprocessed grayscale/colored images of hand gestures.

Architecture: 3 Conv2D layers, MaxPooling, Flatten, and Dense layers.

Pros: Learns features automatically from raw data.

Cons: Requires more compute and training time.

📈 Results
Model	Accuracy	Precision	Recall	F1 Score
MLP	92%	91%	92%	91%
CNN	96%	95%	96%	95%
✅ CNN outperformed MLP, but the MLP model still achieved competitive results with significantly lower computational requirements.

📦 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/hand-gesture-mlp-vs-cnn.git
cd hand-gesture-mlp-vs-cnn
pip install -r requirements.txt
▶️ How to Run
Train MLP:
bash
Copy
Edit
cd mediapipe_landmark_model
python train_mlp.py
Train CNN:
bash
Copy
Edit
cd cnn_image_model
python train_cnn.py
🗃️ Dataset
The dataset contains labeled images of different hand gestures (e.g., rock, paper, scissors, thumbs up).

You can use your own dataset or a public one like Rock Paper Scissors or collect data using webcam + MediaPipe.

💡 Future Work
Extend to dynamic gesture recognition (video).

Integrate with real-time applications (e.g., sign language interpreter).

Add gesture augmentation and noise resilience techniques.

👨‍💻 Author
Salah Hossam
Machine Learning & Embedded Systems Engineer
