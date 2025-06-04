# 🧴 Skin Disease Prediction App

A deep learning-based web application for predicting skin diseases using image classification. Built using **TensorFlow/Keras** and deployed with **Streamlit**, this project aims to assist in the early detection of skin-related diseases through a user-friendly interface.

![Skin Disease Prediction Banner](https://img.shields.io/badge/DeepLearning-TensorFlow-brightgreen?style=flat-square)
![Skin Disease Dataset](https://img.shields.io/badge/Dataset-HAM10000-blueviolet?style=flat-square)
![WebApp](https://img.shields.io/badge/UI-Streamlit-orange?style=flat-square)

---

## 🔍 Project Overview

This project leverages convolutional neural networks (CNNs) to classify images of skin lesions into different disease categories. The model is trained on the **HAM10000 dataset** from Kaggle, which contains 10,000+ dermatoscopic images across 7 types of skin diseases.

### 🔬 Skin Disease Classes
- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)
- Melanoma (mel)

---

## 📂 Dataset

- **Name**: [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download)
- **Format**: JPEG images with CSV metadata
- **Size**: ~2GB
- **Source**: Kaggle

---

## ⚙️ Tech Stack

| Component | Description |
|----------|-------------|
| `Streamlit` | Web app framework for interactive UI |
| `TensorFlow / Keras` | Model building, training, and prediction |
| `CNN` | Convolutional Neural Network for image classification |
| `PIL & NumPy` | Image loading, resizing, and preprocessing |
| `JSON` | Stores user-uploaded data and predicted output |
| `Matplotlib` | Data visualization (optional use) |

---

## 🚀 Features

- Upload your own skin lesion image
- Real-time disease prediction using a trained CNN model
- Disease class name with confidence score
- Clean, responsive web UI using Streamlit
- Model trained on high-quality dermatological data

---

## 🛠️ Installation & Setup

### 🔧 Prerequisites

- Python 3.7+
- pip or conda

### 📦 Clone and Install

git clone https://github.com/deepakgit-1/Skin_Disease_Prediction_App.git
cd Skin_Disease_Prediction_App
pip install -r requirements.txt

## ▶️ Run the App
streamlit run app.py

## 🧠 Model Architecture
- The model uses a custom CNN architecture, which includes:
- Convolutional layers
-Batch normalization
- MaxPooling
- Dropout for regularization
- Dense layers for classification

## ✅ Training
- Dataset preprocessed (resized to 64x64 or 128x128)
- Images normalized and augmented
- Model trained using categorical cross-entropy loss
- Metrics: Accuracy and Loss

 ## 📁 Project Structure

Skin_Disease_Prediction_App/
├── app.py                 # Streamlit app main script
├── model/                 # Directory for trained CNN model (.h5 files)
│   └── skin_disease_model.h5
├── data/                  # Directory for sample images or preprocessing scripts
│   └── sample_image.jpg
├── utils/                 # Utility functions (preprocessing, prediction helpers)
│   └── helper.py
├── disease_info.json      # JSON file with skin disease details
├── requirements.txt       # List of required Python libraries
└── README.md              # Project documentation



## 📜 License
This project is open-source and available under the MIT License.

