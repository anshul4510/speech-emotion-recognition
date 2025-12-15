# 🎧 Speech Emotion Recognition using MFCC & Deep Learning

## 📌 Project Overview
This project implements a **Speech Emotion Recognition (SER)** system that classifies human emotions from speech audio signals. The system uses **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and a **Deep Learning model (CNN / ANN)** for emotion classification.

The model is trained on a structured speech dataset containing multiple emotional categories spoken by different speakers.

---

## 🎯 Objectives
- Load and preprocess speech audio files
- Extract MFCC features from audio
- Normalize and merge inconsistent emotion labels
- Train a deep learning model for emotion classification
- Save the trained model and training metrics
- Evaluate model performance using accuracy and loss

---

## 🗂️ Dataset Structure
drive/My Drive/SpeechRecognition/kaggle/input/
├── OAF_angry/
├── OAF_disgust/
├── OAF_fear/
├── OAF_happy/
├── OAF_neutral/
├── OAF_Pleasant_surprise/
├── OAF_Sad/
├── YAF_angry/
├── YAF_disgust/
├── YAF_fear/
├── YAF_happy/
├── YAF_neutral/
├── YAF_pleasant_surprised/
├── YAF_sad/


> Note: All variants of *pleasant surprise* are merged into a single label:  
**pleasant_surprise**

---

## 🧠 Emotion Classes
The final normalized emotion classes used for training are:
angry
disgust
fear
happy
neutral
sad
pleasant_surprise


---

## ⚙️ Technologies Used
- Python
- NumPy
- Pandas
- Librosa
- TensorFlow / Keras
- Scikit-learn

---
