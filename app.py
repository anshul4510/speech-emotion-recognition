from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "speech_emotion_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "pleasant_surprise"
]

def extract_mfcc(file_path, n_mfcc=20):
    audio, sr = librosa.load(file_path, sr=16000, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.mean(axis=1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]
    file_path = "temp.wav"
    file.save(file_path)

    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)
    emotion = EMOTIONS[np.argmax(prediction)]

    os.remove(file_path)

    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)
