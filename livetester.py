import tkinter as tk
import os
from os import path
import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model

def extract_features(audio):
    try:
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=22050).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=22050))
        roll_off = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=22050))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13).T, axis=0)

        return np.concatenate([chroma_stft, [spectral_centroid, spectral_bandwidth, roll_off, zero_crossing_rate], mfccs])
    except Exception as e:
        print(f"Error encountered while processing audio")
        print(f"Error details: {str(e)}")
        return None

def process_audio(indata, frames, time, status):
    global model
    global label

    input_features = extract_features(indata.flatten())

    if input_features is not None:
        input_features = input_features.reshape(1, -1)
        prediction = model.predict(input_features)
        
        predicted_label = 1 if prediction > 0.5 else 0
        if predicted_label == 1:
            label.config(bg='green')
        else:
            label.config(bg='red')

current_directory = os.getcwd()
model_file_relative_path = path.join('models', 'trained_model.keras')

model_file_path = path.join(current_directory, model_file_relative_path)
if os.path.exists(model_file_path):
    print(f"Model file '{model_file_path}' exists.")
    try:
        model = load_model(model_file_path)
        print("Model loaded successfully.")
        
        print("Model summary:")
        model.summary()
        
    except Exception as e:
        print(f"Error encountered while loading the model: {e}")
    
else:
    print(f"Model file '{model_file_path}' does not exist.")

window = tk.Tk()
window.title("Live Audio Detection")
label = tk.Label(window, text="Detection Result", width=20, height=10)
label.pack()

with sd.InputStream(callback=process_audio):
    window.mainloop()
