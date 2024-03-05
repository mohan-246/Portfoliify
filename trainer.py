import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to extract audio features
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')) , tf.config.list_logical_devices('GPU'))
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
    for device in physical_devices:
        print(f"Using GPU: {device}")
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("No GPU devices available. Using CPU.")

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=22050).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=22050))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=22050))
        roll_off = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=22050))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13).T, axis=0)

        return np.concatenate([chroma_stft, [spectral_centroid, spectral_bandwidth, roll_off, zero_crossing_rate], mfccs])
    except Exception as e:
        print(f"Error encountered while processing file: {file_path}")
        print(f"Error details: {str(e)}")
        return None

# Load and preprocess the data
ambulance_path = "./Ambulance data"  # Replace with the actual path
street_path = "./Road Noises"  # Replace with the actual path

ambulance_features = []
street_features = []
labels = []

# Process ambulance sounds
for filename in os.listdir(ambulance_path):
   if filename.endswith('.wav'):
        file_path = os.path.join(ambulance_path, filename)
        print(f"Processing file: {file_path}")
        features = extract_features(file_path)
        if features is not None:
            ambulance_features.append(features)
            labels.append(1)  # Label 1 for ambulance sounds

# Process street sounds
for filename in os.listdir(street_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(street_path, filename)
        print(f"Processing file: {file_path}")
        features = extract_features(file_path)
        if features is not None:
            street_features.append(features)
            labels.append(0)  # Label 0 for street sounds

# Combine features and labels
            
X = np.vstack((ambulance_features, street_features))
y = np.array(labels)

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the MLP model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with 1 node for binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy * 100:.2f}%")
