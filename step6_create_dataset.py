import os
import librosa
import numpy as np
import pandas as pd

BASE_DIR = "kaggle_before_after_dataset/AFTER_TRAIN/train"
FEATURES = []
LABELS = []

def extract_mfcc_mean(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

for label in ["ai", "human"]:
    folder = os.path.join(BASE_DIR, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            mfcc_mean = extract_mfcc_mean(path)
            FEATURES.append(mfcc_mean)
            LABELS.append(label)

# Create DataFrame
df = pd.DataFrame(FEATURES)
df["label"] = LABELS

# Save dataset
df.to_csv("train_features.csv", index=False)

print("Dataset created successfully ✅")
print("Total samples:", len(df))
