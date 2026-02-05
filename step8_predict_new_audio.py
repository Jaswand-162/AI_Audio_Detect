import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load trained dataset
df = pd.read_csv("train_features.csv")

X = df.drop("label", axis=1)
y = df["label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train model again (simple approach)
model = SVC(kernel="rbf")
model.fit(X, y)

def extract_mfcc_mean(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# 🔴 CHANGE THIS PATH to your test audio
test_audio_path = "test_audio.wav"

features = extract_mfcc_mean(test_audio_path)
features_df = pd.DataFrame(features, columns=X.columns)
prediction = model.predict(features_df)


if prediction[0] == 0:
    print("🔴 Prediction: AI Voice")
else:
    print("🟢 Prediction: Human Voice")
