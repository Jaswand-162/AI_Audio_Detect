import os
import librosa
import numpy as np

AUDIO_DIR = "kaggle_before_after_dataset/AFTER_TRAIN/train/ai"

files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
audio_path = os.path.join(AUDIO_DIR, files[0])

audio, sr = librosa.load(audio_path, sr=16000)

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

# Convert MFCC (40, frames) → (40,)
mfcc_mean = np.mean(mfcc.T, axis=0)

print("MFCC mean shape:", mfcc_mean.shape)
print("First 10 MFCC mean values:")
print(mfcc_mean[:10])
