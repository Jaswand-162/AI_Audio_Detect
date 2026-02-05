import os
import librosa
import numpy as np

# Path to AI audio folder
AUDIO_DIR = "kaggle_before_after_dataset/AFTER_TRAIN/train/ai"

# Pick one audio file
files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
audio_path = os.path.join(AUDIO_DIR, files[0])

# Load audio
audio, sr = librosa.load(audio_path, sr=16000)

# Extract MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

# Print details
print("Audio file:", files[0])
print("MFCC shape:", mfcc.shape)
print("First 5 MFCC values of first frame:")
print(mfcc[:, 0][:5])
