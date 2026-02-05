import os
import librosa

AUDIO_DIR = "kaggle_before_after_dataset/AFTER_TRAIN/train/ai"


files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

print("Total audio files found:", len(files))

audio_path = os.path.join(AUDIO_DIR, files[0])

audio, sr = librosa.load(audio_path, sr=16000)

print("Loaded file:", files[0])
print("Sample rate:", sr)
print("Audio length:", len(audio))
