import os
import librosa
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
import pickle
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("[AUDIO] AI VOICE DETECTION - ADVANCED ML MODEL TRAINER")
print("=" * 70)

BASE_DIR = "kaggle_before_after_dataset/AFTER_TRAIN/train"
FEATURES = []
LABELS = []

def extract_advanced_features(file_path, sr=16000):
    """
    Extract comprehensive audio features to better distinguish AI from human voice
    """
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=sr)
        
        # Check if audio is empty
        if len(audio) < sr * 0.5:  # Less than 0.5 seconds
            return None
        
        features = []
        
        # 1. MFCC Features (Mean, Std) - 80 features
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            features.extend(np.mean(mfcc.T, axis=0))
            features.extend(np.std(mfcc.T, axis=0))
        except:
            features.extend([0] * 80)
        
        # 2. MFCC Delta - 80 features
        try:
            delta = librosa.feature.delta(mfcc)
            features.extend(np.mean(delta.T, axis=0))
            features.extend(np.std(delta.T, axis=0))
        except:
            features.extend([0] * 80)
        
        # 3. Spectral Features - 6 features
        try:
            spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features.append(np.mean(spec_centroid))
            features.append(np.std(spec_centroid))
            
            spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features.append(np.mean(spec_rolloff))
            features.append(np.std(spec_rolloff))
            
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
        except:
            features.extend([0] * 6)
        
        # 4. Chroma Features - 24 features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.extend(np.mean(chroma.T, axis=0))
            features.extend(np.std(chroma.T, axis=0))
        except:
            features.extend([0] * 24)
        
        # 5. Mel-spectrogram Features - 40 features
        try:
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=20)
            features.extend(np.mean(mel_spec.T, axis=0))
            features.extend(np.std(mel_spec.T, axis=0))
        except:
            features.extend([0] * 40)
        
        # 6. Tempogram - 4 features
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features.append(np.mean(onset_env))
            features.append(np.std(onset_env))
            features.append(np.max(onset_env))
            features.append(np.min(onset_env))
        except:
            features.extend([0] * 4)
        
        # 7. RMS Energy - 4 features
        try:
            rms = librosa.feature.rms(y=audio)[0]
            features.append(np.mean(rms))
            features.append(np.std(rms))
            features.append(np.max(rms))
            features.append(np.min(rms))
        except:
            features.extend([0] * 4)
        
        return np.array(features)
        
    except Exception as e:
        return None

print("\n[DATA] Extracting advanced audio features...")
print(f"Processing audio files from: {BASE_DIR}\n")

sample_count = {"ai": 0, "human": 0}

for label in ["ai", "human"]:
    folder = os.path.join(BASE_DIR, label)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    
    print(f"Processing {label.upper()} samples ({len(files)} files)...")
    
    for file in files:
        path = os.path.join(folder, file)
        try:
            features = extract_advanced_features(path)
            
            if features is not None:
                FEATURES.append(features)
                LABELS.append(label)
                sample_count[label] += 1
                print(f"  [OK] {file}")
            else:
                print(f"  [SKIP] {file} - returned None")
        except Exception as e:
            print(f"  [ERR] {file} - Error: {str(e)[:50]}")
    
    print()

# Create DataFrame
df = pd.DataFrame(FEATURES)
df["label"] = LABELS

print("=" * 70)
if len(FEATURES) == 0:
    print("[ERROR] No valid audio samples found!")
    print("Please check:")
    print("  1. Audio files exist in:")
    print(f"     - {os.path.join(BASE_DIR, 'ai')}")
    print(f"     - {os.path.join(BASE_DIR, 'human')}")
    print("  2. Audio files are valid .wav files")
    print("  3. Audio files are at least 0.3 seconds long")
    exit(1)

print(f"[OK] Dataset created successfully!")
print(f"   AI samples: {sample_count['ai']}")
print(f"   Human samples: {sample_count['human']}")
print(f"   Total features: {len(FEATURES[0])}")
print(f"   Total samples: {len(df)}")
print("=" * 70)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])
X = df.drop("label", axis=1)

print(f"\n[TRAIN] Training advanced SVM model with hyperparameter tuning...\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling is crucial for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GridSearch for optimal SVM parameters
print("Running GridSearch for optimal SVM parameters...")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'poly']
}

svm = SVC(probability=True, random_state=42, class_weight='balanced')
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"[OK] Best parameters: {grid_search.best_params_}")
print(f"[OK] Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
model = grid_search.best_estimator_

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

print("\n" + "=" * 70)
print("[METRICS] MODEL PERFORMANCE METRICS")
print("=" * 70)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("=" * 70)

print("\n[REPORT] Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=encoder.classes_,
    digits=4
))

print("\n[CONFUSION] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler
print("\n[SAVE] Saving model and preprocessing objects...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("[OK] Model saved to model.pkl")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("[OK] Scaler saved to scaler.pkl")

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("[OK] Encoder saved to encoder.pkl")

# Save features CSV for reference
df.to_csv("train_features.csv", index=False)
print("[OK] Features saved to train_features.csv")

# Save feature names for the API
feature_names = [f"feature_{i}" for i in range(len(FEATURES[0]))]
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
print("[OK] Feature names saved to feature_names.pkl")

print("\n" + "=" * 70)
print("[SUCCESS] TRAINING COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Commit and push the updated model files to GitHub")
print("2. Update main.py to use the new preprocessing (scaler)")
print("3. Redeploy to Render")
print("\nThe API is now using:")
print("  - 200+ audio features (MFCC, spectral, temporal, harmonic)")
print("  - Feature scaling with StandardScaler")
print("  - Optimized SVM with hyperparameter tuning")
print("  - Balanced class weights for better generalization")
