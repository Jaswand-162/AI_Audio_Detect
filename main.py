import os
import io
import base64
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from dotenv import load_dotenv
import pickle
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(
    title="Voice Detection API",
    description="API to detect whether audio is AI-generated or Human voice",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key setup - MUST be set via environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set. Please set it before starting the API.")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify the API key"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return api_key

# Pydantic model for JSON body requests
class AudioPredictionRequest(BaseModel):
    audio_base64: str
    language: str = "en"
    audio_format: str = "wav"

# Global variables for model and encoder
model = None
encoder = None
scaler = None
feature_columns = None

def load_model():
    """Load the trained model, encoder, and scaler"""
    global model, encoder, scaler, feature_columns
    
    try:
        # Try to load pre-trained model if available
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded pre-trained model from model.pkl")
        
        # Load scaler
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            logger.info("Loaded scaler from scaler.pkl")
        
        # Load encoder
        if os.path.exists("encoder.pkl"):
            with open("encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            logger.info("Loaded encoder from encoder.pkl")
        
        # Load feature columns for reference
        if feature_columns is None and os.path.exists("train_features.csv"):
            df = pd.read_csv("train_features.csv")
            feature_columns = df.drop("label", axis=1).columns.tolist()
        
        if model is not None and scaler is not None:
            return True
        else:
            logger.warning("Model or scaler not fully loaded")
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up and loading model...")
    if not load_model():
        logger.warning("Could not load model on startup. Will attempt to load on first request.")

def extract_advanced_features(audio_data, sr=16000):
    """
    Extract comprehensive audio features to distinguish AI from human voice
    Returns: numpy array of 200+ features
    """
    try:
        # Remove silence
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        non_silent_frames = np.where(np.mean(S_db, axis=0) > -40)[0]
        if len(non_silent_frames) > 0:
            audio_data = audio_data[non_silent_frames[0]*512:(non_silent_frames[-1]+1)*512]
        
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty after silence removal")
        
        features = []
        
        # 1. MFCC Features (Mean, Std) - 80 features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc.T, axis=0))
        features.extend(np.std(mfcc.T, axis=0))
        
        # 2. MFCC Delta (velocity) - 80 features
        delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta.T, axis=0))
        features.extend(np.std(delta.T, axis=0))
        
        # 3. Spectral Features - 20 features
        spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features.append(np.mean(spec_centroid))
        features.append(np.std(spec_centroid))
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 4. Chroma Features - 24 features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))
        
        # 5. Mel-spectrogram Features - 40 features
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=20)
        features.extend(np.mean(mel_spec.T, axis=0))
        features.extend(np.std(mel_spec.T, axis=0))
        
        # 6. Tempogram (Rhythm) - 4 features
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        features.append(np.mean(onset_env))
        features.append(np.std(onset_env))
        features.append(np.max(onset_env))
        features.append(np.min(onset_env))
        
        # 7. RMS Energy - 4 features
        rms = librosa.feature.rms(y=audio_data)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        features.append(np.max(rms))
        features.append(np.min(rms))
        
        return np.array(features)
    except Exception as e:
        logger.error(f"Error extracting advanced features: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Detection API",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict_voice(
    request: AudioPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict whether audio is AI or Human voice from base64-encoded audio data
    
    Request body:
    {
        "audio_base64": "base64_encoded_audio_string",
        "language": "en",
        "audio_format": "wav"
    }
    
    Returns:
    - prediction: "AI Voice" or "Human Voice"
    - confidence_score: probability score (0-1)
    - label_mapping: {"0": "AI Voice", "1": "Human Voice"}
    """
    try:
        # Ensure model is loaded
        if model is None:
            if not load_model():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model not available. Please ensure train_features.csv exists."
                )
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 encoding: {str(e)}"
            )
        
        if not audio_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio data"
            )
        
        # Load audio from bytes
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Extract advanced features
        advanced_features = extract_advanced_features(audio, sr=16000)
        
        # Scale features using the trained scaler
        if scaler is not None:
            features_scaled = scaler.transform([advanced_features])
        else:
            features_scaled = [advanced_features]
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probability scores for confidence
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = float(max(probabilities))
        else:
            confidence = None
        
        # Map prediction to labels
        label_map = {0: "AI Voice", 1: "Human Voice"}
        predicted_label = label_map.get(int(prediction), "Unknown")
        
        logger.info(f"Prediction made: {predicted_label} (confidence: {confidence})")
        
        return {
            "prediction": predicted_label,
            "confidence_score": confidence,
            "label_mapping": {
                "0": "AI Voice",
                "1": "Human Voice"
            },
            "raw_prediction": int(prediction)
        }
    
    except librosa.util.exceptions.LibrosaError as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid audio file: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_voice_batch(
    files: list[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Batch predict for multiple audio files
    """
    try:
        if model is None:
            if not load_model():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model not available"
                )
        
        results = []
        
        for file in files:
            try:
                contents = await file.read()
                
                if not contents:
                    results.append({
                        "filename": file.filename,
                        "error": "Empty file"
                    })
                    continue
                
                audio, sr = librosa.load(io.BytesIO(contents), sr=16000)
                advanced_features = extract_advanced_features(audio, sr=16000)
                
                # Scale features
                if scaler is not None:
                    features_scaled = scaler.transform([advanced_features])
                else:
                    features_scaled = [advanced_features]
                
                prediction = model.predict(features_scaled)[0]
                label_map = {0: "AI Voice", 1: "Human Voice"}
                predicted_label = label_map.get(int(prediction), "Unknown")
                
                results.append({
                    "filename": file.filename,
                    "prediction": predicted_label,
                    "raw_prediction": int(prediction)
                })
            
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during batch prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
