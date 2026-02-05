import os
import io
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
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

# Global variables for model and encoder
model = None
encoder = None
feature_columns = None

def load_model():
    """Load the trained model and encoder"""
    global model, encoder, feature_columns
    
    try:
        # Try to load pre-trained model if available
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded pre-trained model from model.pkl")
        else:
            # Train on the fly from features CSV if available
            if os.path.exists("train_features.csv"):
                df = pd.read_csv("train_features.csv")
                X = df.drop("label", axis=1)
                y = df["label"]
                
                feature_columns = X.columns.tolist()
                
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
                
                model = SVC(kernel="rbf", probability=True)
                model.fit(X, y)
                
                # Save model for future use
                with open("model.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                logger.info("Trained new model from train_features.csv")
            else:
                logger.warning("No model or training data found")
                return False
        
        # Load or create encoder
        if encoder is None and os.path.exists("train_features.csv"):
            df = pd.read_csv("train_features.csv")
            y = df["label"]
            encoder = LabelEncoder()
            encoder.fit_transform(y)
        
        if feature_columns is None and os.path.exists("train_features.csv"):
            df = pd.read_csv("train_features.csv")
            feature_columns = df.drop("label", axis=1).columns.tolist()
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def extract_mfcc_features(audio_data, sr=16000, n_mfcc=40):
    """Extract MFCC features from audio data"""
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        logger.error(f"Error extracting MFCC features: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up and loading model...")
    if not load_model():
        logger.warning("Could not load model on startup. Will attempt to load on first request.")

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
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Predict whether audio is AI or Human voice
    
    Returns:
    - prediction: "AI Voice" or "Human Voice"
    - confidence: probability score (0-1)
    - label_encoder_mapping: {"0": "AI Voice", "1": "Human Voice"}
    """
    try:
        # Ensure model is loaded
        if model is None:
            if not load_model():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Model not available. Please ensure train_features.csv exists."
                )
        
        # Read audio file
        contents = await file.read()
        
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Load audio from bytes
        audio, sr = librosa.load(io.BytesIO(contents), sr=16000)
        
        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio, sr=16000)
        
        # Create DataFrame with proper column names
        if feature_columns:
            features_df = pd.DataFrame([mfcc_features], columns=feature_columns)
        else:
            # Fallback: assume 40 MFCC coefficients
            features_df = pd.DataFrame([mfcc_features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get confidence scores if model supports probability
        if hasattr(model, 'decision_function'):
            confidence_scores = model.decision_function(features_df)[0]
            confidence = float(abs(confidence_scores))
        else:
            confidence = None
        
        # Map prediction to labels
        label_map = {0: "AI Voice", 1: "Human Voice"}
        predicted_label = label_map.get(int(prediction), "Unknown")
        
        logger.info(f"Prediction made: {predicted_label}")
        
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
                mfcc_features = extract_mfcc_features(audio, sr=16000)
                
                if feature_columns:
                    features_df = pd.DataFrame([mfcc_features], columns=feature_columns)
                else:
                    features_df = pd.DataFrame([mfcc_features])
                
                prediction = model.predict(features_df)[0]
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
