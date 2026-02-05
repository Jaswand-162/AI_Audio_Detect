# Voice Detection API - Deployment Guide

This guide explains how to deploy the Voice Detection API on Render for production use.

## Prerequisites

- GitHub account with this repository pushed to it
- Render account (free tier available at https://render.com)
- Your trained model (`train_features.csv` or `model.pkl`)

## Local Testing

Before deploying, test locally:

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and change API_KEY to your secret key
   ```

3. **Run the API**
   ```bash
   python main.py
   ```

4. **Test the API**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Predict with API key
   curl -X POST http://localhost:8000/predict \
     -H "X-API-Key: your-secret-api-key-here-change-this-in-production" \
     -F "file=@path/to/your/audio.wav"
   ```

5. **View API documentation**
   - Open http://localhost:8000/docs (Swagger UI)
   - Or http://localhost:8000/redoc (ReDoc)

## Deploy to Render

### Step 1: Prepare Your Repository

1. Make sure all files are committed to GitHub:
   ```bash
   git add .
   git commit -m "Add FastAPI voice detection API"
   git push origin main
   ```

2. Ensure `train_features.csv` is in the repository root (or update the path in main.py)

### Step 2: Create Render Service

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Fill in the details:
   - **Name**: `voice-detection-api` (or your preferred name)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: 
     ```
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```

### Step 3: Add Environment Variables

1. In Render dashboard, go to your service
2. Click "Environment" tab
3. Add the following environment variables:
   - `API_KEY`: Set to your secret API key (e.g., `your-super-secret-key-12345`)
   - `PORT`: Set to `8000` (Render will override this automatically)

### Step 4: Deploy

1. Click "Deploy" button
2. Wait for the build to complete (usually 2-5 minutes)
3. Once deployed, your API URL will be displayed (e.g., `https://voice-detection-api.onrender.com`)

## API Usage

### Authentication
All requests require the `X-API-Key` header:
```
X-API-Key: your-secret-api-key
```

### Endpoints

#### 1. Health Check
```bash
curl https://voice-detection-api.onrender.com/health \
  -H "X-API-Key: your-secret-api-key"
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Single Prediction
```bash
curl -X POST https://voice-detection-api.onrender.com/predict \
  -H "X-API-Key: your-secret-api-key" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "prediction": "Human Voice",
  "confidence_score": 2.45,
  "label_mapping": {
    "0": "AI Voice",
    "1": "Human Voice"
  },
  "raw_prediction": 1
}
```

#### 3. Batch Prediction
```bash
curl -X POST https://voice-detection-api.onrender.com/predict/batch \
  -H "X-API-Key: your-secret-api-key" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "audio1.wav",
      "prediction": "Human Voice",
      "raw_prediction": 1
    },
    {
      "filename": "audio2.wav",
      "prediction": "AI Voice",
      "raw_prediction": 0
    }
  ]
}
```

## API Documentation

Once deployed, access interactive API documentation at:
- Swagger UI: `https://voice-detection-api.onrender.com/docs`
- ReDoc: `https://voice-detection-api.onrender.com/redoc`

## Changing Your API Key

To change your API key on Render:

1. Go to your service on Render dashboard
2. Click "Environment"
3. Modify the `API_KEY` value
4. Render will automatically redeploy with the new key

## Monitoring

- Check logs in Render dashboard under the service
- Logs show prediction results and any errors
- Free tier may have some cold start delays

## Limits

- **File size**: Up to 100MB per request (can be modified in code)
- **Requests**: Depends on your Render plan
- **Free tier**: Limited to 750 hours/month, may sleep after 15 minutes of inactivity

## Troubleshooting

### Model not loading
- Ensure `train_features.csv` is in the repository
- Check Render logs for errors
- Verify the CSV file format matches the training script output

### API Key not working
- Make sure you're using the exact key set in environment variables
- Check header format: `X-API-Key: your-key`
- Verify Render deployment has completed

### Slow responses
- First request may be slow due to cold start
- Upgrade from Render free tier for better performance

## Security Tips

1. **Keep API key secret**: Never commit it to GitHub
2. **Use environment variables**: Store sensitive data in Render environment
3. **API key rotation**: Change your key periodically
4. **HTTPS**: Render provides HTTPS by default
5. **CORS**: Configured to accept requests from any origin (modify if needed)

## Example Usage with Python

```python
import requests

API_URL = "https://voice-detection-api.onrender.com"
API_KEY = "your-secret-api-key"

def predict_audio(file_path):
    headers = {"X-API-Key": API_KEY}
    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}
        response = requests.post(
            f"{API_URL}/predict",
            headers=headers,
            files=files
        )
    return response.json()

result = predict_audio("test_audio.wav")
print(result["prediction"])
```

## Example Usage with JavaScript

```javascript
const API_URL = "https://voice-detection-api.onrender.com";
const API_KEY = "your-secret-api-key";

async function predictAudio(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY
    },
    body: formData
  });

  return await response.json();
}

// Usage
const audioFile = document.getElementById("audioInput").files[0];
predictAudio(audioFile).then(result => {
  console.log(result.prediction); // "AI Voice" or "Human Voice"
});
```

---

For more information, visit:
- Render Documentation: https://render.com/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
