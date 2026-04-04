from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import os
import pandas as pd
from datetime import datetime



app = FastAPI(title="Sentiment Analysis API")

logging.basicConfig(level=logging.INFO)

LOG_FILE = "logs/predictions.csv"
os.makedirs("logs", exist_ok=True)

# =========================
# LOAD MODEL
# =========================

try:
    model = joblib.load("models/sentiment_model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None


# =========================
# REQUEST SCHEMA
# =========================

class NewsRequest(BaseModel):
    text: str


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


@app.get("/health")
def health():
    if model:
        return {"status": "healthy"}
    return {"status": "model not loaded"}


@app.post("/predict")
def predict(request: NewsRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        logging.info(f"Input text: {request.text}")

        prediction = model.predict([request.text])[0]

        logging.info(f"Prediction: {prediction}")

        # =========================
        # LOG PREDICTIONS (MONITORING)
        # =========================

        log_entry = pd.DataFrame([{
            "text": request.text,
            "prediction": prediction,
            "timestamp": datetime.now()
        }])

        if os.path.exists(LOG_FILE):
            log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            log_entry.to_csv(LOG_FILE, index=False)

        return {
            "input": request.text,
            "sentiment": prediction
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))