from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging

# Initialize app
app = FastAPI(title="Sentiment Analysis API")

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load model safely
try:
    model = joblib.load("models/sentiment_model.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None


# Request schema
class NewsRequest(BaseModel):
    text: str


# Root endpoint
@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


# Health check endpoint
@app.get("/health")
def health():
    if model:
        return {"status": "healthy"}
    return {"status": "model not loaded"}


# Prediction endpoint
@app.post("/predict")
def predict(request: NewsRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        logging.info(f"Input text: {request.text}")

        prediction = model.predict([request.text])[0]

        logging.info(f"Prediction: {prediction}")

        return {
            "input": request.text,
            "sentiment": prediction
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))