from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI(title="Business Sentiment Analysis API")

# Enable CORS (helps frontend apps call your API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# Input format
class Review(BaseModel):
    text: str


# Root route
@app.get("/")
def home():
    """Redirects to the API docs."""
    return RedirectResponse(url="/docs")


# Prediction endpoint
@app.post("/predict")
def predict_sentiment(review: Review):
    """Predict sentiment from text."""
    text_vector = vectorizer.transform([review.text])
    prediction = model.predict(text_vector)[0]
    return {"sentiment": prediction}


# Run locally: uvicorn src.app:app --reload
