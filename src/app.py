from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Initialize FastAPI
app = FastAPI(title="Business Sentiment Analysis API")

# Define input schema
class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: ReviewInput):
    text_vector = vectorizer.transform([review.text])
    prediction = model.predict(text_vector)[0]
    return {"sentiment": prediction}
