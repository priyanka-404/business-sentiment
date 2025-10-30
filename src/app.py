from fastapi import FastAPI, Form
import joblib

app = FastAPI(title="Business Sentiment Analysis API", description="Predict sentiment of business reviews", version="1.0")

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Business Sentiment API! Visit /docs for Swagger UI."}

@app.post("/predict")
def predict_sentiment(text: str = Form(...)):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return {"sentiment": prediction}
