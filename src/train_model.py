import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

def main():
    # Load processed data
    df = pd.read_csv('data/processed/processed_reviews.csv')
    print("✅ Data loaded successfully for training!")
    print(df.head())

    # Features and labels
    X = df['Text']
    y = df['Sentiment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    print("\n✅ Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Save model and vectorizer
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

    print("\n✅ Model and vectorizer saved in 'models/'")

if __name__ == "__main__":
    main()
