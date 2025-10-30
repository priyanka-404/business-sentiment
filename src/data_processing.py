import pandas as pd
import os

def main():
    # Step 1: Load the dataset safely
    # quoting=3 ignores quotes inside text fields (important for reviews)
    df = pd.read_csv(
        'data/raw/Reviews.csv',
        low_memory=False,
        quoting=3,
        on_bad_lines='skip',
        encoding='utf-8'
    )

    print("✅ File loaded successfully!")
    print("Shape:", df.shape)
    print("Columns detected:", df.columns.tolist())

    # Step 2: Keep only useful columns
    if 'Text' not in df.columns or 'Score' not in df.columns:
        raise KeyError("❌ Columns 'Text' or 'Score' not found. Check dataset structure!")

    df = df[['Text', 'Score']].dropna()

    # Step 3: Convert numeric scores to sentiment categories
    def score_to_sentiment(score):
        if score >= 4:
            return 'positive'
        elif score == 3:
            return 'neutral'
        else:
            return 'negative'

    df['Sentiment'] = df['Score'].apply(score_to_sentiment)

    # Step 4: Create output folder if not exists
    os.makedirs('data/processed', exist_ok=True)

    # Step 5: Save the processed dataset
    output_path = 'data/processed/processed_reviews.csv'
    df.to_csv(output_path, index=False)

    print("✅ Processed data saved at:", output_path)
    print("✅ New shape:", df.shape)
    print("✅ Sample data:\n", df.head())

if __name__ == "__main__":
    main()
