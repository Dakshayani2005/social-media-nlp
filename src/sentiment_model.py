import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_sentiment():
    df = pd.read_csv("output/preprocessed_data.csv")

    # 🔥 FIX: remove NaN / empty rows
    df = df.dropna(subset=["cleaned_text", "airline_sentiment"])
    df = df[df["cleaned_text"].str.strip() != ""]

    X = df["cleaned_text"]
    y = df["airline_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
        "f1_score_macro": f1_score(y_test, preds, average="macro"),
    }

    with open("output/sentiment_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save vectorizer & model
    joblib.dump(pipeline.named_steps["tfidf"], "output/tfidf_vectorizer.pkl")
    joblib.dump(pipeline.named_steps["clf"], "output/sentiment_model.pkl")

    # Save predictions with correct tweet IDs
    test_ids = df.loc[X_test.index, "tweet_id"]

    pd.DataFrame({
        "tweet_id": test_ids,
        "predicted_sentiment": preds
    }).to_csv("output/sentiment_predictions.csv", index=False)

if __name__ == "__main__":
    train_sentiment()
