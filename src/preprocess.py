import pandas as pd
import re
import nltk
import spacy

from nltk.corpus import stopwords

nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 2
    ]
    return " ".join(tokens)

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df[["tweet_id", "text", "airline_sentiment"]]

    df["cleaned_text"] = df["text"].apply(clean_text)
    df[["tweet_id", "cleaned_text", "airline_sentiment"]].to_csv(
    output_path, index=False
)


if __name__ == "__main__":
    preprocess(
        "data/Tweets.csv",
        "output/preprocessed_data.csv"
    )
