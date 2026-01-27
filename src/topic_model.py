import pandas as pd
import json
import joblib
import pyLDAvis
import pyLDAvis.lda_model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def train_lda():
    df = pd.read_csv("output/preprocessed_data.csv")

    # 🔥 FIX: remove NaN / empty texts
    df = df.dropna(subset=["cleaned_text"])
    df = df[df["cleaned_text"].str.strip() != ""]

    vectorizer = CountVectorizer(max_df=0.9, min_df=10)
    X = vectorizer.fit_transform(df["cleaned_text"])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)

    joblib.dump(lda, "output/lda_model.pkl")

    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics[f"topic_{idx}"] = top_words

    with open("output/topics.json", "w") as f:
        json.dump(topics, f, indent=2)

    vis = pyLDAvis.lda_model.prepare(
        lda, X, vectorizer
    )
    pyLDAvis.save_html(vis, "output/lda_visualization.html")

if __name__ == "__main__":
    train_lda()
