import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components

st.set_page_config(page_title="Social Media NLP Dashboard", layout="wide")

st.title("📊 Social Media Sentiment & Topic Dashboard")

# ---- Load files ----
sentiment_df = pd.read_csv(
    "output/sentiment_predictions.csv",
    names=["id", "sentiment"]
)


with open("output/topics.json") as f:
    topics = json.load(f)

# ---- Topic Selector ----
st.sidebar.header("🔍 Topic Explorer")
topic_id = st.sidebar.selectbox(
    "Select a Topic",
    list(topics.keys())
)

st.subheader(f"🧠 Topic {topic_id}: What people are talking about")

st.write("**Top Keywords:**")
st.write(", ".join(topics[topic_id]))

# ---- Sentiment Distribution ----
st.subheader("😊 Sentiment Distribution")

sentiment_counts = sentiment_df["sentiment"].value_counts()
st.bar_chart(sentiment_counts)

# ---- LDA Visualization ----
st.subheader("🌀 Topic Visualization (LDA)")
with open("output/lda_visualization.html", "r", encoding="utf-8") as f:
    html_data = f.read()

components.html(html_data, height=800, scrolling=True)
