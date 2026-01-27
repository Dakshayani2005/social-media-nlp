import streamlit as st
import json
import pandas as pd

st.set_page_config(layout="wide")
st.title("Social Media Sentiment & Topic Analysis")

with open("output/sentiment_metrics.json") as f:
    metrics = json.load(f)

st.subheader("Sentiment Metrics")
st.json(metrics)

st.subheader("Topics Discovered")
with open("output/topics.json") as f:
    topics = json.load(f)

for topic, words in topics.items():
    st.write(f"**{topic}**:", ", ".join(words))

st.subheader("LDA Visualization")
with open("output/lda_visualization.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)
