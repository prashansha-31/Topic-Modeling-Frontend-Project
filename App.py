import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# -------------------------
# Load Models Safely
# -------------------------
@st.cache_resource
def load_lda():
    try:
        with open("models/lda.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_nmf():
    try:
        with open("models/nmf.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_vectorizer():
    try:
        with open("models/vectorizer.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_tfidf():
    try:
        with open("models/tfidf.pkl", "rb") as f:
            return pickle.load(f)
    except:
        return None

@st.cache_resource
def load_bertopic():
    try:
        return BERTopic.load("models/bertopic_model")
    except:
        return None


# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Load All Models
# -------------------------
lda = load_lda()
nmf = load_nmf()
vectorizer = load_vectorizer()
tfidf = load_tfidf()
bertopic_model = load_bertopic()
embedder = load_embedder()

st.title("📘 Document Topic Classification – Streamlit App")
st.write("Enter any text below to predict the topic using LDA, NMF, or BERTopic.")

user_input = st.text_area("Enter text here:", height=200)

model_choice = st.selectbox(
    "Choose a model:",
    ["BERTopic", "LDA", "NMF"]
)

if st.button("Predict Topic"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        st.subheader(f"🔍 Using Model: {model_choice}")

        # -------------------------
        # BERTopic Prediction
        # -------------------------
        if model_choice == "BERTopic":
            embedding = embedder.encode([user_input])
            topic, prob = bertopic_model.transform([user_input])
            topic_info = bertopic_model.get_topic(topic[0])

            st.write(f"### ✅ Predicted Topic ID: **{topic[0]}**")
            st.write("### 🔑 Top Keywords:")
            st.write(pd.DataFrame(topic_info, columns=["Word", "Score"]))

        # -------------------------
        # LDA Prediction
        # -------------------------
        elif model_choice == "LDA":
            bow = vectorizer.transform([user_input])
            topic_distribution = lda.transform(bow)[0]
            top_topic = np.argmax(topic_distribution)

            st.write(f"### ✅ Predicted LDA Topic: **{top_topic}**")
            st.write("### 🔢 Topic Probability Distribution:")
            st.bar_chart(topic_distribution)

        # -------------------------
        # NMF Prediction
        # -------------------------
        elif model_choice == "NMF":
            tfidf_feats = tfidf.transform([user_input])
            topic_distribution = nmf.transform(tfidf_feats)[0]
            top_topic = np.argmax(topic_distribution)

            st.write(f"### ✅ Predicted NMF Topic: **{top_topic}**")
            st.write("### 🔢 Topic Probability Distribution:")
            st.bar_chart(topic_distribution)
