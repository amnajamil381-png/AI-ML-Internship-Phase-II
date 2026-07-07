import streamlit as st
from transformers import pipeline

# -----------------------
# Load Model
# -----------------------

classifier = pipeline(
    "text-classification",
    model="bert_news_classifier",
    tokenizer="bert_news_classifier"
)

labels = {
    "LABEL_0": "🌍 World",
    "LABEL_1": "🏆 Sports",
    "LABEL_2": "💼 Business",
    "LABEL_3": "💻 Sci/Tech"
}

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(
    page_title="News Classifier",
    page_icon="📰"
)

st.title("📰 News Category Classifier")

st.write(
    """
This application classifies a news headline into one of four categories using
a fine-tuned BERT model.

Categories:

- 🌍 World
- 🏆 Sports
- 💼 Business
- 💻 Sci/Tech
"""
)

headline = st.text_area(
    "Enter a News Headline",
    height=120
)

if st.button("Predict Category"):

    if headline.strip() == "":
        st.warning("Please enter a news headline.")
    else:

        prediction = classifier(headline)[0]

        category = labels[prediction["label"]]
        confidence = prediction["score"]

        st.success(f"Predicted Category: **{category}**")

        st.write(f"Confidence Score: **{confidence:.2%}**")
