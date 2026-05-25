import streamlit as st
import torch
from transformers import pipeline

# Load sentiment analysis pipeline (your fine-tuned model)
@st.cache_resource
def load_sentiment():
    return pipeline("text-classification", model="EBSQ/amazon-sentiment-distilbert")

# Load text generation pipeline (small GPT-2 model)
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

# Predefined completion suffixes
POSITIVE_SUFFIX = " We are happy you liked it. Thank you for shopping with us!"
NEGATIVE_SUFFIX = " We apologize for the inconvenience. Please contact our support team for assistance."

st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and a generated reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing sentiment..."):
            sentiment = load_sentiment()
            result = sentiment(review, truncation=True, max_length=512)[0]
            label = result['label']  # e.g., LABEL_0 or LABEL_1
            score = result['score']
            sentiment_label = "Negative" if "LABEL_0" in label else "Positive"
            st.info(f"**Sentiment:** {sentiment_label} (confidence: {score:.2f})")

            # Generate a short response using distilgpt2
            generator = load_generator()
            if "LABEL_0" in label:
                prompt = "Sorry"
            else:
                prompt = "Thank you"
            # Generate a few tokens
            gen_output = generator(prompt, max_new_tokens=5, do_sample=True, temperature=0.7, pad_token_id=50256)
            short_reply = gen_output[0]['generated_text'].strip()
            # Combine with suffix
            if "LABEL_0" in label:
                full_reply = short_reply + NEGATIVE_SUFFIX
            else:
                full_reply = short_reply + POSITIVE_SUFFIX
            st.success(f"**Suggested Reply:** {full_reply}")
    else:
        st.warning("Please enter a review.")
