import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import random

# Cache sentiment model
@st.cache_resource
def load_sentiment_model():
    model_name = "EBSQ/amazon-sentiment-distilbert"  # replace with your HF username
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Cache T5 text generation pipeline
@st.cache_resource
def load_t5_pipeline():
    # Use text2text-generation pipeline with t5-small
    return pipeline("text2text-generation", model="t5-small")

# Predefined reply templates
positive_replies = [
    "Thank you for your kind feedback! We're delighted you enjoyed the product.",
    "We appreciate your positive review! We'll keep working hard to maintain quality.",
    "Thanks for your support! We look forward to serving you again.",
    "Your satisfaction is our goal. Thank you for choosing us!"
]

negative_replies = [
    "We're sorry to hear about your experience. Could you share more details? We'll improve.",
    "Thank you for your honest feedback. We'll address the issue and do better.",
    "We apologize for the inconvenience. Please contact our support team for assistance.",
    "We value your feedback and will work on improving the product quality."
]

st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and a suggested reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing..."):
            # Sentiment analysis (first pipeline)
            tok, model = load_sentiment_model()
            inputs = tok(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            prob = torch.softmax(outputs.logits, dim=-1)[0][pred].item()
            label_map = {0: "😞 Negative", 1: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[pred]} (confidence: {prob:.2f})")

            # Second pipeline: T5 generates a short additional sentence
            t5_pipe = load_t5_pipeline()
            if pred == 1:  # Positive
                base_reply = random.choice(positive_replies)
                # Ask T5 to generate a short positive closing
                extra = t5_pipe("Generate a short thank you sentence: ", max_length=20, do_sample=False)[0]['generated_text']
            else:  # Negative
                base_reply = random.choice(negative_replies)
                extra = t5_pipe("Generate a short apology sentence: ", max_length=20, do_sample=False)[0]['generated_text']
            
            # Combine
            full_reply = f"{base_reply} {extra}"
            st.success(f"**Suggested Reply:** {full_reply}")
    else:
        st.warning("Please enter a review.")
