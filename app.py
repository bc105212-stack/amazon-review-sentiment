import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

NEGATIVE_KEYWORDS = ["bad", "terrible", "awful", "horrible", "poor", "not good", "waste", "useless", "disappointed", "frustrated", "hate"]
NEUTRAL_KEYWORDS = ["ok", "fine", "average", "decent", "mediocre"]

def correct_prediction(text, pred_label, confidence):

    text_lower = text.lower()
    if pred_label == 2:  # positive

        for kw in NEGATIVE_KEYWORDS:
            if kw in text_lower:
                return 0  # negative

    if pred_label == 2 and confidence < 0.7:

        has_positive = any(word in text_lower for word in ["good", "great", "awesome", "excellent"])
        if not has_positive:
            return 1  # neutral
    return pred_label

REPLY_TEMPLATES = {
    0: [
        "We sincerely apologize for your bad experience. Please contact our support for a refund or replacement.",
        "Thank you for your feedback. We are sorry to hear that. We will improve our product quality.",
        "We regret that you faced an issue. Please share your order details so we can assist you."
    ],
    1: [
        "Thank you for your feedback. We will continue to improve our product.",
        "We appreciate your review. Let us know if you have any suggestions."
    ],
    2: [
        "Thank you for your positive review! We are glad you enjoyed our product.",
        "Great to hear you liked it! Your satisfaction is our priority.",
        "Thanks for your kind words! We look forward to serving you again."
    ]
}

def generate_reply(sentiment_label):
    return random.choice(REPLY_TEMPLATES[sentiment_label])

@st.cache_resource
def load_sentiment():

    model_name = "EBSQ/amazon-sentiment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and a suggested reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing..."):
            tokenizer, model = load_sentiment()
            inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probs[0][pred].item()

            corrected_pred = correct_prediction(review, pred, confidence)

            label_map = {0: "😞 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
            original_label = label_map[pred]
            final_label = label_map[corrected_pred]

            if corrected_pred != pred:
                st.warning(f"Model predicted {original_label} (conf: {confidence:.2f}), but due to keywords, we interpret as {final_label}.")
            else:
                st.info(f"**Sentiment:** {final_label} (confidence: {confidence:.2f})")

            reply = generate_reply(corrected_pred)
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
