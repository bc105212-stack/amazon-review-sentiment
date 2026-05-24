import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

@st.cache_resource
def load_sentiment():
    model_name = "EBSQ/amazon-sentiment-distilbert"   # 替换为你的真实模型名称
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


REPLY_TEMPLATES = {
    0: [   # negative
        "We sincerely apologize for your bad experience. Please contact our support for a refund or replacement.",
        "Thank you for your feedback. We are sorry to hear that. We will improve our product quality.",
        "We regret that you faced an issue. Please share your order details so we can assist you."
    ],
    1: [   # neutral
        "Thank you for your feedback. We will continue to improve our product.",
        "We appreciate your review. Let us know if you have any suggestions."
    ],
    2: [   # positive
        "Thank you for your positive review! We are glad you enjoyed our product.",
        "Great to hear you liked it! Your satisfaction is our priority.",
        "Thanks for your kind words! We look forward to serving you again."
    ]
}

def generate_reply(sentiment_label):
    return random.choice(REPLY_TEMPLATES.get(sentiment_label, REPLY_TEMPLATES[1]))

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

            label_map = {0: "😞 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[pred]} (confidence: {confidence:.2f})")

            if confidence < 0.6:
                st.warning("Low confidence prediction. The review might be ambiguous.")

            reply = generate_reply(pred)
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
