import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM

REPLY_TEMPLATES = {
    "negative": "We value your feedback. Our customer support team is ready to assist you. Please reach out to support@shopease.com, and we will resolve the issue as quickly as possible.",
    "positive": "We are thrilled that you had a great experience. Your satisfaction means the world to us. We look forward to serving you again!"
}

NEGATIVE_PHRASES = [
    "bad", "terrible", "awful", "horrible", "worst", "useless", "disappointing", "broke", "defective", "waste", "doesn't work", "not good",
    "poor", "don't like", "do not like", "hate", "dislike", "not worth", "very bad", "extremely bad", "trash", "garbage", "avoid"
]

def is_negative_text(text):
    text_lower = text.lower()
    for phrase in NEGATIVE_PHRASES:
        if phrase in text_lower:
            return True
    return False

@st.cache_resource
def load_sentiment():
    model_name = "EBSQ/amazon-sentiment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_t5():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and a suggested reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing..."):

            if is_obviously_negative(review):
                sentiment_label = "negative"
                confidence = 0.99
                st.info(f"**Sentiment:** 😞 Negative (rule-based, confidence: {confidence:.2f})")
            else:
                tokenizer, model = load_sentiment()
                inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred].item()
                sentiment_label = "negative" if pred == 0 else "positive"

                if sentiment_label == "positive" and confidence > 0.9 and is_obviously_negative(review):
                    sentiment_label = "negative"
                    confidence = 0.95
                    st.info(f"**Sentiment:** 😞 Negative (corrected, confidence: {confidence:.2f})")
                else:
                    emoji = "😞" if sentiment_label == "negative" else "😊"
                    st.info(f"**Sentiment:** {emoji} {sentiment_label.capitalize()} (confidence: {confidence:.2f})")
            
            t5_tokenizer, t5_model = load_t5()

            prompt = "Reply very short with one word: " + sentiment_label
            inputs_t5 = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
            with torch.no_grad():
                outputs_t5 = t5_model.generate(
                    inputs_t5["input_ids"],
                    max_length=5,
                    num_beams=2,
                    early_stopping=True
                )
            short_reply = t5_tokenizer.decode(outputs_t5[0], skip_special_tokens=True).strip()

            if len(short_reply) > 10 or not short_reply:
                short_reply = "Thanks" if sentiment_label == "positive" else "Sorry"

            full_reply = f"{short_reply}! {REPLY_TEMPLATES[sentiment_label]}"
            st.success(f"**Suggested Reply:** {full_reply}")
    else:
        st.warning("Please enter a review.")
