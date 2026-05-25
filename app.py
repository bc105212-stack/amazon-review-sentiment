import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM

# Cache sentiment model
@st.cache_resource
def load_sentiment():
    model_name = "EBSQ/amazon-sentiment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Cache T5 model
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
            # Sentiment analysis
            tok, model = load_sentiment()
            inputs = tok(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            label_map = {0: "😞 Negative", 1: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[pred]}")

            # Generate reply using T5-small
            t5_tok, t5_model = load_t5()
            sentiment_word = ["negative", "positive"][pred]
            prompt = f"Generate a customer service reply for a {sentiment_word} review: {review}"
            inputs_t5 = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs_t5 = t5_model.generate(inputs_t5["input_ids"], max_length=80, do_sample=False)
            reply = t5_tok.decode(outputs_t5[0], skip_special_tokens=True)
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
