import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM

# Cache sentiment model
@st.cache_resource
def load_sentiment_model():
    model_name = "EBSQ/amazon-sentiment-distilbert"  # replace with your HF username
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Cache T5 model
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and suggested replies.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Replies"):
    if review.strip():
        with st.spinner("Analyzing sentiment..."):
            # Sentiment analysis
            tok, model = load_sentiment_model()
            inputs = tok(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(torch.argmax(logits, dim=-1).item())
            confidence = probs[pred]
            label_map = {0: "😞 Negative", 1: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[pred]} (confidence: {confidence:.2f})")

            # Generate replies (3 candidate replies)
            t5_tok, t5_model = load_t5_model()
            sentiment_word = "negative" if pred == 0 else "positive"
            # Improved prompt for better relevance
            if pred == 0:  # negative
                prompt = f"Write a helpful customer service reply apologizing for the issue and offering a solution for this negative review: {review}"
            else:  # positive
                prompt = f"Write a thank you reply to a customer who gave a positive review: {review}"
            
            inputs_t5 = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs_t5 = t5_model.generate(
                inputs_t5["input_ids"],
                max_length=100,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=3,
                no_repeat_ngram_size=2
            )
            replies = [t5_tok.decode(out, skip_special_tokens=True) for out in outputs_t5]
            
            st.subheader("Suggested Replies:")
            selected_reply = st.selectbox("Choose a reply to use:", replies)
            st.success(f"**Selected Reply:** {selected_reply}")
    else:
        st.warning("Please enter a review.")
