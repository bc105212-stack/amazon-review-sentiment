import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as T5Tokenizer, AutoModelFor Seq2SeqLM

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
st.markdown("Enter a customer review below to get sentiment analysis and a suggested reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing..."):
            # Sentiment analysis
            tok, model = load_sentiment_model()
            inputs = tok(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
            label_map = {0: "😞 Negative", 1: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[pred]} (confidence: {confidence:.2f})")

            # Generate customer service reply
            t5_tok, t5_model = load_t5_model()
            sentiment_word = "negative" if pred == 0 else "positive"
            # Improved prompt: ask T5 to write a helpful customer service reply
            prompt = f"Write a helpful customer service reply to a {sentiment_word} review: \"{review}\""
            inputs_t5 = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs_t5 = t5_model.generate(
                inputs_t5["input_ids"],
                max_length=100,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            reply = t5_tok.decode(outputs_t5[0], skip_special_tokens=True)
            # Fallback if reply is too short or meaningless
            if len(reply) < 10 or reply.lower().startswith("generate"):
                reply = "Thank you for your feedback. We will work to improve our service."
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
