import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM

# Cache sentiment model
@st.cache_resource
def load_sentiment_model():
    model_name = "EBSQ/amazon-sentiment-distilbert"
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
            # ----- Sentiment analysis with confidence -----
            tok, model = load_sentiment_model()
            inputs = tok(review, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)  # shape (1, 2)
            pred = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][pred].item()
            
            label_map = {0: "😞 Negative", 1: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[pred]} (confidence: {confidence:.2%})")

            # ----- Generate customer service reply based on sentiment -----
            t5_tok, t5_model = load_t5_model()
            sentiment_word = "negative" if pred == 0 else "positive"
            # Improved prompt to encourage proper customer service response
            prompt = (
                f"Write a helpful and polite customer service reply for a {sentiment_word} review. "
                f"Review: \"{review}\"\nReply:"
            )
            inputs_t5 = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs_t5 = t5_model.generate(
                inputs_t5["input_ids"],
                max_length=100,
                do_sample=True,          # 允许一定随机性，使回复更自然
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                repetition_penalty=1.2
            )
            reply = t5_tok.decode(outputs_t5[0], skip_special_tokens=True)
            # 清理可能的重复前缀（T5 有时会重复 prompt 中的关键词）
            if reply.startswith("Reply:"):
                reply = reply.replace("Reply:", "").strip()
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
