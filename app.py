import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer as T5Tokenizer, AutoModelForSeq2SeqLM

# Page config
st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review to get sentiment analysis and a suggested reply.")

@st.cache_resource
def load_sentiment():
    # Replace with your actual Hugging Face model ID
    model_name = "EBSQ/amazon-sentiment-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_t5():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

def generate_reply(review, sentiment_label):
    """
    sentiment_label: 0 = negative, 1 = positive
    """
    sentiment_word = "negative" if sentiment_label == 0 else "positive"
    # Improved prompt: clear instruction, one sentence reply
    prompt = f"Customer review: {review}\nWrite a helpful customer service reply for a {sentiment_word} review in one sentence:"
    tokenizer, model = load_t5()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # If reply is still the prompt or empty, fallback to a default message
    if reply.startswith("Customer review:") or len(reply) < 5:
        reply = "Thank you for your feedback. We appreciate your business and will work to improve."
    return reply

# UI
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

            # Generate reply using T5
            reply = generate_reply(review, pred)
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
