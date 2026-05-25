import streamlit as st
import torch
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

def extract_issue(review_text):
    """Simple keyword extraction for common complaints"""
    keywords = {
        "quality": "product quality",
        "price": "pricing",
        "shipping": "delivery time",
        "battery": "battery life",
        "screen": "display quality",
        "sound": "audio quality",
        "camera": "camera performance",
        "software": "software experience",
        "customer service": "customer support"
    }
    for kw, issue in keywords.items():
        if kw in review_text.lower():
            return issue
    return None

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
                confidence = torch.max(probs).item()
            pred = torch.argmax(outputs.logits, dim=-1).item()
            label_map = {0: "Negative", 1: "Positive"}
            sentiment = label_map[pred]
            st.info(f"**Sentiment:** {sentiment} (confidence: {confidence:.2%})")

            # Generate reply based on sentiment
            t5_tok, t5_model = load_t5_model()
            if pred == 1:  # Positive
                prompt = f"Write a short thank you reply to a customer for a positive review. Express gratitude and promise to keep improving. Review: {review}"
                inputs_t5 = t5_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
                outputs_t5 = t5_model.generate(
                    inputs_t5["input_ids"],
                    max_length=80,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                reply = t5_tok.decode(outputs_t5[0], skip_special_tokens=True)
                # Fallback if generation is too weird
                if len(reply) < 10 or "?" in reply[:30]:
                    reply = "Thank you so much for your positive feedback! We're glad you enjoyed it. We'll keep working to make our products even better. ❤️"
            else:  # Negative
                issue = extract_issue(review)
                if issue:
                    reply = f"We sincerely apologize for the issue you faced with {issue}. Your feedback is very important to us, and we will work with our team to improve it. Please contact our support for a resolution."
                else:
                    reply = "We're sorry to hear that your experience wasn't satisfactory. We value your feedback and will use it to improve our products. Please reach out to our customer service for further assistance."
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
