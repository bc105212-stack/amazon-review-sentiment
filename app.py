import streamlit as st
from transformers import pipeline
import random

# 使用预训练的情感分析模型（三分类）
@st.cache_resource
def load_sentiment():
    # 使用卡迪夫大学的 RoBERTa 模型，输出 label: negative, neutral, positive
    return pipeline("sentiment-analysis", 
                    model="EBSQ/amazon-sentiment-distilbert",
                    tokenizer="EBSQ/amazon-sentiment-distilbert")

# 回复模板（仍然用模板，保证回复质量）
REPLY_TEMPLATES = {
    "negative": [
        "We sincerely apologize for your bad experience. Please contact our support for a refund or replacement.",
        "Thank you for your feedback. We are sorry to hear that. We will improve our product quality.",
        "We regret that you faced an issue. Please share your order details so we can assist you."
    ],
    "neutral": [
        "Thank you for your feedback. We will continue to improve our product.",
        "We appreciate your review. Let us know if you have any suggestions."
    ],
    "positive": [
        "Thank you for your positive review! We are glad you enjoyed our product.",
        "Great to hear you liked it! Your satisfaction is our priority.",
        "Thanks for your kind words! We look forward to serving you again."
    ]
}

def generate_reply(sentiment):
    """根据情感标签返回一条随机客服回复"""
    return random.choice(REPLY_TEMPLATES.get(sentiment, REPLY_TEMPLATES["neutral"]))

# Streamlit UI
st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and a suggested reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing..."):
            sentiment_pipe = load_sentiment()
            # 注意：该模型返回格式 [{'label': 'negative', 'score': 0.99}]
            result = sentiment_pipe(review)[0]
            sentiment = result['label']   # 'negative', 'neutral', 'positive'
            score = result['score']
            
            # 显示结果
            emoji = {"negative": "😞", "neutral": "😐", "positive": "😊"}
            st.info(f"**Sentiment:** {emoji[sentiment]} {sentiment.capitalize()} (confidence: {score:.2f})")
            
            # 生成回复
            reply = generate_reply(sentiment)
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
