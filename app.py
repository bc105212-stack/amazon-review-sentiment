import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

# ------------------ 1. 加载情感分类模型 ------------------
@st.cache_resource
def load_sentiment():
    model_name = "your_username/amazon-sentiment-distilbert"   # 替换为你的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# ------------------ 2. 规则引擎（优先于模型） ------------------
def rule_based_sentiment(text):
    """基于关键词的快速判断，返回 (label, confidence)"""
    text_lower = text.lower()
    # 负面关键词列表（可根据需要扩充）
    negative_words = ["bad", "terrible", "awful", "horrible", "worst", "useless", "disappointing", 
                      "not good", "poor", "waste", "defective", "broken", "faulty", "not work",
                      "never again", "hate", "sucks", "worse", "damage", "failure", "issue",
                      "problem", "return", "refund", "complaint", "unhappy"]
    # 正面关键词（用于避免误判）
    positive_words = ["good", "great", "excellent", "amazing", "perfect", "love", "best", "awesome",
                      "wonderful", "satisfied", "happy", "recommend", "fast", "quality"]

    # 优先检查负面词（避免同时出现时误判）
    for word in negative_words:
        if word in text_lower:
            return 0, 0.95   # negative with high pseudo-confidence
    for word in positive_words:
        if word in text_lower:
            return 2, 0.95   # positive
    
    # 无匹配则返回 None，交给模型
    return None, None

# ------------------ 3. 基于模板的回复生成 ------------------
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
    return random.choice(REPLY_TEMPLATES.get(sentiment_label, REPLY_TEMPLATES[1]))

# ------------------ 4. Streamlit UI ------------------
st.set_page_config(page_title="ShopEase AI Assistant", layout="centered")
st.title("🛍️ ShopEase Review Sentiment & Auto Reply")
st.markdown("Enter a customer review below to get sentiment analysis and a suggested reply.")

review = st.text_area("Customer Review:", height=150)

if st.button("Analyze & Generate Reply"):
    if review.strip():
        with st.spinner("Analyzing..."):
            # Step A: 先使用规则
            rule_label, rule_conf = rule_based_sentiment(review)
            if rule_label is not None:
                # 规则匹配成功，直接使用规则结果
                label = rule_label
                confidence = rule_conf
                method = "rule-based"
            else:
                # Step B: 使用模型预测
                tokenizer, model = load_sentiment()
                inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                confidence = probs[0][pred].item()
                label = pred
                method = "model"

            # 显示结果
            label_map = {0: "😞 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
            st.info(f"**Sentiment:** {label_map[label]} (confidence: {confidence:.2f}) using {method}")
            
            if method == "model" and confidence < 0.6:
                st.warning("Model confidence is low. The review might be ambiguous or contain mixed feelings.")
            
            # 生成回复
            reply = generate_reply(label)
            st.success(f"**Suggested Reply:** {reply}")
    else:
        st.warning("Please enter a review.")
