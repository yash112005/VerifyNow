import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base_path, 'fake_news_model.pkl'))
    vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.pkl'))
    return model, vectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")
st.title("🔍 Fake News Detector")
st.markdown("Paste any news article or headline below to check authenticity.")

model, vectorizer = load_model()

news_input = st.text_area("📰 Enter News Text:", height=200,
                           placeholder="Paste news article here...")

if st.button("🔎 Analyze", type="primary"):
    if news_input.strip():
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0].max() * 100

        if prediction == 1:
            st.success(f"✅ REAL NEWS — Confidence: {confidence:.1f}%")
        else:
            st.error(f"❌ FAKE NEWS — Confidence: {confidence:.1f}%")

        st.markdown("---")
        st.caption("Model: Random Forest + TF-IDF | Dataset: 44K+ articles")
    else:
        st.warning("Please enter some news text!")