import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(page_title="Financial Sentiment & Alpha", page_icon="📰", layout="wide")
st.title("📰 Financial News Sentiment & Quant Alpha Engine")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **Portfolio**: [GitHub](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("📝 Enter Financial News Headline")
    headline = st.text_area(
        "Financial Headline / Breaking News:",
        "Quarterly profit surges 45% beat Wall Street analyst expectations significantly"
    )

with col2:
    st.subheader("📈 Alpha Signal Extraction")
    model_path = os.path.join(os.path.dirname(__file__), "models/sentiment_nlp_pipeline.joblib")
    if os.path.exists(model_path) and headline.strip():
        pipeline = joblib.load(model_path)
        pred = pipeline.predict([headline])[0]
        probs = pipeline.predict_proba([headline])[0]
        
        if pred == "Bullish":
            st.success(f"### Market Stance: **{pred} 🟢**")
        elif pred == "Bearish":
            st.error(f"### Market Stance: **{pred} 🔴**")
        else:
            st.info(f"### Market Stance: **{pred} ⚪**")
            
        prob_df = pd.DataFrame({"Sentiment": pipeline.classes_, "Confidence": probs})
        st.bar_chart(prob_df.set_index("Sentiment"))
