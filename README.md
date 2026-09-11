# 📰 Financial News Sentiment & Quant Alpha Signal Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF_Ngrams-00A67E?style=for-the-badge)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An enterprise **Natural Language Processing (NLP) & Quantitative Finance Pipeline** designed to ingest unstructured financial breaking news, classify directional market sentiment (**Bullish / Bearish / Neutral**), and synthesize statistical **Quant Alpha Trading Signals** with real-time probabilistic confidence.

---

## 📌 Executive Summary & Financial Value

Quantitative trading desks and algorithmic hedge funds process thousands of breaking regulatory filings (10-K, 10-Q) and news feeds per second. Automated sentiment analysis enables high-frequency directional positioning before market consensus forms:

$$	ext{Alpha Signal Score } (lpha_i) = \mathbb{E}[R_{i, t+\Delta t}] - eta_i R_{m, t+\Delta t} = f_{	ext{NLP}}(	ext{Headline}_i)$$

### TF-IDF Vectorization Formulation
Term frequency-inverse document frequency weighting extracts high-information financial n-grams (e.g., *"revenue surges"*, *"SEC investigation"*, *"rate cuts"*):

$$	ext{TF-IDF}(t, d, D) = 	ext{TF}(t, d) 	imes \ln\left(rac{1 + |D|}{1 + |\{d \in D : t \in d\}\|}
ight) + 1$$

---

## 🏗️ Architecture & Pipeline Flow

```
┌────────────────────────────────────────────────────────┐
│     Live Financial News Wire & SEC Regulatory Feeds    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  NLP Preprocessing & High-Order N-Gram Tokenization    │
│  - Lowercase, Punctuation Filtering, Stopword Cleaning │
│  - Bi-Gram & Tri-Gram TF-IDF Vectorization Matrix     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Multi-Class Calibrated Sentiment Classifier Engine    │
│  - Logistic Softmax with Regularization ($C=1.5$)      │
│  - Output: [P(Bullish), P(Bearish), P(Neutral)]       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Quant Alpha Signal Synthesizer & Execution Triage     │
│  - Bullish (P > 0.60) -> Long Entry Order Signal 🟢    │
│  - Bearish (P > 0.60) -> Short / Hedge Signal 🔴      │
│  - Neutral / Mixed    -> Zero Delta Exposure ⚪        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Interactive Streamlit Live Headline Testing Dashboard │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Model Benchmark & Performance Metrics

Evaluated on stratified financial headline test corpora:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Bullish** 🟢 | **1.0000** | **1.0000** | **1.0000** | 168 |
| **Bearish** 🔴 | **1.0000** | **1.0000** | **1.0000** | 168 |
| **Neutral** ⚪ | **1.0000** | **1.0000** | **1.0000** | 144 |
| **Overall Macro / Weighted** | **1.0000** | **1.0000** | **1.0000** | **480** |

> **Key Takeaway**: High-dimensional TF-IDF with bi-gram collocation captures critical financial inflection adjectives with sub-millisecond scoring latency.

---

## 📁 Repository Structure

```
kaggle-financial-news-sentiment-transformer/
├── app.py                     # Streamlit sentiment inference & alpha extraction UI
├── data/
│   ├── raw/
│   │   └── financial_news_dataset.csv     # Raw news headline corpora
│   └── processed/
│       └── sentiment_processed.csv        # Tokenized & labeled dataset
├── models/
│   └── sentiment_nlp_pipeline.joblib      # Serialized TF-IDF + Classifier pipeline
├── notebooks/
│   └── financial_news_sentiment_analysis.ipynb # NLP EDA & pipeline development
├── reports/
│   ├── metrics.json                       # Benchmark metrics report
│   └── sentiment_confusion_matrix.png     # Multi-class confusion matrix
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md                              # Technical documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ArjunaFransesco/kaggle-financial-news-sentiment-transformer.git
cd kaggle-financial-news-sentiment-transformer
python -m venv venv
venv\Scriptsctivate  # On Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Alpha Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to test live custom headlines.

### 3. Open Jupyter Notebook Pipeline
```bash
jupyter notebook notebooks/financial_news_sentiment_analysis.ipynb
```

---

## 👤 Author & Portfolio
- **Author**: **[Arjuna Fransesco](https://github.com/ArjunaFransesco)**
- **GitHub Repositories**: [https://github.com/ArjunaFransesco?tab=repositories](https://github.com/ArjunaFransesco?tab=repositories)
- **Portfolio Website**: [https://github.com/ArjunaFransesco/arjuna-portfolio](https://github.com/ArjunaFransesco/arjuna-portfolio)


<!-- Last Maintenance Audit: 2026-09-11 -->
