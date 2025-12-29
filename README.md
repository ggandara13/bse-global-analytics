# 🏀 BSE Global Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bse-global-analytics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org)

<p align="center">
  <img src="https://media.licdn.com/dms/image/v2/C5622AQGlk2E3fhKiBA/feedshare-shrink_800/feedshare-shrink_800/0/1654477792148?e=2147483647&v=beta&t=TWJuLtxooQURrBz9U2GTG7X9t5iufy2YcuWpyhkcwM4" width="400">
</p>

## 📊 Overview

A **Data Science Prototype** for **BSE Global** (Brooklyn Nets, NY Liberty, Barclays Center), demonstrating machine learning, analytics, and business insights capabilities.

**Built for:** Senior Data Scientist Interview  
**Author:** Gerardo Gandara

---

## 🤖 Machine Learning Components

| Model | Type | Purpose |
|-------|------|---------|
| **Price Prediction** | Random Forest Regression | Predict ticket prices from game features |
| **Attendance Classification** | Multi-class Classifier | Predict attendance levels (Low/Med/High/Sellout) |
| **Sentiment Analysis** | NLP Text Classification | Analyze Reddit fan discussions |

---

## 🎯 Key Findings

| Insight | Data |
|---------|------|
| **Pricing Gap** | Knicks charge 6.4x more than Nets for same opponents |
| **Attendance Paradox** | Nets fill 98% capacity while ranked #21 |
| **ML Insight** | Opponent tier explains 70% of price variation |
| **Revenue Opportunity** | $12.1M potential capturing 10% of gap |

---

## 🛠️ Skills Demonstrated

- **Python/ML**: scikit-learn, Random Forest, Gradient Boosting
- **Data Collection**: API integration (NBA, SeatGeek, Reddit)
- **Visualization**: Plotly, Streamlit dashboards
- **NLP**: Sentiment classification
- **Business Analytics**: ROI, pricing optimization, segmentation
- **MLOps**: Model comparison, cross-validation, metrics

---

## 🚀 Live Demo

**[👉 Launch Dashboard](https://bse-global-analytics.streamlit.app)**

---

## 📁 Data Sources

| Source | Type | Records |
|--------|------|---------|
| NBA API | ✅ Real | 1,400+ |
| Weather API | ✅ Real | 481 |
| RapidAPI/SeatGeek | ✅ Real | 35 |
| Reddit API | ✅ Real | 215 |
| Research Data | Curated | 1,100+ |

**Total: 48 files, 3,236 rows**

---

## 🛠️ Installation

### Local Development

```bash
# Clone the repo
git clone https://github.com/ggandara13/bse-global-analytics.git
cd bse-global-analytics

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and `app.py`
5. Deploy!

---

## 📊 Dashboard Pages

### 🏠 Executive Summary
Overview of data science components, key findings, and skills demonstrated.

### 🤖 ML: Price Prediction
- Random Forest vs Gradient Boosting vs Linear Regression
- Feature importance analysis
- R², MAE, RMSE metrics
- Actual vs Predicted visualization

### 📊 ML: Attendance Model
- Multi-class classification
- Confusion matrix
- Attendance distribution analysis

### 💬 ML: Sentiment Analysis
- NLP classification of Reddit posts
- Sentiment distribution
- Sample posts by sentiment

### 🔮 Interactive Predictor
- **User inputs game features → Model predicts price & attendance**
- Scenario comparison table
- Real-time predictions

### 💰 Pricing Deep Dive
- Nets vs Knicks comparison charts
- Price by opponent tier

### 💡 Recommendations
- Data-driven action items
- Revenue opportunity quantification

---

## 📈 Technologies

- **Python 3.10+**
- **Streamlit** - Dashboard framework
- **scikit-learn** - Machine learning
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

---

## 👤 Author

**Gerardo Gandara**  
Senior Data Scientist Candidate | BSE Global

- 📧 Email: [gerardo.gandara@gmail.com](mailto:gerardo.gandara@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/gerardo-gandara](https://www.linkedin.com/in/gerardo-gandara/)
- 🐙 GitHub: [github.com/ggandara13](https://github.com/ggandara13)

---

## 📄 License

This project is for interview demonstration purposes.  
Data collected from public APIs and sources.

---

<p align="center">
  <b>🏀 Built for BSE Global Senior Data Scientist Interview | December 2025</b>
</p>
