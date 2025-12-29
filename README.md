# 🏀 BSE Global Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bse-global-analytics.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="https://media.licdn.com/dms/image/v2/C5622AQGlk2E3fhKiBA/feedshare-shrink_800/feedshare-shrink_800/0/1654477792148?e=2147483647&v=beta&t=TWJuLtxooQURrBz9U2GTG7X9t5iufy2YcuWpyhkcwM4" width="400">
</p>

## 📊 Overview

A comprehensive data analytics dashboard for **BSE Global** (Brooklyn Nets, NY Liberty, Barclays Center), built as part of a Senior Data Scientist interview project.

**Key Analysis Areas:**
- 🎫 Real-time ticket pricing analysis (SeatGeek API)
- 📊 NBA attendance rankings & comparisons
- 💬 AI-powered sentiment analysis (Reddit + Claude)
- 🔮 Attendance prediction model
- 💡 Strategic recommendations

---

## 🎯 Key Findings

| Insight | Data |
|---------|------|
| **Pricing Gap** | Knicks charge 6.4x more than Nets for same opponents |
| **Attendance Paradox** | Nets fill 98.1% capacity while ranked #21 |
| **Sentiment** | 32% positive, 28% negative (100 posts analyzed) |
| **Revenue Opportunity** | $12.1M potential capturing 10% of Knicks gap |

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

### 1. 📊 Executive Summary
Key metrics, findings overview, and data sources.

### 2. 💰 Pricing Analysis
- Real-time ticket prices from SeatGeek
- Nets vs Knicks comparison (same opponents)
- Opponent tier pricing (420% premium for Tier 1)

### 3. 🎟️ Attendance
- NBA attendance rankings (30 teams)
- Fair comparison with rebuilding teams
- Capacity utilization analysis

### 4. 💬 Sentiment Analysis
- AI classification of 100 Reddit posts
- Pain point breakdown
- Top complaints and positive themes

### 5. 🔮 Predictive Model
- Attendance prediction by game
- Price-as-demand proxy methodology
- Games needing promotional intervention

### 6. 💡 Recommendations
- Immediate, short-term, and long-term actions
- Revenue opportunity quantification

---

## 🧠 Methodology

### Pricing Analysis
- Real-time data from SeatGeek via RapidAPI
- 26 Nets games + 9 Knicks games
- Same-opponent comparison for fair analysis

### Sentiment Analysis
- 215 Reddit posts collected from r/GoNets
- AI classification using Claude API
- Pain point categorization and frequency analysis

### Predictive Model
- Ticket price as proxy for demand
- Tier-based classification
- Prototype for production ML model

---

## 📈 Technologies

- **Python 3.10+** - Core language
- **Streamlit** - Dashboard framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **Claude AI** - Sentiment analysis
- **NBA API** - Game data
- **RapidAPI** - Ticket pricing

---

## 👤 Author

**Gerardo Gandara**
- GitHub: [@ggandara13](https://github.com/ggandara13)
- LinkedIn: [Connect](https://linkedin.com/in/gerardogandara)

---

## 📄 License

This project is for interview demonstration purposes.
Data collected from public APIs and sources.

---

<p align="center">
  <b>🏀 Built for BSE Global Senior Data Scientist Interview | December 2025</b>
</p>
