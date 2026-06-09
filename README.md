# 🏥 Healthcare Provider Fraud Detection

A Streamlit ML app to detect fraudulent healthcare insurance providers.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📂 Data Setup

Upload your 8 CSV files directly in the app UI when prompted.

**OR** configure Google Drive auto-download — see SETUP_GUIDE.md.

## 📊 App Pages

| Page | What it does |
|---|---|
| 📋 Overview | Problem statement & dataset summary |
| 📊 EDA | Fraud distribution, demographics, claims charts |
| ⚙️ Features | 31 engineered features, correlation heatmap |
| 🤖 Training | Train RF/GBM/LR, CV, ROC curve, confusion matrix |
| 🔍 Predict | Predictions on unseen data + download CSV |
| 💡 Recommendations | Business insights & model roadmap |
