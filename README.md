# ⚡ Electricity Theft Risk Detection using Machine Learning

An end-to-end **Machine Learning** project that identifies customers with suspicious electricity-consumption patterns.

## 🎯 Project Objective

Electricity theft creates financial losses for utilities and can produce unusual consumption patterns. This project uses classical ML to classify a customer as:

- `0` → Normal
- `1` → Suspicious / Possible Theft

The project intentionally avoids deep learning and focuses on **feature engineering, class imbalance, model comparison and explainable risk scoring**.

## 🧠 Models

The training pipeline compares:

1. Logistic Regression
2. Random Forest
3. HistGradientBoosting

The best model is selected using **F1-score for the theft class**, rather than accuracy alone.

## 🔒 Data Leakage Prevention

The dataset contains a `Meter Tampering` field. It is **not used as a predictor** because it is effectively a direct signal of tampering/theft and could create target leakage.

`Customer - ID` is also excluded because an identifier should not influence the prediction.

## 📊 Features Used

- Connection Type
- Monthly Consumption
- Previous Month Consumption
- Avg Consumption 6M
- Max Consumption 6M
- Min Consumption 6M
- Voltage Fluctuations
- Seasonal Factor
- Consumption Ratio
- Engineered:
  - Consumption Change %
  - Six-Month Range
  - Deviation From 6M Average
  - Consumption Stability
  - Voltage Risk
  - Range Ratio



## 🚀 Run in Google Colab

This project is designed to work without a laptop.

1. Open the notebook in Google Colab.
2. Run all cells.
3. The script downloads the public CSV dataset automatically.
4. Models are trained and compared.
5. Evaluation reports and plots are generated.
6. The best model is saved as `models/best_model.joblib`.

## 🌐 Optional Dashboard

Run:

```bash
streamlit run app.py
```

The dashboard accepts customer information and returns:

- Theft probability
- Risk category
- Main contributing indicators

## 📚 Dataset

The project uses a publicly available electricity-theft CSV dataset containing customer consumption, connection type, seasonal and meter-related fields.

Dataset reference:
https://github.com/Ramakrishna-Kaki/Data-set-for-AI-DS

The repository should be used as the dataset source rather than claiming the data was collected by this project.

## ⚠️ Important

This is an academic risk-classification system. A prediction is **not proof of electricity theft**. Real utility investigations require additional evidence and human review.

## 👨‍💻 Author

**Rudransh Chittoriya**

B.Tech — AI/ML
