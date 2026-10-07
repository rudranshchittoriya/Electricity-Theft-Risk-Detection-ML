import sys
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"

model = joblib.load(MODEL_PATH)

def predict_customer(customer):
    df = pd.DataFrame([customer])
    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= 0.50)

    if probability >= 0.75:
        risk = "HIGH"
    elif probability >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "prediction": prediction,
        "theft_probability": round(probability, 4),
        "risk": risk
    }

if __name__ == "__main__":
    sample = {
        "Connection Type": "Residential",
        "Monthly Consumption": 120,
        "Previous Month Consumption": 205,
        "Avg Consumption 6M": 190,
        "Max Consumption 6M": 240,
        "Min Consumption 6M": 150,
        "Voltage Fluctuations": 5,
        "Seasonal Factor": "Summer",
        "Consumption Ratio": 0.63
    }
    print(predict_customer(sample))
