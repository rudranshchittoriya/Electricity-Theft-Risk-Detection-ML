import pandas as pd
import numpy as np

TARGET = "Theft Label"

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def engineer_features(df):
    df = clean_columns(df)

    # Normalize common column-name variants
    rename = {
        "Customer - ID": "Customer_ID",
        "Customer ID": "Customer_ID",
        "Monthly Consumption": "Monthly_Consumption",
        "Previous Month Consumption": "Previous_Month_Consumption",
        "Avg Consumption 6M": "Avg_Consumption_6M",
        "Max Consumption 6M": "Max_Consumption_6M",
        "Min Consumption 6M": "Min_Consumption_6M",
        "Connection Type": "Connection_Type",
        "Voltage Fluctuations": "Voltage_Fluctuations",
        "Seasonal Factor": "Seasonal_Factor",
        "Consumption Ratio": "Consumption_Ratio",
        "Meter Tampering": "Meter_Tampering",
        "Theft Label": TARGET,
    }
    df = df.rename(columns=rename)

    numeric = [
        "Monthly_Consumption", "Previous_Month_Consumption",
        "Avg_Consumption_6M", "Max_Consumption_6M",
        "Min_Consumption_6M", "Voltage_Fluctuations",
        "Consumption_Ratio"
    ]
    for c in numeric:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    eps = 1e-6
    df["Consumption_Change_Pct"] = (
        (df["Monthly_Consumption"] - df["Previous_Month_Consumption"])
        / (df["Previous_Month_Consumption"].abs() + eps)
    ) * 100

    df["Six_Month_Range"] = (
        df["Max_Consumption_6M"] - df["Min_Consumption_6M"]
    )

    df["Deviation_From_6M_Avg"] = (
        df["Monthly_Consumption"] - df["Avg_Consumption_6M"]
    )

    df["Consumption_Stability"] = (
        df["Six_Month_Range"] /
        (df["Avg_Consumption_6M"].abs() + eps)
    )

    df["Voltage_Risk"] = df["Voltage_Fluctuations"].abs()
    df["Range_Ratio"] = (
        df["Six_Month_Range"] /
        (df["Max_Consumption_6M"].abs() + eps)
    )

    return df

def prepare_xy(df):
    df = engineer_features(df)

    y = df[TARGET].astype(int)

    # Deliberately remove ID and direct tampering signal to prevent leakage.
    drop_cols = [
        TARGET, "Customer_ID", "Meter_Tampering"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    return X, y, numeric, categorical
