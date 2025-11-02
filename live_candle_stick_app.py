import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import time
from datetime import timedelta

# Load simplified model
coef, intercept = joblib.load("model_lr_5m_simple.pkl")

# Number of past bars used in your model (you used 10)
LAGS = 10

while True:
    print("Fetching latest data...")

    # Get last 1 day 1-minute data
    data = yf.download("GLD", interval="1m", period="1d")

    if data.empty:
        print("Data unavailable. Retrying...")
        time.sleep(60)
        continue

    data.reset_index(inplace=True)
    data.rename(columns={"Datetime": "timestamp"}, inplace=True)

    # VWAP Calculation
    data['Typical Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['PV'] = data['Typical Price'] * data['Volume']
    data['CumPV'] = data['PV'].cumsum()
    data['CumVol'] = data['Volume'].cumsum()
    data['VWAP'] = data['CumPV'] / data['CumVol']

    # Keep latest rows needed
    df = data.tail(LAGS + 1).copy()

    # Create lag features
    for i in range(LAGS):
        df[f"open{i}"] = df["Open"].shift(i)
        df[f"high{i}"] = df["High"].shift(i)
        df[f"low{i}"] = df["Low"].shift(i)
        df[f"close{i}"] = df["Close"].shift(i)
        df[f"vwap{i}"] = df["VWAP"].shift(i)

    # Drop rows with null lag values
    df = df.dropna()

    org_open = df["open0"].iloc[-1]

    # Prepare feature vector (flat list)
    feature_cols = [col for col in df.columns if any(col.startswith(x) for x in ["open", "high", "low", "close", "vwap"])]
    X = df[feature_cols].iloc[-1].values

    # Manual logistic prediction
    score = np.dot(X, coef) + intercept
    pred = 1 if score > 0 else 0  # BUY = 1, SELL = 0

    print(f"\nPrediction Made → {'BUY' if pred == 1 else 'SELL'}")
    print("Waiting 5 minutes for result...\n")
    time.sleep(5 * 60)

    # Get new live close price
    new_data = yf.download("GLD", interval="1m", period="1d").reset_index()
    latest_close = new_data["Close"].iloc[-1]

    # Verify prediction
    if latest_close > org_open and pred == 1:
        print("✅ Correct Prediction (BUY)")
    elif latest_close < org_open and pred == 0:
        print("✅ Correct Prediction (SELL)")
    else:
        print("❌ Wrong Prediction")

    print("Entry Price:", org_open)
    print("Latest Close:", latest_close)
    print("------------------------------------\n")

    # Loop continues forever
    time.sleep(10)  # Small pause before next cycle
