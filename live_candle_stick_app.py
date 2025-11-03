import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import time

# Load simplified model
coef, intercept = joblib.load("model_lr_5m_simple.pkl")
n_features = len(coef)
FEATURES_PER_LAG = 5
LAGS = n_features // FEATURES_PER_LAG  # Dynamically match model

def fetch_data(ticker="GLD", interval="1m", period="1d"):
    """Fetch minute-level data and flatten columns."""
    data = yf.download(ticker, interval=interval, period=period, progress=False)
    if data.empty:
        return pd.DataFrame()  # Return empty DataFrame if download fails

    # Flatten MultiIndex columns if present
    data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    # Replace spaces with underscores
    data.columns = [col.replace(' ', '_') for col in data.columns]
    data.reset_index(inplace=True)
    return data

def calculate_vwap(df):
    """Calculate VWAP for the DataFrame."""
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Price_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cum_PV'] = df['Price_Volume'].cumsum()
    df['Cum_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / df['Cum_Volume']
    return df

def create_lag_features(df, LAGS):
    """Create lag features for OHLCV and VWAP."""
    df = df.copy()
    for i in range(LAGS):
        df[f"open{i}"] = df["Open"].shift(-i)
        df[f"high{i}"] = df["High"].shift(-i)
        df[f"low{i}"] = df["Low"].shift(-i)
        df[f"close{i}"] = df["Close"].shift(-i)
        df[f"vwap{i}"] = df["VWAP"].shift(-i)
    df = df[df[f"vwap{LAGS-1}"].notnull()].copy()  # Keep only rows where all lags exist
    return df

while True:
    try:
        print("Fetching latest data...")
        data = fetch_data("GLD")
        if data.empty:
            print("Could not retrieve data. Retrying in 60 seconds...")
            time.sleep(60)
            continue

        data = calculate_vwap(data)

        # Use only last LAGS + 1 rows for feature calculation
        last_rows = data.tail(LAGS + 1).reset_index(drop=True)
        last_rows = create_lag_features(last_rows, LAGS)

        if last_rows.empty:
            print("Not enough data for lag features. Retrying in 10 seconds...")
            time.sleep(10)
            continue

        # Feature vector
        feature_cols = []
        for i in range(LAGS):
            feature_cols.extend([f"open{i}", f"high{i}", f"low{i}", f"close{i}", f"vwap{i}"])
        X = last_rows[feature_cols].iloc[-1].values

        if len(X) != n_features:
            print(f"Feature vector length mismatch ({len(X)}), expected {n_features}. Skipping...")
            time.sleep(10)
            continue

        org_open = last_rows["open0"].iloc[-1]

        # Logistic prediction
        score = np.dot(X, coef) + intercept
        pred = 1 if score > 0 else 0
        print(f"\nPrediction Made → {'BUY' if pred == 1 else 'SELL'}")

        # Latest close for verification
        latest_close = last_rows["close0"].iloc[-1]
        if latest_close > org_open and pred == 1:
            print("✅ Correct Prediction (BUY)")
        elif latest_close < org_open and pred == 0:
            print("✅ Correct Prediction (SELL)")
        else:
            print("❌ Wrong Prediction")

        print("Entry Price:", org_open)
        print("Latest Close:", latest_close)
        print("------------------------------------\n")

        time.sleep(60)  # Wait before next update

    except Exception as e:
        print(f"Error occurred: {e}. Retrying in 60 seconds...")
        time.sleep(60)
