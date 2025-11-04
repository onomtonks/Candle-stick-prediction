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
    yf.set_tz_cache_location("~/.cache/yfinance")
    data = yf.download(ticker, interval=interval, period=period, auto_adjust=False,group_by="ticker")
    if data.empty:
        return pd.DataFrame()  # Return empty DataFrame if download fails
    data.columns = data.columns.droplevel(0)
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
    for i in range(1,LAGS):
        df[f"open{i-1}"] = df["Open"].shift(-i)
        df[f"high{i-1}"] = df["High"].shift(-i)
        df[f"low{i-1}"] = df["Low"].shift(-i)
        df[f"close{i-1}"] = df["Close"].shift(-i)
        df[f"vwap{i-1}"] = df["VWAP"].shift(-i)
    df.rename(columns= {"Datetime": "timestamp",
                      'Close':'close',
                     "High":"high",
                     "Low":"low",
                     "Open":'open',
                     'Volume':'volume',
                      "VWAP":'vwap'
                     }, inplace=True)
    df = df[df[f"vwap{LAGS-2}"].notnull()].copy()  # Keep only rows where all lags exist
    return df

while True:
    try:
        print("Fetching latest data...")
        data = fetch_data()
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
        new_order = ["open", "high", "low",'close','volume','vwap']
        for i in range(LAGS-1):
                new_order.extend([f"open{i}", f"high{i}", f"low{i}", f"close{i}", f"vwap{i}"])
        last_rows = last_rows[new_order]
        X = last_rows[new_order].iloc[-1].values

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
        time.sleep(5 * 60)

# Get latest close price
        new_data = fetch_data()
        if new_data.empty:
            print("Unable to fetch new data. Skipping verification...")
    

        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [col[0] if isinstance(col, tuple) else col for col in new_data.columns]

        new_data.reset_index(inplace=True)
        latest_close = new_data["Close"].iloc[-1]
        if latest_close > org_open and pred == 1:
            print("✅ Correct Prediction (BUY)")
        elif latest_close < org_open and pred == 0:
            print("✅ Correct Prediction (SELL)")
        else:
            print("❌ Wrong Prediction")

        print("Entry Price:", org_open)
        print("Latest Close:", latest_close)
        print("------------------------------------\n")

        time.sleep(10*60)  # Wait before next update

    except Exception as e:
        print(f"Error occurred: {e}. Retrying in 60 seconds...")
        time.sleep(60)
