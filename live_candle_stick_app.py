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
    ata = yf.download("GLD", interval="1m", period="1d")
    if not data.empty:
    # Simplify column names for easier access
        data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]

    # 1. Calculate the Typical Price (TP)
        data['Typical Price'] = (data['High_GLD'] + data['Low_GLD'] + data['Close_GLD']) / 3

    # 2. Multiply TP by Volume to get Price-Volume (PV)
        data['Price-Volume'] = data['Typical Price'] * data['Volume_GLD']

    # 3. Calculate the Cumulative PV and Cumulative Volume
        data['Cumulative PV'] = data['Price-Volume'].cumsum()
        data['Cumulative Volume'] = data['Volume_GLD'].cumsum()

    # 4. Calculate VWAP by dividing Cumulative PV by Cumulative Volume
        data['VWAP'] = data['Cumulative PV'] / data['Cumulative Volume']

    # Display the minute-by-minute high, low, volume, and VWAP
        print(data[['High_GLD', 'Low_GLD', 'Volume_GLD', 'VWAP']].tail())
    else:
        print("Could not retrieve minute-by-minute data to calculate VWAP.")
    last_5 = data.tail(10).reset_index()
    last_5.rename(columns={'Datetime':'timestamp'}, inplace=True)
    last_1 = data.tail(1).reset_index()
    last_1.rename(columns={'Datetime':'timestamp'}, inplace=True)
    LAGS = 10
    for i in range(LAGS):
            last_5[f"open{i}"] = last_5["Open_GLD"].shift(-i)
            last_5[f"high{i}"] = last_5["High_GLD"].shift(-i)
            last_5[f"low{i}"] = last_5["Low_GLD"].shift(-i)
            last_5[f"close{i}"] = last_5["Close_GLD"].shift(-i)
            last_5[f"vwap{i}"] = last_5["VWAP"].shift(-i)
    last_5.rename(columns= {"Datetime": "timestamp",
                          'Close_GLD':'close',
                         "High_GLD":"high",
                         "Low_GLD":"low",
                         "Open_GLD":'open',
                         'Volume_GLD':'volume',
                          "VWAP":'vwap'
                         }, inplace=True)
    last_5 = last_5[last_5["vwap9"].notnull()].copy()
    last_5 = last_5.drop(columns=["timestamp",'Typical Price','Price-Volume','Cumulative PV','Cumulative Volume'])
    new_order = ["open", "high", "low",'close','volume','vwap']
    for i in range(LAGS):
            new_order.extend([f"open{i}", f"high{i}", f"low{i}", f"close{i}", f"vwap{i}"])
    last_5 = last_5[new_order]
    # Get last 1 day 1-minute data
    org_open = last_5["open"].iloc[-1]

X = last_5[new_order].iloc[-1].values
if len(X) != n_features:
    print(f"Warning: feature vector length ({len(X)}) does not match model ({n_features}). Skipping...")
        time.sleep(60)
        
    
    # Logistic prediction
    score = np.dot(X, coef) + intercept
    pred = 1 if score > 0 else 0
    
    print(f"\nPrediction Made → {'BUY' if pred == 1 else 'SELL'}")
    print("Waiting 5 minutes for result...\n")
    time.sleep(5 * 60)
    
    # Get latest close price
    new_data = yf.download("GLD", interval="1m", period="1d")
    if new_data.empty:
        print("Unable to fetch new data. Skipping verification...")
        
    
    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = [col[0] if isinstance(col, tuple) else col for col in new_data.columns]
    
    new_data.reset_index(inplace=True)
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
    time.sleep(10)  # Small pause before next cycle
