import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="XAUUSD Predictor", layout="wide")

st.title("📊 XAUUSD Live Trading Signal App")
st.write("Educational tool only — NOT financial advice")

# -------------------------
# LOAD DATA SAFELY
# -------------------------
@st.cache_data
def load_data():
    try:
        data = yf.download("XAUUSD=X", period="6mo", interval="1h")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# -------------------------
# CHECK DATA
# -------------------------
if df is None or df.empty:
    st.error("❌ No data loaded. Please check internet or try again later.")
    st.stop()

# Debug view (IMPORTANT for you)
st.subheader("📌 Raw Data Preview")
st.write(df.tail())

# -------------------------
# INDICATORS (SAFE CALCULATION)
# -------------------------
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# RSI calculation
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Remove NaN rows safely
df = df.dropna()

# -------------------------
# LATEST VALUES
# -------------------------
latest = df.iloc[-1]

price = latest["Close"]
rsi = latest["RSI"]
sma20 = latest["SMA_20"]
sma50 = latest["SMA_50"]

# -------------------------
# SIGNAL LOGIC
# -------------------------
if sma20 > sma50 and rsi < 70:
    signal = "🟢 BUY"
elif sma20 < sma50 and rsi > 30:
    signal = "🔴 SELL"
else:
    signal = "🟡 HOLD"

# -------------------------
# DASHBOARD
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Price", f"{price:.2f}")
col2.metric("📉 RSI", f"{rsi:.2f}")
col3.metric("📊 Signal", signal)

# -------------------------
# PRICE CHART
# -------------------------
st.subheader("📈 Gold Price Chart")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Close"], label="Close Price")
ax.plot(df["SMA_20"], label="SMA 20")
ax.plot(df["SMA_50"], label="SMA 50")
ax.legend()
ax.set_title("XAUUSD Trend Analysis")

st.pyplot(fig)

# -------------------------
# RSI CHART
# -------------------------
st.subheader("📉 RSI Indicator")

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(df["RSI"], label="RSI", color="purple")
ax2.axhline(70, linestyle="--", color="red")
ax2.axhline(30, linestyle="--", color="green")
ax2.set_title("RSI (14)")
ax2.legend()

st.pyplot(fig2)

# -------------------------
# FOOTER
# -------------------------
st.info("Data source: Yahoo Finance (XAUUSD=X). Refresh page for updates.")
st.warning("⚠ This is NOT a real trading predictor. Use for learning only.")
