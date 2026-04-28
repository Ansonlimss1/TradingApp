import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# SAFE IMPORT HANDLING
# -------------------------
try:
    import yfinance as yf
except ImportError:
    st.error("❌ yfinance is not installed. Run: pip install yfinance")
    st.stop()

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="XAUUSD Trading App", layout="wide")

st.title("📊 XAUUSD Live Trading Signal App (Stable Version)")
st.write("Educational tool only — NOT financial advice")

# -------------------------
# LOAD DATA WITH FALLBACK
# -------------------------
@st.cache_data
def load_data():
    symbols = ["XAUUSD=X", "GC=F"]  # fallback system

    for symbol in symbols:
        try:
            data = yf.download(symbol, period="6mo", interval="1h")

            if data is not None and not data.empty:
                st.success(f"✅ Data loaded from {symbol}")
                return data

        except Exception:
            continue

    return pd.DataFrame()

df = load_data()

# -------------------------
# HANDLE EMPTY DATA
# -------------------------
if df.empty:
    st.error("❌ Failed to load market data. Try again later.")
    st.stop()

# -------------------------
# SHOW RAW DATA
# -------------------------
st.subheader("📌 Market Data Preview")
st.write(df.tail())

# -------------------------
# INDICATORS
# -------------------------
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

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
ax.plot(df["Close"], label="Price")
ax.plot(df["SMA_20"], label="SMA 20")
ax.plot(df["SMA_50"], label="SMA 50")
ax.set_title("XAUUSD Trend Analysis")
ax.legend()

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
st.info("Data source: Yahoo Finance. Refresh to update.")
st.warning("⚠ Not financial advice. For learning only.")
