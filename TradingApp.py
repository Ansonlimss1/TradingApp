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

# Bollinger Bands
df["BB_Mid"] = df["Close"].rolling(window=20).mean()
df["BB_Std"] = df["Close"].rolling(window=20).std()
df["BB_Upper"] = df["BB_Mid"] + (df["BB_Std"] * 2)
df["BB_Lower"] = df["BB_Mid"] - (df["BB_Std"] * 2)

# RSI calculation
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()

rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD calculation
ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema_12 - ema_26
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

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
macd = latest["MACD"]
signal_line = latest["Signal_Line"]
bb_upper = latest["BB_Upper"]
bb_lower = latest["BB_Lower"]
bb_mid = latest["BB_Mid"]

# -------------------------
# IMPROVED SIGNAL LOGIC WITH CONFIDENCE
# -------------------------
buy_signals = 0
sell_signals = 0
total_indicators = 0

# SMA Trend (2 points)
if sma20 > sma50:
    buy_signals += 2
    total_indicators += 2
elif sma20 < sma50:
    sell_signals += 2
    total_indicators += 2

# RSI Momentum (2 points)
if rsi < 30:
    buy_signals += 2
    total_indicators += 2
elif rsi > 70:
    sell_signals += 2
    total_indicators += 2
elif 45 < rsi < 55:
    total_indicators += 2  # Neutral zone
else:
    total_indicators += 1

# MACD Signal (2 points)
if macd > signal_line and macd > 0:
    buy_signals += 2
    total_indicators += 2
elif macd < signal_line and macd < 0:
    sell_signals += 2
    total_indicators += 2
else:
    total_indicators += 1

# Bollinger Bands (1 point)
if price < bb_lower:
    buy_signals += 1
    total_indicators += 1
elif price > bb_upper:
    sell_signals += 1
    total_indicators += 1

# Calculate confidence score (0-100%)
if total_indicators > 0:
    if buy_signals > sell_signals:
        signal = "🟢 BUY"
        confidence = int((buy_signals / total_indicators) * 100)
    elif sell_signals > buy_signals:
        signal = "🔴 SELL"
        confidence = int((sell_signals / total_indicators) * 100)
    else:
        signal = "🟡 HOLD"
        confidence = 50
else:
    signal = "🟡 HOLD"
    confidence = 0

# -------------------------
# DASHBOARD
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Price", f"{price:.2f}")
col2.metric("📉 RSI", f"{rsi:.2f}")
col3.metric("📊 Signal", signal)
col4.metric("💪 Confidence", f"{confidence}%")

# -------------------------
# PRICE CHART
# -------------------------
st.subheader("📈 Gold Price Chart")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["Close"], label="Close Price", linewidth=2)
ax.plot(df["SMA_20"], label="SMA 20", alpha=0.7)
ax.plot(df["SMA_50"], label="SMA 50", alpha=0.7)
ax.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.2, color="gray", label="Bollinger Bands")
ax.plot(df["BB_Upper"], "--", color="gray", alpha=0.5)
ax.plot(df["BB_Lower"], "--", color="gray", alpha=0.5)
ax.legend()
ax.set_title("XAUUSD Trend Analysis with Bollinger Bands")
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# -------------------------
# RSI CHART
# -------------------------
st.subheader("📉 RSI Indicator")

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(df["RSI"], label="RSI", color="purple")
ax2.axhline(70, linestyle="--", color="red")
ax2.axhline(30, linestyle="--", color="green")
ax2.fill_between(df.index, 30, 70, alpha=0.1, color="yellow")
ax2.set_title("RSI (14)")
ax2.legend()
ax2.grid(True, alpha=0.3)

st.pyplot(fig2)

# -------------------------
# MACD CHART
# -------------------------
st.subheader("📊 MACD Indicator")

fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(df["MACD"], label="MACD", color="blue")
ax3.plot(df["Signal_Line"], label="Signal Line", color="red")
ax3.bar(df.index, df["MACD_Histogram"], label="Histogram", color="gray", alpha=0.3)
ax3.axhline(0, linestyle="-", color="black", linewidth=0.5)
ax3.set_title("MACD (12, 26, 9)")
ax3.legend()
ax3.grid(True, alpha=0.3)

st.pyplot(fig3)

# -------------------------
# DETAILED METRICS
# -------------------------
st.subheader("🔍 Detailed Indicator Analysis")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.write("**Moving Averages**")
metric_col1.write(f"SMA 20: {sma20:.2f}")
metric_col1.write(f"SMA 50: {sma50:.2f}")
metric_col1.write(f"Status: {'📈 Bullish' if sma20 > sma50 else '📉 Bearish'}")

metric_col2.write("**Momentum**")
metric_col2.write(f"RSI: {rsi:.2f}")
metric_col2.write(f"MACD: {macd:.6f}")
metric_col2.write(f"Signal: {signal_line:.6f}")

metric_col3.write("**Volatility**")
metric_col3.write(f"BB Upper: {bb_upper:.2f}")
metric_col3.write(f"BB Lower: {bb_lower:.2f}")
metric_col3.write(f"BB Mid: {bb_mid:.2f}")

# -------------------------
# FOOTER
# -------------------------
st.info("Data source: Yahoo Finance (XAUUSD=X). Refresh page for updates.")
st.warning("⚠ This is NOT a real trading predictor. Use for learning only.")
