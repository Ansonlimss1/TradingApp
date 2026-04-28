import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="XAUUSD Live Bot", layout="wide")

st.title("🤖 XAUUSD Live Trading Bot")
st.write("⚠️ Educational only — not financial advice")

# -------------------------
# AUTO REFRESH (FIXED)
# -------------------------
refresh_rate = st.sidebar.slider("Refresh (seconds)", 5, 60, 10)

st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=10)
def load_data():
    for symbol in ["XAUUSD=X", "GC=F"]:
        try:
            data = yf.download(symbol, period="5d", interval="1m", progress=False)
            if data is not None and not data.empty:
                return data
        except:
            continue
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Failed to load data (Yahoo may block request)")
    st.stop()

# -------------------------
# FIX MULTI COLUMN
# -------------------------
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# -------------------------
# INDICATORS
# -------------------------
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()

# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].fillna(50)

# ATR
df["HL"] = df["High"] - df["Low"]
df["HC"] = abs(df["High"] - df["Close"].shift())
df["LC"] = abs(df["Low"] - df["Close"].shift())
df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)
df["ATR"] = df["TR"].rolling(14).mean().bfill()

df = df.dropna()

latest = df.iloc[-1]
price = latest["Close"]
atr = latest["ATR"]

# -------------------------
# SIGNAL
# -------------------------
buy = 0
sell = 0

if latest["SMA_20"] > latest["SMA_50"]:
    buy += 1
else:
    sell += 1

if latest["RSI"] < 30:
    buy += 1
elif latest["RSI"] > 70:
    sell += 1

if buy > sell:
    signal = "🟢 BUY"
    entry = price
    sl = price - 2 * atr
    tp = price + 3 * atr

elif sell > buy:
    signal = "🔴 SELL"
    entry = price
    sl = price + 2 * atr
    tp = price - 3 * atr

else:
    signal = "🟡 HOLD"
    entry = price
    sl = price - atr
    tp = price + atr

# -------------------------
# DISPLAY
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Live Price", f"{price:.2f}")
col2.metric("📉 RSI", f"{latest['RSI']:.2f}")
col3.metric("📊 Signal", signal)

st.divider()

# Trade setup
st.subheader("📍 Trade Setup")

c1, c2, c3 = st.columns(3)
c1.metric("Entry", f"{entry:.2f}")
c2.metric("Stop Loss", f"{sl:.2f}")
c3.metric("Take Profit", f"{tp:.2f}")

# Risk reward
risk = abs(entry - sl)
reward = abs(tp - entry)
if risk > 0:
    st.metric("Risk/Reward", f"1 : {reward/risk:.2f}")

# -------------------------
# CHART
# -------------------------
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(df["Close"], label="Price")
ax.plot(df["SMA_20"], label="SMA20")
ax.plot(df["SMA_50"], label="SMA50")

ax.axhline(entry, linestyle="--", label="Entry")
ax.axhline(sl, linestyle="--", label="SL")
ax.axhline(tp, linestyle="--", label="TP")

ax.legend()
st.pyplot(fig)

st.info(f"🔄 Refreshing every {refresh_rate} seconds")
