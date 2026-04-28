import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="XAUUSD Trading Bot", layout="wide")
st.title("🤖 XAUUSD Trading Bot (Stable Final Version)")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data(ttl=300)
def load_data():
    for symbol in ["XAUUSD=X", "GC=F"]:
        try:
            data = yf.download(symbol, period="3mo", interval="1h", progress=False)
            if data is not None and not data.empty:
                return data
        except:
            continue
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("❌ Failed to load data")
    st.stop()

# -------------------------
# FIX MULTI COLUMN BUG
# -------------------------
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in df.columns and isinstance(df[col], pd.DataFrame):
        df[col] = df[col].iloc[:, 0]

# -------------------------
# INDICATORS
# -------------------------

# SMA
df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()

# EMA
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].fillna(50)

# MACD
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Bollinger
df["BB_Mid"] = df["Close"].rolling(20).mean()
df["BB_Std"] = df["Close"].rolling(20).std()
df["BB_Upper"] = df["BB_Mid"] + 2 * df["BB_Std"]
df["BB_Lower"] = df["BB_Mid"] - 2 * df["BB_Std"]

# -------------------------
# FIXED ATR
# -------------------------
df["HL"] = df["High"] - df["Low"]
df["HC"] = abs(df["High"] - df["Close"].shift())
df["LC"] = abs(df["Low"] - df["Close"].shift())

df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)

df["ATR"] = df["TR"].rolling(14).mean()
df["ATR"] = df["ATR"].bfill()

# -------------------------
# SAFE STOCHASTIC
# -------------------------
df["Low_14"] = df["Low"].rolling(14).min()
df["High_14"] = df["High"].rolling(14).max()

diff = (df["High_14"] - df["Low_14"]).replace(0, np.nan)

df["Stochastic"] = 100 * (df["Close"] - df["Low_14"]) / diff
df["Stochastic"] = df["Stochastic"].replace([np.inf, -np.inf], np.nan).fillna(50)

# -------------------------
# SAFE CCI
# -------------------------
df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
tp_mean = df["TP"].rolling(20).mean()
tp_std = df["TP"].rolling(20).std().replace(0, np.nan)

df["CCI"] = (df["TP"] - tp_mean) / (0.015 * tp_std)
df["CCI"] = df["CCI"].replace([np.inf, -np.inf], np.nan).fillna(0)

# -------------------------
# SAFE ADX
# -------------------------
df["Plus_DM"] = (df["High"] - df["High"].shift()).clip(lower=0)
df["Minus_DM"] = (df["Low"].shift() - df["Low"]).clip(lower=0)

df["Plus_DI"] = 100 * (df["Plus_DM"].rolling(14).mean() / df["ATR"])
df["Minus_DI"] = 100 * (df["Minus_DM"].rolling(14).mean() / df["ATR"])

di_sum = (df["Plus_DI"] + df["Minus_DI"]).replace(0, np.nan)

df["ADX"] = abs(df["Plus_DI"] - df["Minus_DI"]) / di_sum * 100
df["ADX"] = df["ADX"].replace([np.inf, -np.inf], np.nan).fillna(20)

# -------------------------
# CLEAN
# -------------------------
df = df.dropna()

latest = df.iloc[-1]

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

if latest["MACD"] > latest["Signal_Line"]:
    buy += 1
else:
    sell += 1

if latest["Stochastic"] < 20:
    buy += 1
elif latest["Stochastic"] > 80:
    sell += 1

signal = "HOLD"
if buy > sell:
    signal = "BUY"
elif sell > buy:
    signal = "SELL"

# -------------------------
# DISPLAY
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Price", f"{latest['Close']:.2f}")
col2.metric("📉 RSI", f"{latest['RSI']:.2f}")
col3.metric("📊 Signal", signal)

# -------------------------
# CHART
# -------------------------
fig, ax = plt.subplots()
ax.plot(df["Close"], label="Price")
ax.plot(df["SMA_20"], label="SMA 20")
ax.plot(df["SMA_50"], label="SMA 50")
ax.legend()
st.pyplot(fig)

st.warning("⚠️ Educational use only")
