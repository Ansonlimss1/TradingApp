import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="XAUUSD REAL Bot", layout="wide")

st.title("🔥 REAL XAUUSD Live Trading Bot")
st.write("⚠️ Real data source (metals.live API)")

# -------------------------
# GET REAL PRICE (FIXED)
# -------------------------
def get_price():
    try:
        url = "https://api.metals.live/v1/spot/gold"
        r = requests.get(url, timeout=5)
        data = r.json()
        
        # format: [[timestamp, price]]
        price = float(data[0][1])
        return price
    except Exception as e:
        return None

price = get_price()

if price is None:
    st.error("❌ Failed to fetch real XAUUSD price (API blocked or down)")
    st.stop()

# -------------------------
# CREATE MINI DATA FOR INDICATORS
# -------------------------
# (Because API gives only latest price)
prices = [price + np.random.uniform(-5, 5) for _ in range(100)]
df = pd.DataFrame(prices, columns=["Close"])

# -------------------------
# INDICATORS
# -------------------------
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].fillna(50)

latest = df.iloc[-1]

# -------------------------
# SIGNAL LOGIC
# -------------------------
buy = 0
sell = 0

if latest["SMA20"] > latest["SMA50"]:
    buy += 1
else:
    sell += 1

if latest["RSI"] < 30:
    buy += 1
elif latest["RSI"] > 70:
    sell += 1

if buy > sell:
    signal = "🟢 BUY"
elif sell > buy:
    signal = "🔴 SELL"
else:
    signal = "🟡 HOLD"

# -------------------------
# TRADE LEVELS
# -------------------------
entry = price

if signal == "🟢 BUY":
    sl = price - 15
    tp = price + 25
elif signal == "🔴 SELL":
    sl = price + 15
    tp = price - 25
else:
    sl = price - 10
    tp = price + 10

# -------------------------
# DISPLAY
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🔥 REAL XAUUSD", f"{price:.2f}")
col2.metric("RSI", f"{latest['RSI']:.2f}")
col3.metric("Signal", signal)

st.divider()

st.subheader("📍 Trade Setup")

c1, c2, c3 = st.columns(3)
c1.metric("Entry", f"{entry:.2f}")
c2.metric("Stop Loss", f"{sl:.2f}")
c3.metric("Take Profit", f"{tp:.2f}")

# -------------------------
# CHART
# -------------------------
fig, ax = plt.subplots()
ax.plot(df["Close"], label="Simulated Price")
ax.legend()
st.pyplot(fig)

st.info("✔ Real price | ❗ Indicators simulated")
st.warning("⚠️ Not financial advice")
