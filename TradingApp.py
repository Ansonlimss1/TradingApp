import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="XAUUSD Trading Bot", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
    .buy-signal { color: #2ecc71; font-weight: bold; font-size: 20px; }
    .sell-signal { color: #e74c3c; font-weight: bold; font-size: 20px; }
    .hold-signal { color: #f39c12; font-weight: bold; font-size: 20px; }
    .metric-box { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 XAUUSD Live Trading Prediction Bot")
st.markdown("**AI-Powered Trading Signals | XAUUSD Gold Futures**")
st.write("⚠️ **Disclaimer:** This is an educational tool only. NOT financial advice. Trade at your own risk.")

# -------------------------
# SIDEBAR - USER CONTROLS
# -------------------------
with st.sidebar:
    st.header("⚙️ Bot Settings")
    
    timeframe = st.selectbox(
        "Select Timeframe:",
        ["1h (1 Hour)", "4h (4 Hours)", "1d (1 Day)", "1w (1 Week)"],
        index=0
    )
    
    period_map = {"1h (1 Hour)": "1mo", "4h (4 Hours)": "3mo", "1d (1 Day)": "6mo", "1w (1 Week)": "1y"}
    interval_map = {"1h (1 Hour)": "1h", "4h (4 Hours)": "4h", "1d (1 Day)": "1d", "1w (1 Week)": "1wk"}
    
    period = period_map[timeframe]
    interval = interval_map[timeframe]
    
    risk_level = st.slider("Risk Level (%):", 1, 10, 5)
    
    st.divider()
    st.markdown("**Bot Sensitivity**")
    rsi_threshold = st.slider("RSI Threshold:", 20, 50, 30)
    macd_strength = st.slider("MACD Strength:", 0.1, 1.0, 0.5, step=0.1)
    
    st.divider()
    st.markdown("**🔧 Diagnostics**")
    
    if st.button("🔄 Force Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("🧪 Test Connection", use_container_width=True):
        st.info("Testing Yahoo Finance connection...")
        try:
            test_data = yf.download("GC=F", period="1d", interval="1h", progress=False)
            if test_data is not None and len(test_data) > 0:
                st.success("✅ Connection successful!")
                st.write(f"Retrieved {len(test_data)} candles")
            else:
                st.error("❌ No data returned")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")


# -------------------------
# LOAD DATA SAFELY
# -------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(symbol, period, interval):
    try:
        # Show status while downloading
        with st.spinner("📡 Fetching XAUUSD data from Yahoo Finance..."):
            data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        if data is None or len(data) == 0:
            return None
        
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 Troubleshooting tips:\n- Check internet connection\n- Yahoo Finance may be temporarily unavailable\n- Try refreshing the page (F5)\n- Try a different timeframe")
        return None

# Try to load data
df = load_data("XAUUSD=X", period, interval)

# If main symbol fails, try alternative
if df is None or df.empty:
    st.warning("⚠️ Trying alternative data source...")
    try:
        df = load_data("GC=F", "6mo", "1h")  # Gold futures as backup
        if df is not None and not df.empty:
            st.success("✅ Successfully loaded Gold Futures data")
    except:
        df = None

# -------------------------
# CHECK DATA
# -------------------------
if df is None or df.empty:
    st.error("❌ No data loaded. Unable to fetch XAUUSD data.")
    st.error("**Possible causes:**")
    st.error("1. ⚠️ Yahoo Finance API is temporarily unavailable")
    st.error("2. 🌐 Check your internet connection")
    st.error("3. 🔄 Try refreshing the page (Press F5)")
    st.error("4. ⏱️ Wait a few minutes and try again")
    st.error("\n**Note:** Yahoo Finance sometimes blocks automated requests. This is normal.")
    st.stop()

# -------------------------
# CALCULATE ALL TECHNICAL INDICATORS
# -------------------------
# Moving Averages
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()

# Exponential Moving Averages
df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

# Bollinger Bands
df["BB_Mid"] = df["Close"].rolling(window=20).mean()
df["BB_Std"] = df["Close"].rolling(window=20).std()
df["BB_Upper"] = df["BB_Mid"] + (df["BB_Std"] * 2)
df["BB_Lower"] = df["BB_Mid"] - (df["BB_Std"] * 2)
df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

# RSI Calculation
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].fillna(50)  # Fill NaN with neutral value

# MACD
df["MACD"] = df["EMA_12"] - df["EMA_26"]
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

# ATR (Average True Range)
df["High_Low"] = df["High"] - df["Low"]
df["High_Close"] = abs(df["High"] - df["Close"].shift(1))
df["Low_Close"] = abs(df["Low"] - df["Close"].shift(1))
df["TR"] = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1)
df["ATR"] = df["TR"].rolling(window=14).mean()
df["ATR"] = df["ATR"].fillna(df["ATR"].mean())  # Fill NaN with mean

# ADX (Average Directional Index)
df["Plus_DM"] = np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]), 
                          df["High"] - df["High"].shift(1), 0)
df["Minus_DM"] = np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)), 
                           df["Low"].shift(1) - df["Low"], 0)

# Handle division by zero in DI calculations
df["Plus_DI"] = np.where(df["ATR"] != 0, 100 * (df["Plus_DM"].rolling(window=14).mean() / df["ATR"]), 0)
df["Minus_DI"] = np.where(df["ATR"] != 0, 100 * (df["Minus_DM"].rolling(window=14).mean() / df["ATR"]), 0)

# ADX calculation with division by zero handling
di_sum = df["Plus_DI"] + df["Minus_DI"]
df["ADX"] = np.where(di_sum != 0, abs(df["Plus_DI"] - df["Minus_DI"]) / di_sum * 100, 0)
df["ADX"] = df["ADX"].fillna(20)  # Fill NaN with neutral value

# Stochastic Oscillator with division by zero handling
df["Low_14"] = df["Low"].rolling(window=14).min()
df["High_14"] = df["High"].rolling(window=14).max()
high_low_diff = df["High_14"] - df["Low_14"]
df["Stochastic"] = np.where(high_low_diff != 0, 100 * ((df["Close"] - df["Low_14"]) / high_low_diff), 50)
df["Stochastic"] = df["Stochastic"].fillna(50)  # Fill NaN with neutral value
df["Stochastic_SMA"] = df["Stochastic"].rolling(window=3).mean()
df["Stochastic_SMA"] = df["Stochastic_SMA"].fillna(50)

# CCI (Commodity Channel Index)
df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
tp_mean = df["TP"].rolling(window=20).mean()
tp_std = df["TP"].rolling(window=20).std()
df["CCI"] = np.where(tp_std != 0, (df["TP"] - tp_mean) / (0.015 * tp_std), 0)
df["CCI"] = df["CCI"].fillna(0)  # Fill NaN with neutral value

# Remove rows with any NaN values that might remain
df = df.dropna()

# -------------------------
# LATEST VALUES
# -------------------------
latest = df.iloc[-1]
price = latest["Close"]
previous_close = df.iloc[-2]["Close"]
price_change = price - previous_close
price_change_pct = (price_change / previous_close) * 100

# -------------------------
# ADVANCED SIGNAL GENERATION ALGORITHM
# -------------------------
def generate_trading_signal(latest, df, rsi_threshold, macd_strength):
    """
    Advanced multi-indicator signal generation
    Returns: (signal, confidence, reason, entry_price, stop_loss, take_profit)
    """
    
    buy_score = 0
    sell_score = 0
    total_weight = 0
    reasons = []
    
    # 1. TREND ANALYSIS (Weight: 25%)
    trend_weight = 25
    total_weight += trend_weight
    
    if latest["SMA_10"] > latest["SMA_20"] > latest["SMA_50"]:
        buy_score += trend_weight * 0.7
        reasons.append("✅ Strong Uptrend (SMA aligned)")
    elif latest["SMA_10"] < latest["SMA_20"] < latest["SMA_50"]:
        sell_score += trend_weight * 0.7
        reasons.append("✅ Strong Downtrend (SMA aligned)")
    elif latest["EMA_12"] > latest["EMA_26"]:
        buy_score += trend_weight * 0.4
        reasons.append("📈 EMA bullish crossover")
    elif latest["EMA_12"] < latest["EMA_26"]:
        sell_score += trend_weight * 0.4
        reasons.append("📉 EMA bearish crossover")
    
    # 2. RSI MOMENTUM (Weight: 20%)
    rsi_weight = 20
    total_weight += rsi_weight
    
    if latest["RSI"] < rsi_threshold:
        buy_score += rsi_weight * (1 - (latest["RSI"] / rsi_threshold))
        reasons.append(f"🔥 Oversold (RSI: {latest['RSI']:.1f})")
    elif latest["RSI"] > (100 - rsi_threshold):
        sell_score += rsi_weight * ((latest["RSI"] - (100 - rsi_threshold)) / rsi_threshold)
        reasons.append(f"⚠️ Overbought (RSI: {latest['RSI']:.1f})")
    
    # 3. MACD SIGNAL (Weight: 25%)
    macd_weight = 25
    total_weight += macd_weight
    
    if latest["MACD"] > latest["Signal_Line"] and latest["MACD"] > 0:
        macd_strength_score = min(abs(latest["MACD_Histogram"]) * 100 * macd_strength, macd_weight)
        buy_score += macd_strength_score
        reasons.append(f"📊 MACD bullish (Hist: {latest['MACD_Histogram']:.6f})")
    elif latest["MACD"] < latest["Signal_Line"] and latest["MACD"] < 0:
        macd_strength_score = min(abs(latest["MACD_Histogram"]) * 100 * macd_strength, macd_weight)
        sell_score += macd_strength_score
        reasons.append(f"📊 MACD bearish (Hist: {latest['MACD_Histogram']:.6f})")
    
    # 4. BOLLINGER BANDS (Weight: 15%)
    bb_weight = 15
    total_weight += bb_weight
    
    bb_position = (latest["Close"] - latest["BB_Lower"]) / (latest["BB_Upper"] - latest["BB_Lower"])
    
    if latest["Close"] < latest["BB_Lower"]:
        buy_score += bb_weight * 0.8
        reasons.append("🎯 Price below lower BB (oversold)")
    elif latest["Close"] > latest["BB_Upper"]:
        sell_score += bb_weight * 0.8
        reasons.append("🎯 Price above upper BB (overbought)")
    
    # 5. STOCHASTIC OSCILLATOR (Weight: 10%)
    stoch_weight = 10
    total_weight += stoch_weight
    
    if latest["Stochastic"] < 20:
        buy_score += stoch_weight * 0.8
        reasons.append(f"⚡ Stochastic oversold ({latest['Stochastic']:.1f})")
    elif latest["Stochastic"] > 80:
        sell_score += stoch_weight * 0.8
        reasons.append(f"⚡ Stochastic overbought ({latest['Stochastic']:.1f})")
    
    # 6. TREND STRENGTH (ADX) (Weight: 5%)
    adx_weight = 5
    total_weight += adx_weight
    
    if latest["ADX"] > 25:
        if buy_score > sell_score:
            buy_score += adx_weight * (latest["ADX"] / 100)
            reasons.append(f"💪 Strong trend confirmed (ADX: {latest['ADX']:.1f})")
        else:
            sell_score += adx_weight * (latest["ADX"] / 100)
    
    # Calculate confidence
    if total_weight > 0:
        buy_confidence = (buy_score / total_weight) * 100
        sell_confidence = (sell_score / total_weight) * 100
    else:
        buy_confidence = sell_confidence = 0
    
    # Determine signal
    if buy_confidence > sell_confidence + 5:
        signal = "🟢 BUY"
        confidence = buy_confidence
    elif sell_confidence > buy_confidence + 5:
        signal = "🔴 SELL"
        confidence = sell_confidence
    else:
        signal = "🟡 HOLD"
        confidence = 50
    
    # Calculate Entry, Stop Loss, and Take Profit
    atr = latest["ATR"]
    
    if signal == "🟢 BUY":
        entry_price = price
        stop_loss = entry_price - (atr * 2)
        take_profit = entry_price + (atr * 3)
    elif signal == "🔴 SELL":
        entry_price = price
        stop_loss = entry_price + (atr * 2)
        take_profit = entry_price - (atr * 3)
    else:
        entry_price = price
        stop_loss = price - atr
        take_profit = price + atr
    
    return signal, confidence, reasons, entry_price, stop_loss, take_profit

signal, confidence, reasons, entry_price, stop_loss, take_profit = generate_trading_signal(
    latest, df, rsi_threshold, macd_strength
)

# -------------------------
# MAIN DASHBOARD
# -------------------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Main Signal Box
    if "BUY" in signal:
        st.markdown(f'<div style="background-color: #2ecc71; padding: 30px; border-radius: 15px; text-align: center; color: white;">'
                   f'<h1>🟢 BUY SIGNAL</h1>'
                   f'<p style="font-size: 24px; font-weight: bold;">{confidence:.1f}% Confidence</p>'
                   f'</div>', unsafe_allow_html=True)
    elif "SELL" in signal:
        st.markdown(f'<div style="background-color: #e74c3c; padding: 30px; border-radius: 15px; text-align: center; color: white;">'
                   f'<h1>🔴 SELL SIGNAL</h1>'
                   f'<p style="font-size: 24px; font-weight: bold;">{confidence:.1f}% Confidence</p>'
                   f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: #f39c12; padding: 30px; border-radius: 15px; text-align: center; color: white;">'
                   f'<h1>🟡 HOLD</h1>'
                   f'<p style="font-size: 24px; font-weight: bold;">{confidence:.1f}% Confidence</p>'
                   f'</div>', unsafe_allow_html=True)

with col2:
    st.metric("💰 Current Price", f"{price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

with col3:
    st.metric("⏰ Last Update", datetime.now().strftime("%H:%M:%S"))

# -------------------------
# TRADING RECOMMENDATIONS
# -------------------------
st.subheader("📋 Trading Recommendations")

rec_col1, rec_col2, rec_col3 = st.columns(3)

with rec_col1:
    st.markdown("### 📍 Entry Price")
    st.markdown(f"<h3 style='color: #3498db;'>${entry_price:.2f}</h3>", unsafe_allow_html=True)

with rec_col2:
    st.markdown("### 🛑 Stop Loss")
    st.markdown(f"<h3 style='color: #e74c3c;'>${stop_loss:.2f}</h3>", unsafe_allow_html=True)

with rec_col3:
    st.markdown("### 🎯 Take Profit")
    st.markdown(f"<h3 style='color: #2ecc71;'>${take_profit:.2f}</h3>", unsafe_allow_html=True)

# Risk/Reward Ratio
risk = abs(entry_price - stop_loss)
reward = abs(take_profit - entry_price)
if risk > 0:
    rr_ratio = reward / risk
    st.metric("📊 Risk/Reward Ratio", f"1:{rr_ratio:.2f}")

# -------------------------
# SIGNAL REASONING
# -------------------------
st.subheader("🔍 Signal Analysis Breakdown")

reason_col1, reason_col2 = st.columns(2)

with reason_col1:
    st.markdown("#### ✅ Bullish Indicators")
    bullish_reasons = [r for r in reasons if "✅" in r or "📈" in r or "🔥" in r or "🎯" in r or "⚡" in r or "💪" in r]
    if bullish_reasons:
        for reason in bullish_reasons:
            st.write(reason)
    else:
        st.write("None detected")

with reason_col2:
    st.markdown("#### ⚠️ Bearish Indicators")
    bearish_reasons = [r for r in reasons if "📉" in r or "⚠️" in r]
    if bearish_reasons:
        for reason in bearish_reasons:
            st.write(reason)
    else:
        st.write("None detected")

# -------------------------
# ADVANCED CHARTS
# -------------------------
st.divider()
st.subheader("📊 Technical Analysis Charts")

# Create three chart columns
chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Price Action", "📉 Momentum", "🎯 Entry Signals"])

with chart_tab1:
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Price line
    ax.plot(df.index, df["Close"], label="Close Price", linewidth=2.5, color="#3498db")
    
    # Moving Averages
    ax.plot(df.index, df["SMA_10"], label="SMA 10", alpha=0.6, linestyle="--", color="#e74c3c")
    ax.plot(df.index, df["SMA_20"], label="SMA 20", alpha=0.6, linestyle="--", color="#f39c12")
    ax.plot(df.index, df["EMA_12"], label="EMA 12", alpha=0.6, linestyle=":", color="#2ecc71")
    
    # Bollinger Bands
    ax.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.15, color="gray", label="Bollinger Bands")
    ax.plot(df.index, df["BB_Upper"], "--", color="gray", alpha=0.4, linewidth=0.8)
    ax.plot(df.index, df["BB_Lower"], "--", color="gray", alpha=0.4, linewidth=0.8)
    
    # Highlight latest price
    ax.scatter(df.index[-1], price, color="red", s=200, zorder=5, marker="o", label="Current Price")
    
    ax.set_title("XAUUSD Price Action with Moving Averages & Bollinger Bands", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with chart_tab2:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 2, 2]})
    
    # RSI
    ax1.plot(df.index, df["RSI"], label="RSI (14)", color="purple", linewidth=2)
    ax1.axhline(70, linestyle="--", color="red", alpha=0.5, label="Overbought (70)")
    ax1.axhline(30, linestyle="--", color="green", alpha=0.5, label="Oversold (30)")
    ax1.fill_between(df.index, 30, 70, alpha=0.1, color="yellow")
    ax1.scatter(df.index[-1], latest["RSI"], color="red", s=100, zorder=5)
    ax1.set_ylabel("RSI")
    ax1.set_title("RSI (14) Momentum Indicator")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # MACD
    colors = ['green' if x > 0 else 'red' for x in df["MACD_Histogram"]]
    ax2.bar(df.index, df["MACD_Histogram"], label="Histogram", color=colors, alpha=0.4)
    ax2.plot(df.index, df["MACD"], label="MACD", color="blue", linewidth=2)
    ax2.plot(df.index, df["Signal_Line"], label="Signal Line", color="red", linewidth=2)
    ax2.axhline(0, linestyle="-", color="black", linewidth=0.5)
    ax2.scatter(df.index[-1], latest["MACD"], color="red", s=100, zorder=5)
    ax2.set_ylabel("MACD")
    ax2.set_title("MACD (12, 26, 9) Trend Following")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Stochastic
    ax3.plot(df.index, df["Stochastic"], label="Stochastic", color="orange", linewidth=2)
    ax3.plot(df.index, df["Stochastic_SMA"], label="Stochastic SMA", color="blue", linewidth=2)
    ax3.axhline(80, linestyle="--", color="red", alpha=0.5, label="Overbought (80)")
    ax3.axhline(20, linestyle="--", color="green", alpha=0.5, label="Oversold (20)")
    ax3.fill_between(df.index, 20, 80, alpha=0.1, color="yellow")
    ax3.scatter(df.index[-1], latest["Stochastic"], color="red", s=100, zorder=5)
    ax3.set_ylabel("Stochastic")
    ax3.set_xlabel("Date")
    ax3.set_title("Stochastic Oscillator")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    st.pyplot(fig)

with chart_tab3:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot price with entry, stop loss, and take profit
    ax.plot(df.index, df["Close"], label="Close Price", linewidth=2, color="#3498db")
    
    # Entry point
    ax.axhline(entry_price, color="blue", linestyle="-", linewidth=2, label=f"Entry: ${entry_price:.2f}")
    
    # Stop Loss
    ax.axhline(stop_loss, color="red", linestyle="--", linewidth=2, label=f"Stop Loss: ${stop_loss:.2f}")
    
    # Take Profit
    ax.axhline(take_profit, color="green", linestyle="--", linewidth=2, label=f"Take Profit: ${take_profit:.2f}")
    
    # Highlight zones
    ax.fill_between(df.index, stop_loss, entry_price, alpha=0.1, color="red")
    ax.fill_between(df.index, entry_price, take_profit, alpha=0.1, color="green")
    
    # Current price marker
    ax.scatter(df.index[-1], price, color="red", s=200, zorder=5, marker="*")
    
    ax.set_title("Trading Setup: Entry, Stop Loss, and Take Profit Levels", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# -------------------------
# DETAILED INDICATORS METRICS
# -------------------------
st.divider()
st.subheader("📊 Complete Indicator Metrics")

indicators_tab1, indicators_tab2, indicators_tab3 = st.tabs(["Averages & Bands", "Momentum", "Volatility & Trend"])

with indicators_tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Simple Moving Averages**")
        st.metric("SMA 10", f"{latest['SMA_10']:.2f}", f"{latest['SMA_10'] - latest['Close']:.2f}")
        st.metric("SMA 20", f"{latest['SMA_20']:.2f}", f"{latest['SMA_20'] - latest['Close']:.2f}")
        st.metric("SMA 50", f"{latest['SMA_50']:.2f}", f"{latest['SMA_50'] - latest['Close']:.2f}")
    
    with col2:
        st.markdown("**Exponential Moving Averages**")
        st.metric("EMA 12", f"{latest['EMA_12']:.2f}", f"{latest['EMA_12'] - latest['Close']:.2f}")
        st.metric("EMA 26", f"{latest['EMA_26']:.2f}", f"{latest['EMA_26'] - latest['Close']:.2f}")
        st.metric("EMA 50", f"{latest['EMA_50']:.2f}", f"{latest['EMA_50'] - latest['Close']:.2f}")
    
    with col3:
        st.markdown("**Bollinger Bands**")
        st.metric("BB Upper", f"{latest['BB_Upper']:.2f}", f"{latest['BB_Upper'] - latest['Close']:.2f}")
        st.metric("BB Middle", f"{latest['BB_Mid']:.2f}", f"{latest['BB_Mid'] - latest['Close']:.2f}")
        st.metric("BB Lower", f"{latest['BB_Lower']:.2f}", f"{latest['BB_Lower'] - latest['Close']:.2f}")

with indicators_tab2:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**RSI & Stochastic**")
        st.metric("RSI (14)", f"{latest['RSI']:.2f}", 
                 "🔴 Overbought" if latest['RSI'] > 70 else "🟢 Oversold" if latest['RSI'] < 30 else "🟡 Neutral")
        st.metric("Stochastic", f"{latest['Stochastic']:.2f}")
        st.metric("Stochastic SMA", f"{latest['Stochastic_SMA']:.2f}")
    
    with col2:
        st.markdown("**MACD**")
        st.metric("MACD", f"{latest['MACD']:.6f}", 
                 "🟢 Bullish" if latest['MACD'] > latest['Signal_Line'] else "🔴 Bearish")
        st.metric("Signal Line", f"{latest['Signal_Line']:.6f}")
        st.metric("Histogram", f"{latest['MACD_Histogram']:.6f}")
    
    with col3:
        st.markdown("**CCI**")
        st.metric("CCI", f"{latest['CCI']:.2f}",
                 "🔴 Overbought" if latest['CCI'] > 100 else "🟢 Oversold" if latest['CCI'] < -100 else "🟡 Neutral")

with indicators_tab3:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Average True Range**")
        st.metric("ATR (14)", f"{latest['ATR']:.4f}")
        st.markdown("*Volatility measure*")
    
    with col2:
        st.markdown("**Directional Indicators**")
        st.metric("Plus DI", f"{latest['Plus_DI']:.2f}")
        st.metric("Minus DI", f"{latest['Minus_DI']:.2f}")
    
    with col3:
        st.markdown("**ADX**")
        st.metric("ADX (14)", f"{latest['ADX']:.2f}",
                 "💪 Strong Trend" if latest['ADX'] > 25 else "⚠️ Weak Trend")

# -------------------------
# MARKET STATISTICS
# -------------------------
st.divider()
st.subheader("📈 Market Statistics")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

# Calculate returns and volatility
returns = df["Close"].pct_change().dropna()
volatility = returns.std() * 100

with stat_col1:
    st.metric("Daily Volatility", f"{volatility:.2f}%")

with stat_col2:
    recent_high = df["High"].tail(20).max()
    recent_low = df["Low"].tail(20).min()
    st.metric("20-Bar Range", f"${recent_high - recent_low:.2f}")

with stat_col3:
    st.metric("Avg Volume", f"{df['Volume'].tail(20).mean():,.0f}")

with stat_col4:
    win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    st.metric("Positive Candles %", f"{win_rate:.1f}%")

# -------------------------
# TRADING CONDITIONS
# -------------------------
st.divider()
st.subheader("⚠️ Important Trading Conditions")

conditions = []

if latest["ADX"] > 25:
    conditions.append(f"✅ Strong trend detected (ADX: {latest['ADX']:.1f}) - Good for trending strategies")
else:
    conditions.append(f"⚠️ Weak trend (ADX: {latest['ADX']:.1f}) - Market may be ranging")

if latest["BB_Width"] > df["BB_Width"].quantile(0.75):
    conditions.append("✅ High volatility - Good for breakout trades")
else:
    conditions.append("⚠️ Low volatility - May indicate consolidation")

if abs(price - latest["BB_Mid"]) > latest["BB_Upper"] - latest["BB_Mid"]:
    conditions.append("⚠️ Price far from middle band - Potential mean reversion")

for condition in conditions:
    st.write(condition)

# -------------------------
# FOOTER
# -------------------------
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.info("📊 **Data Source:** Yahoo Finance (XAUUSD=X)")

with footer_col2:
    st.warning("⚠️ **Disclaimer:** Educational tool only. NOT financial advice.")

with footer_col3:
    st.info(f"🔄 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
