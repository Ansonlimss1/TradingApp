import streamlit as st
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="🚀 Live Trading Bot - 5 Min Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ===========================
# CUSTOM CSS
# ===========================
st.markdown("""
<style>
    .stMetricValue { font-size: 2.5rem; }
    .price-up { color: #2ecc71; }
    .price-down { color: #e74c3c; }
    .signal-buy { background-color: rgba(46, 204, 113, 0.2); padding: 15px; border-left: 5px solid #2ecc71; border-radius: 5px; }
    .signal-sell { background-color: rgba(231, 76, 60, 0.2); padding: 15px; border-left: 5px solid #e74c3c; border-radius: 5px; }
    .signal-hold { background-color: rgba(243, 156, 18, 0.2); padding: 15px; border-left: 5px solid #f39c12; border-radius: 5px; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .live-indicator { display: inline-block; width: 12px; height: 12px; background-color: #2ecc71; border-radius: 50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>
""", unsafe_allow_html=True)

# ===========================
# TITLE & HEADER
# ===========================
st.title("🚀 Live Trading Bot - 5 Minute Prediction")
st.markdown("### Real-time Price Tracking | Auto-Refresh | Entry/TP/SL Signals")
st.markdown("**Disclaimer:** Educational purpose only. Not financial advice. Trade at your own risk.")

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []

# ===========================
# LOAD LIVE PRICE DATA
# ===========================
@st.cache_data(ttl=30)
def get_live_price(symbol="EURUSD=X"):
    """Fetch current live price"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if len(data) > 0:
            return data.iloc[-1]['Close'], data
        return None, None
    except:
        return None, None

@st.cache_data(ttl=60)
def get_historical_data(symbol, period="5d", interval="5m"):
    """Fetch 5-minute candle data for analysis"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        return data
    except:
        return None

# ===========================
# TECHNICAL INDICATORS
# ===========================
def calculate_indicators(df):
    """Calculate all technical indicators for 5-minute timeframe"""
    
    if len(df) < 50:
        return df
    
    df = df.copy()
    
    # Moving Averages (fast for 5-min)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    
    # RSI (14 periods)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # ATR (Average True Range)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Stochastic
    df['Low_14'] = df['Low'].rolling(14).min()
    df['High_14'] = df['High'].rolling(14).max()
    df['Stochastic'] = 100 * ((df['Close'] - df['Low_14']) / (df['High_14'] - df['Low_14']))
    df['Stochastic'] = df['Stochastic'].fillna(50)
    
    # ADX
    df['Plus_DM'] = df['High'].diff()
    df['Minus_DM'] = -df['Low'].diff()
    df['Plus_DM'] = df['Plus_DM'].where((df['Plus_DM'] > df['Minus_DM']) & (df['Plus_DM'] > 0), 0)
    df['Minus_DM'] = df['Minus_DM'].where((df['Minus_DM'] > df['Plus_DM']) & (df['Minus_DM'] > 0), 0)
    
    df['ATR14'] = df['TR'].rolling(14).mean()
    df['Plus_DI'] = 100 * (df['Plus_DM'].rolling(14).mean() / df['ATR14'])
    df['Minus_DI'] = 100 * (df['Minus_DM'].rolling(14).mean() / df['ATR14'])
    df['ADX'] = 100 * abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])
    df['ADX'] = df['ADX'].fillna(20)
    
    return df.dropna()

# ===========================
# PRICE PREDICTION MODEL (5-MINUTE AHEAD)
# ===========================
def predict_next_price(df):
    """Predict price movement for next 5 minutes"""
    
    if len(df) < 30:
        return None, None, None
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features
        features = ['RSI', 'MACD', 'Histogram', 'Stochastic', 'ATR', 'ADX']
        X = df[features].tail(30).values
        y = df['Close'].tail(30).values
        
        # Train quick model
        model = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Predict next price
        next_features = df[features].iloc[-1:].values
        predicted_price = model.predict(next_features)[0]
        
        current_price = df['Close'].iloc[-1]
        price_change = predicted_price - current_price
        change_pct = (price_change / current_price) * 100
        
        return predicted_price, change_pct, current_price
    except:
        return None, None, None

# ===========================
# GENERATE TRADING SIGNAL
# ===========================
def generate_signal(df):
    """Generate BUY/SELL/HOLD signal with entry, TP, SL"""
    
    if len(df) < 20:
        return "WAIT", 0, {}, {}
    
    latest = df.iloc[-1]
    
    buy_signals = 0
    sell_signals = 0
    reasons = []
    
    # 1. Trend Analysis (SMA)
    if latest['SMA_5'] > latest['SMA_10'] > latest['SMA_20']:
        buy_signals += 2
        reasons.append("📈 SMA aligned bullish")
    elif latest['SMA_5'] < latest['SMA_10'] < latest['SMA_20']:
        sell_signals += 2
        reasons.append("📉 SMA aligned bearish")
    
    # 2. RSI Signal
    if latest['RSI'] < 30:
        buy_signals += 2
        reasons.append(f"🔥 RSI Oversold ({latest['RSI']:.1f})")
    elif latest['RSI'] > 70:
        sell_signals += 2
        reasons.append(f"⚠️ RSI Overbought ({latest['RSI']:.1f})")
    
    # 3. MACD Signal
    if latest['MACD'] > latest['Signal'] and latest['Histogram'] > 0:
        buy_signals += 2
        reasons.append("📊 MACD Bullish")
    elif latest['MACD'] < latest['Signal'] and latest['Histogram'] < 0:
        sell_signals += 2
        reasons.append("📊 MACD Bearish")
    
    # 4. Bollinger Bands
    if latest['Close'] < latest['BB_Lower']:
        buy_signals += 1.5
        reasons.append("🎯 Price at Lower BB")
    elif latest['Close'] > latest['BB_Upper']:
        sell_signals += 1.5
        reasons.append("🎯 Price at Upper BB")
    
    # 5. Stochastic
    if latest['Stochastic'] < 20:
        buy_signals += 1
        reasons.append(f"⚡ Stochastic Oversold")
    elif latest['Stochastic'] > 80:
        sell_signals += 1
        reasons.append(f"⚡ Stochastic Overbought")
    
    # 6. ADX (Trend Strength)
    if latest['ADX'] > 25:
        if buy_signals > sell_signals:
            buy_signals += 1
            reasons.append(f"💪 Strong Uptrend (ADX: {latest['ADX']:.1f})")
        elif sell_signals > buy_signals:
            sell_signals += 1
            reasons.append(f"💪 Strong Downtrend (ADX: {latest['ADX']:.1f})")
    
    # Determine signal
    confidence = max(buy_signals, sell_signals) / 12 * 100
    
    if buy_signals > sell_signals:
        signal = "BUY"
        
        # Entry: Current price
        entry = latest['Close']
        
        # TP: Based on ATR and recent movement
        atr = latest['ATR']
        tp = entry + (atr * 1.5)  # 1.5x ATR for target
        
        # SL: Below support
        sl = latest['BB_Lower'] if latest['Close'] > latest['BB_Lower'] else entry - (atr * 0.7)
        
        trade_info = {
            'Entry': entry,
            'TP': tp,
            'SL': sl,
            'R:R': round((tp - entry) / (entry - sl), 2) if entry > sl else 0,
            'Risk %': round((entry - sl) / entry * 100, 2)
        }
        
    elif sell_signals > buy_signals:
        signal = "SELL"
        
        # Entry: Current price
        entry = latest['Close']
        
        # TP: Based on ATR
        atr = latest['ATR']
        tp = entry - (atr * 1.5)
        
        # SL: Above resistance
        sl = latest['BB_Upper'] if latest['Close'] < latest['BB_Upper'] else entry + (atr * 0.7)
        
        trade_info = {
            'Entry': entry,
            'TP': tp,
            'SL': sl,
            'R:R': round((entry - tp) / (sl - entry), 2) if sl > entry else 0,
            'Risk %': round((sl - entry) / entry * 100, 2)
        }
    else:
        signal = "HOLD"
        trade_info = {}
    
    indicator_info = {
        'RSI': round(latest['RSI'], 2),
        'MACD': round(latest['MACD'], 6),
        'Stochastic': round(latest['Stochastic'], 2),
        'ADX': round(latest['ADX'], 2),
        'ATR': round(latest['ATR'], 4),
        'Price': round(latest['Close'], 4)
    }
    
    return signal, confidence, trade_info, indicator_info, reasons

# ===========================
# MAIN DISPLAY
# ===========================

# Get data
symbol = "EURUSD=X"  # Default to EUR/USD

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("## 📊 Live Price Chart (5-Minute)")

with col2:
    auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=True)

with col3:
    if st.button("🔃 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Fetch data
try:
    hist_data = get_historical_data(symbol, period="7d", interval="5m")
    
    if hist_data is None or len(hist_data) == 0:
        st.error("❌ Unable to fetch data. Trying alternative source...")
        symbol = "GC=F"  # Gold futures fallback
        hist_data = get_historical_data(symbol, period="7d", interval="5m")
    
    if hist_data is not None and len(hist_data) > 0:
        # Calculate indicators
        df = calculate_indicators(hist_data)
        
        if len(df) > 0:
            # Get current values
            current_price = df['Close'].iloc[-1]
            previous_close = df['Close'].iloc[-2]
            price_change = current_price - previous_close
            price_change_pct = (price_change / previous_close) * 100
            
            # Get signal
            signal, confidence, trade_info, indicator_info, reasons = generate_signal(df)
            
            # Predict next price
            pred_price, pred_change_pct, _ = predict_next_price(df)
            
            # ===========================
            # DISPLAY CURRENT PRICE
            # ===========================
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                price_color = "🟢" if price_change >= 0 else "🔴"
                st.metric(
                    "Current Price",
                    f"{current_price:.5f}",
                    f"{price_change:+.5f} ({price_change_pct:+.2f}%)",
                    delta_color="normal"
                )
            
            with col2:
                st.metric("Symbol", symbol)
            
            with col3:
                if pred_change_pct is not None:
                    st.metric("5-Min Forecast", f"{pred_change_pct:+.3f}%", "Predicted")
                else:
                    st.metric("5-Min Forecast", "Loading...", "")
            
            with col4:
                st.metric("Confidence", f"{confidence:.1f}%", signal)
            
            st.markdown("---")
            
            # ===========================
            # TRADING SIGNAL SECTION
            # ===========================
            
            if signal == "BUY":
                st.markdown(f"""
                <div class="signal-buy">
                    <h2>🟢 BUY SIGNAL ({confidence:.1f}% Confidence)</h2>
                    <p><b>Entry Price:</b> {trade_info['Entry']:.5f}</p>
                    <p><b>Take Profit (TP):</b> {trade_info['TP']:.5f} ⭐</p>
                    <p><b>Stop Loss (SL):</b> {trade_info['SL']:.5f} ⛔</p>
                    <p><b>Risk/Reward Ratio:</b> 1:{trade_info['R:R']}</p>
                    <p><b>Risk %:</b> {trade_info['Risk %']}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif signal == "SELL":
                st.markdown(f"""
                <div class="signal-sell">
                    <h2>🔴 SELL SIGNAL ({confidence:.1f}% Confidence)</h2>
                    <p><b>Entry Price:</b> {trade_info['Entry']:.5f}</p>
                    <p><b>Take Profit (TP):</b> {trade_info['TP']:.5f} ⭐</p>
                    <p><b>Stop Loss (SL):</b> {trade_info['SL']:.5f} ⛔</p>
                    <p><b>Risk/Reward Ratio:</b> 1:{trade_info['R:R']}</p>
                    <p><b>Risk %:</b> {trade_info['Risk %']}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                <div class="signal-hold">
                    <h2>🟡 HOLD - NO CLEAR SIGNAL ({confidence:.1f}% Confidence)</h2>
                    <p>Waiting for stronger confirmation...</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ===========================
            # TECHNICAL INDICATORS
            # ===========================
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                rsi = indicator_info['RSI']
                rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                st.metric("RSI (14)", f"{rsi:.1f}", f"{rsi_color}")
            
            with col2:
                macd = indicator_info['MACD']
                macd_color = "🟢" if macd > 0 else "🔴"
                st.metric("MACD", f"{macd:.6f}", f"{macd_color}")
            
            with col3:
                stoch = indicator_info['Stochastic']
                stoch_color = "🟢" if stoch < 20 else "🔴" if stoch > 80 else "🟡"
                st.metric("Stochastic", f"{stoch:.1f}", f"{stoch_color}")
            
            with col4:
                adx = indicator_info['ADX']
                adx_color = "💪" if adx > 25 else "⚠️"
                st.metric("ADX", f"{adx:.1f}", f"{adx_color}")
            
            with col5:
                atr = indicator_info['ATR']
                st.metric("ATR", f"{atr:.5f}", "Volatility")
            
            st.markdown("---")
            
            # ===========================
            # SIGNAL REASONS
            # ===========================
            
            st.markdown("### 📋 Signal Analysis")
            for reason in reasons:
                st.write(f"• {reason}")
            
            st.markdown("---")
            
            # ===========================
            # CANDLESTICK CHART WITH INDICATORS
            # ===========================
            
            st.markdown("### 📈 Price Chart with Indicators")
            
            # Prepare data for chart (last 100 candles for clarity)
            chart_data = df.tail(100)
            
            # Create figure
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            # Add Moving Averages
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['SMA_5'],
                name='SMA 5',
                line=dict(color='blue', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['EMA_9'],
                name='EMA 9',
                line=dict(color='orange', width=1)
            ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(100,100,100,0.5)', width=1),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=chart_data.index,
                y=chart_data['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(100,100,100,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(100,100,100,0.1)',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"{symbol} - 5 Minute Chart (Last 500 minutes)",
                yaxis_title="Price",
                xaxis_title="Time",
                height=500,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ===========================
            # RSI & MACD SUBPLOT
            # ===========================
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### RSI Indicator")
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2)
                ))
                
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                
                fig_rsi.update_layout(height=300, hovermode='x unified', template='plotly_dark')
                fig_rsi.update_yaxes(range=[0, 100])
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                st.markdown("### MACD Indicator")
                fig_macd = go.Figure()
                
                fig_macd.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=2)
                ))
                
                fig_macd.add_trace(go.Scatter(
                    x=chart_data.index,
                    y=chart_data['Signal'],
                    name='Signal',
                    line=dict(color='red', width=2)
                ))
                
                fig_macd.add_trace(go.Bar(
                    x=chart_data.index,
                    y=chart_data['Histogram'],
                    name='Histogram',
                    marker_color=['green' if h > 0 else 'red' for h in chart_data['Histogram']],
                    opacity=0.3
                ))
                
                fig_macd.update_layout(height=300, hovermode='x unified', template='plotly_dark')
                
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # ===========================
            # AUTO REFRESH
            # ===========================
            if auto_refresh:
                import time
                time.sleep(30)
                st.rerun()

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Try refreshing the page or checking your internet connection.")

# ===========================
# FOOTER
# ===========================
st.markdown("---")
st.markdown("""
**⚠️ DISCLAIMER:**
- This bot is for **educational purposes only**
- Past performance does not guarantee future results
- **Always use proper risk management** (never risk more than 2% per trade)
- Set your stop losses and take profits BEFORE entering
- **This is NOT financial advice** - trade at your own risk
- Cryptocurrency and forex markets are highly volatile
- Start with demo/paper trading first
""")
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Live Trading Bot - 5 Min Predictor", layout="wide", initial_sidebar_state="collapsed")

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
