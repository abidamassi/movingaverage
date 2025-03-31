import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# --- Streamlit Dashboard Config ---
st.set_page_config(page_title="Stock Forecast & Moving Average — Finance Modeling", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #050915;
        color: #E1E6ED;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #050915;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stSidebar {
        background-color: #0a3d62;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar p, .stSidebar label, .stSidebar span {
        color: white !important;
    }
    h1 {
        font-size: 26px !important;
        color: #F0F4F8;
    }
    h4 {
        font-size: 18px !important;
        color: #F0F4F8;
    }
    .metric-box {
        background-color: #2c3e50;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #2f3640;
        padding: 1rem;
        border-radius: 10px;
        font-size: 14px;
        color: #f1c40f;
        margin-top: 2rem;
    }
    .footer-text {
        margin: 3rem auto 1rem;
        text-align: center;
        font-size: 17px;
        font-weight: bold;
        color: white;
    }
    @media (max-width: 768px) {
        .metric-box { font-size: 14px; padding: 0.8rem; }
        h1 { font-size: 20px !important; }
        .stSidebar::before {
            display: none;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.header("🧮 Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (Yahoo Finance Format)", value="BBCA.JK")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-12-12"))
end_date = st.sidebar.date_input("End Date", value=datetime.today())
short_window = st.sidebar.number_input("Short MA", min_value=5, max_value=100, value=50)
long_window = st.sidebar.number_input("Long MA", min_value=20, max_value=500, value=200)
forecast_days = st.sidebar.slider("Forecast Days Ahead", min_value=7, max_value=60, value=30)

# --- Header ---
st.title("📊 Athaya Forecast & Moving Average")
st.markdown("""<hr style='margin-top:0; border-color:#34495e;'>""", unsafe_allow_html=True)

# --- Load Data ---
st.markdown(f"<h4>📈 Stock Price Data: {ticker}</h4>", unsafe_allow_html=True)
data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("❌ No data found. Please check the stock ticker and date range.")
    st.stop()

# --- Preprocessing ---
df = data.reset_index()[['Date', 'Close']].dropna()
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df.dropna(inplace=True)

# --- Calculate Moving Averages ---
df['MA_Short'] = df['y'].rolling(window=short_window).mean()
df['MA_Long'] = df['y'].rolling(window=long_window).mean()
df['Signal'] = 0
df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
df['Position'] = df['Signal'].diff()

# --- ARIMA Forecasting (Tuned for more fluctuation) ---
model = ARIMA(df['y'], order=(8, 1, 5))
model_fit = model.fit()
forecast_values = model_fit.forecast(steps=forecast_days)
forecast_index = pd.date_range(start=df['ds'].iloc[-1], periods=forecast_days+1, freq='B')[1:]
forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast_values})

# --- Metrics Box Layout ---
latest = df.iloc[-1]
current_price = latest['y']
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.markdown(f"""<div class='metric-box'>📌 Current Price<br>{current_price:,.2f}</div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='metric-box'>📈 MA{short_window}<br>{latest['MA_Short']:,.2f}</div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='metric-box'>📉 MA{long_window}<br>{latest['MA_Long']:,.2f}</div>""", unsafe_allow_html=True)

# --- Signal Message ---
if latest['Signal'] == 1:
    st.success("🚀 Signal: BUY (Short MA > Long MA)")
else:
    st.warning("🔻 Signal: SELL / HOLD (Short MA <= Long MA)")

# --- Buy & Sell Signals ---
buy_signals = df[df['Position'] == 1].copy()
sell_signals = df[df['Position'] == -1].copy()

# --- Plotly Chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast',
                         line=dict(color='#00aaff', width=3)))
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Price', line=dict(color='#dddddd', width=1)))
fig.add_trace(go.Scatter(x=df['ds'], y=df['MA_Short'], mode='lines', name=f'MA{short_window}', line=dict(color='#7ed6df')))
fig.add_trace(go.Scatter(x=df['ds'], y=df['MA_Long'], mode='lines', name=f'MA{long_window}', line=dict(color='#ff7675')))
fig.add_trace(go.Scatter(x=[latest['ds']], y=[latest['y']], mode='markers+text', name='Latest Price',
                         marker=dict(size=10, color='#f9ca24', symbol='circle', line=dict(width=2, color='white')),
                         text=["Latest"], textposition="top right"))
fig.add_trace(go.Scatter(x=buy_signals['ds'], y=buy_signals['y'], mode='markers', name='Buy Signal',
                         marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(color='white', width=1.5))))
fig.add_trace(go.Scatter(x=sell_signals['ds'], y=sell_signals['y'], mode='markers', name='Sell Signal',
                         marker=dict(symbol='triangle-down', size=12, color='red', line=dict(color='white', width=1.5))))
fig.update_layout(
    plot_bgcolor='#050915',
    paper_bgcolor='#050915',
    font=dict(color='white'),
    legend=dict(orientation="h", yanchor="top", y=1, xanchor="left", x=0),
    xaxis_title='',
    yaxis_title='Price',
    margin=dict(l=30, r=30, t=30, b=30),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# --- Analysis Notes ---
st.markdown("""
### 📌 Analysis Notes

Based on the crossover of MA{} and MA{}:
- When the **short-term MA (MA{})** crosses **above** the long-term MA (MA{}), it's a **BUY signal**.
- When the **short-term MA** crosses **below** the long-term MA, it's a **SELL/HOLD signal**.

**Current Decision:**
Since MA{} is **{}** MA{}, the signal is **{}**.

---

### 📊 Forecasting Method

This dashboard uses **ARIMA** — a statistical time-series forecasting method.

🧠 **How ARIMA works:**
- ARIMA stands for Autoregressive Integrated Moving Average.
- It uses past price data and the trend to estimate future values.
- The model is based on three components: AR (autoregression), I (differencing), and MA (moving average).
- It works best on stationary time series with consistent trend and variance over time.

---

### 🔁 Backtesting Results

Backtesting simulates how the strategy would have performed in the past using historical data.
""".format(
    short_window, long_window,
    short_window, long_window,
    short_window, "above" if latest['MA_Short'] > latest['MA_Long'] else "below", long_window,
    "BUY" if latest['MA_Short'] > latest['MA_Long'] else "SELL / HOLD"
))

# --- Backtesting ---
position = 0.0
cash = 10000000
buy_price = 0.0
trade_log = []

for i in range(1, len(df)):
    if df['Position'].iloc[i] == 1 and position == 0:
        buy_price = df['y'].iloc[i]
        position = cash / buy_price
        cash = 0
        trade_log.append({'Date': df['ds'].iloc[i], 'Action': 'Buy', 'Price': buy_price})
    elif df['Position'].iloc[i] == -1 and position > 0:
        sell_price = df['y'].iloc[i]
        cash = position * sell_price
        position = 0
        trade_log.append({'Date': df['ds'].iloc[i], 'Action': 'Sell', 'Price': sell_price})

final_value = cash + position * df['y'].iloc[-1]
profit = final_value - 10000000
roi = profit / 10000000 * 100

if trade_log:
    st.markdown(f"""
    - **Initial Capital:** 10,000,000
    - **Final Portfolio Value:** {final_value:,.0f}
    - **Total Profit:** {profit:,.0f} ({roi:.2f}%)
    """)
    trade_df = pd.DataFrame(trade_log)
    trade_df['Date'] = trade_df['Date'].dt.strftime('%Y-%m-%d')
    st.write("**Trade Log:**")
    st.dataframe(trade_df)
else:
    st.info("No buy/sell signals were generated in the given time range.")

# --- Disclaimer ---
st.markdown("""
<div class='disclaimer-box'>
⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. Please do your own research before making any investment decisions.
</div>
""", unsafe_allow_html=True)

# --- Real Footer ---
st.markdown("""
<div class='footer-text'>Created by Abida Massi<br></div>
""", unsafe_allow_html=True)
