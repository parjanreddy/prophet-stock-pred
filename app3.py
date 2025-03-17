import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Utility Functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Prevent NaN issues

def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

# ✅ Use Streamlit Caching to prevent redundant API calls
@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    try:
        hist = stock.history(start=start_date, end=end_date)
        return hist
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Main Streamlit App
def main():
    st.set_page_config(page_title="Stock Analysis", layout="wide")
    st.title("📊 Simple Stock Analysis")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

    # ✅ Fetch Stock Data using Cached Function
    hist = get_stock_data(symbol, start_date, end_date)

    if hist is None or hist.empty:
        st.warning(f"No data found for {symbol} in the selected date range.")
        return

    # Display Stock Info
    st.subheader(f"Stock Information for {symbol}")

    try:
        latest_data = yf.Ticker(symbol).history(period="1d")
        if not latest_data.empty:
            current_price = latest_data['Close'].iloc[-1]
            st.write(f"**Current Price:** ${current_price:.2f} USD")
        else:
            st.write("Current Price: **Data unavailable**")
    except Exception:
        st.write("Current Price: **Error retrieving data**")

    market_cap = yf.Ticker(symbol).info.get("marketCap", "N/A")
    if isinstance(market_cap, (int, float)):
        st.write(f"**Market Cap:** {market_cap:,}")
    else:
        st.write(f"**Market Cap:** {market_cap}")

    # Plot Stock Price (Candlestick)
    st.subheader(f"{symbol} Price Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close']
    )])
    fig.update_layout(title=f"{symbol} Price Chart", yaxis_title="Price (USD)")
    st.plotly_chart(fig)

    # Calculate Technical Indicators and Plot
    hist = calculate_technical_indicators(hist)

    # Plot Moving Averages
    st.subheader("Moving Averages (SMA 20 & SMA 50)")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close Price"))
    fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="SMA 20"))
    fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name="SMA 50"))
    fig_ma.update_layout(title="Moving Averages")
    st.plotly_chart(fig_ma)

    # Plot RSI
    st.subheader("Relative Strength Index (RSI)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(title="Relative Strength Index")
    st.plotly_chart(fig_rsi)

# ✅ Ensure Proper Script Execution
if __name__ == "__main__":
    main()
fix the error while it deploying in streamlit community by using above code
