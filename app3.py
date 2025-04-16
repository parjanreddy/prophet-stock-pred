import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

########################
# Utility Functions
########################

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_technical_indicators(df):
    """Add common technical indicators to the DataFrame."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

def compute_portfolio_metrics(returns, risk_free_rate=0.01):
    """Compute portfolio metrics: annual return, volatility, Sharpe ratio, etc."""
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan
    var_95 = returns.quantile(0.05)
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        '95% VaR': var_95
    }

def plot_portfolio_metrics(metrics_dict):
    """Display metrics in a Streamlit-friendly format."""
    cols = st.columns(3)
    for i, (k, v) in enumerate(metrics_dict.items()):
        cols[i % 3].metric(k, f"{v*100:.2f}%" if isinstance(v, float) else f"{v:.2f}")

def optimize_portfolio(returns_df, num_portfolios=2000, risk_free_rate=0.01):
    """Monte Carlo approach to portfolio optimization."""
    np.random.seed(42)
    num_assets = len(returns_df.columns)
    results = np.zeros((num_portfolios, 3 + num_assets))
    cov_matrix = returns_df.cov() * 252
    mean_returns = returns_df.mean() * 252
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        results[i, :3] = [portfolio_volatility, portfolio_return, sharpe_ratio]
        results[i, 3:] = weights
    
    columns = ['Volatility', 'Return', 'Sharpe'] + list(returns_df.columns)
    results_df = pd.DataFrame(results, columns=columns)
    max_sharpe = results_df.iloc[results_df['Sharpe'].idxmax()]
    min_vol = results_df.iloc[results_df['Volatility'].idxmin()]
    return results_df, max_sharpe, min_vol

def plot_efficient_frontier(results_df, max_sharpe, min_vol):
    """Plot the efficient frontier."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis')
    ax.scatter(max_sharpe['Volatility'], max_sharpe['Return'], marker='*', color='r', s=300, label='Max Sharpe')
    ax.scatter(min_vol['Volatility'], min_vol['Return'], marker='*', color='g', s=300, label='Min Volatility')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()
    ax.set_title("Efficient Frontier")
    st.pyplot(fig)

########################
# Main App
########################

def main():
    st.set_page_config(page_title="Stock & Portfolio Analysis", layout="wide")
    st.title("📊 Stock Analysis & Portfolio Optimization")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    # Fetch data
    st.subheader("Stock Information")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}. Error: {e}")
        return

    metrics_display = {
        "Current Price": info.get('currentPrice', 'N/A'),
        "Market Cap": info.get('marketCap', 'N/A'),
        "Forward P/E": info.get('forwardPE', 'N/A'),
        "52w High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "52w Low": info.get('fiftyTwoWeekLow', 'N/A')
    }
    cols = st.columns(5)
    for col, (k, v) in zip(cols, metrics_display.items()):
        col.metric(k, v)
    st.write("**Business Summary:**")
    st.write(info.get('longBusinessSummary', 'No information available'))

    # Historical Data & Technicals
    st.header("1. Technical Analysis")
    try:
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            st.warning("No data found for the given symbol and date range.")
            return
    except Exception as e:
        st.error(f"Failed to fetch historical data. Error: {e}")
        return

    # Candle chart
    fig = go.Figure(go.Candlestick(
        x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name=symbol
    ))
    fig.update_layout(title=f"{symbol} Price Chart", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # Calculate technical indicators
    hist = calculate_technical_indicators(hist)

    # Plot Price + SMA
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name="Close"))
    fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['SMA_20'], name="SMA 20"))
    fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name="SMA 50"))
    fig_ma.update_layout(title="Moving Averages")
    st.plotly_chart(fig_ma, use_container_width=True)

    # Plot RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title="Relative Strength Index")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # Prophet Forecast
    st.header("2. Price Forecast with Prophet")
    df_prophet = hist[['Close']].reset_index().rename(columns={"Date": "ds", "Close": "y"})
    if pd.api.types.is_datetime64tz_dtype(df_prophet['ds']):
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name="Actual"))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast"))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], line=dict(dash='dash'), name="Upper Bound"))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], line=dict(dash='dash'), name="Lower Bound"))
    fig_forecast.update_layout(title="30-Day Price Forecast")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Profit Estimation
    st.header("3. Profit Estimation (Simple Buy & Hold)")
    first_price = hist['Close'].iloc[0]
    last_price = hist['Close'].iloc[-1]
    profit_pct = (last_price - first_price) / first_price * 100
    st.write(f"If you **bought {symbol}** on **{start_date}** at a price of **{first_price:.2f} USD** "
             f"and sold on **{end_date}** at **{last_price:.2f} USD**, "
             f"your total return would be approximately **{profit_pct:.2f}%**.")

    # Portfolio Performance & Optimization
    st.header("4. Portfolio Performance & Optimization")
    other_symbols = ["MSFT", "GOOGL", "AMZN"]
    all_symbols = [symbol] + other_symbols
    combined_df = pd.DataFrame()

    for s in all_symbols:
        try:
            df_temp = yf.download(s, start=start_date, end=end_date)
            if not df_temp.empty:
                combined_df[s] = df_temp['Close']
        except Exception as e:
            st.warning(f"Could not fetch data for {s}: {str(e)}")

    if combined_df.empty:
        st.error("Failed to fetch data for any of the selected stocks.")
        return

    combined_returns = combined_df.pct_change().dropna()
    st.write("**Portfolio Stocks:**", all_symbols)
    st.line_chart(combined_df, use_container_width=True)

    # Portfolio Performance Metrics
    st.subheader("Portfolio Performance Metrics")
    equal_weights = np.array([1/len(all_symbols)] * len(all_symbols))
    weighted_returns = combined_returns.dot(equal_weights)
    metrics_dict = compute_portfolio_metrics(weighted_returns)
    plot_portfolio_metrics(metrics_dict)

    # Portfolio Optimization
    st.subheader("Portfolio Optimization (Monte Carlo Simulation)")
    results_df, max_sharpe, min_vol = optimize_portfolio(combined_returns)
    st.write("**Max Sharpe Portfolio Weights**")
    st.write(max_sharpe[3:])
    st.write("**Min Volatility Portfolio Weights**")
    st.write(min_vol[3:])
    st.write("**Efficient Frontier**")
    plot_efficient_frontier(results_df, max_sharpe, min_vol)

    # Performance Evaluation
    st.header("5. Performance Evaluation & Comparison")
    cumulative_returns_df = (1 + combined_returns).cumprod()
    fig_cumulative = go.Figure()
    for col in cumulative_returns_df.columns:
        fig_cumulative.add_trace(go.Scatter(x=cumulative_returns_df.index, y=cumulative_returns_df[col], mode='lines', name=col))
    fig_cumulative.update_layout(title="Cumulative Returns Comparison", xaxis_title="Date", yaxis_title="Cumulative Returns (1 = 100%)")
    st.plotly_chart(fig_cumulative, use_container_width=True)
    st.write("**Note**: This chart helps you see how each stock performed relative to each other over time.")

if __name__ == "__main__":
    main()
