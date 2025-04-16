import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for prophet availability
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

# Suppress warnings
warnings.filterwarnings("ignore")

########################
# Utility Functions
########################

@st.cache_data
def fetch_stock_data(symbols, start_date, end_date, retries=5, delay=10):
    """Fetch stock data with retries and future date handling."""
    if end_date > datetime.now().date():
        end_date = datetime.now().date()
        st.warning("End date adjusted to today as future dates are not available.")
    
    for attempt in range(retries):
        try:
            data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker', progress=False)
            if data.empty:
                logger.warning(f"No data returned for {symbols} on attempt {attempt + 1}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
                return None, None
            
            # Normalize DataFrame structure
            if isinstance(symbols, str):
                # Single symbol: return flat DataFrame
                if isinstance(data.columns, pd.MultiIndex):
                    data = data[symbols]
                data = data.reset_index()
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
                available_columns = [col for col in required_columns if col in data.columns]
                if not available_columns:
                    raise ValueError("No expected columns found in data")
                data = data[available_columns]
                info = yf.Ticker(symbols).info
                return data, info
            else:
                # Multiple symbols: return Close prices
                close_data = pd.DataFrame()
                for s in symbols:
                    if isinstance(data.columns, pd.MultiIndex) and s in data.columns.levels[0]:
                        close_data[s] = data[s]['Close']
                    elif s in data.columns:
                        close_data[s] = data[s]
                if close_data.empty:
                    raise ValueError("No valid data found for the provided symbols")
                close_data = close_data.reset_index()
                info = {s: yf.Ticker(s).info for s in symbols}
                return close_data, info
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {symbols}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                st.error(f"No data found for {symbols}. Please check the symbol or try again later. Error: {str(e)}")
                logger.warning("Using mock data due to fetch failure")
                data = pd.DataFrame({
                    'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
                    'Open': np.random.uniform(100, 200, size=(end_date - start_date).days),
                    'High': np.random.uniform(100, 200, size=(end_date - start_date).days),
                    'Low': np.random.uniform(100, 200, size=(end_date - start_date).days),
                    'Close': np.random.uniform(100, 200, size=(end_date - start_date).days)
                })
                return data, {"currentPrice": 150}  # Mock info
    return None, None  # Fallback in case loop exits unexpectedly

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_technical_indicators(df, indicators=['SMA', 'RSI']):
    """Add selected technical indicators to the DataFrame."""
    df = df.copy()
    if 'SMA' in indicators and len(df) >= 50:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
    if 'RSI' in indicators and len(df) >= 14:
        df['RSI'] = calculate_rsi(df['Close'])
    return df

@st.cache_data
def compute_portfolio_metrics(returns, risk_free_rate=0.01):
    """Compute portfolio metrics."""
    if returns.empty:
        return None
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252) if not negative_returns.empty else np.nan
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan
    var_95 = returns.quantile(0.05) if not returns.empty else np.nan
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / rolling_max) - 1
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        '95% VaR': var_95
    }

def plot_portfolio_metrics(metrics_dict):
    """Display metrics in a table."""
    if metrics_dict is None:
        st.warning("Portfolio metrics unavailable due to insufficient data.")
        return
    metrics_df = pd.DataFrame(metrics_dict.items(), columns=["Metric", "Value"])
    metrics_df["Value"] = metrics_df["Value"].apply(
        lambda x: f"{x*100:.2f}%" if isinstance(x, float) and not pd.isna(x) else "N/A"
    )
    st.table(metrics_df)

@st.cache_data
def optimize_portfolio(returns_df, num_portfolios=1000, risk_free_rate=0.01):
    """Monte Carlo approach to portfolio optimization."""
    if returns_df.empty or len(returns_df.columns) < 1:
        return None, None, None
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
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else np.nan
        results[i, :3] = [portfolio_volatility, portfolio_return, sharpe_ratio]
        results[i, 3:] = weights
    
    columns = ['Volatility', 'Return', 'Sharpe'] + list(returns_df.columns)
    results_df = pd.DataFrame(results, columns=columns)
    max_sharpe = results_df.iloc[results_df['Sharpe'].idxmax()] if not results_df['Sharpe'].isna().all() else None
    min_vol = results_df.iloc[results_df['Volatility'].idxmin()] if not results_df['Volatility'].isna().all() else None
    return results_df, max_sharpe, min_vol

def plot_efficient_frontier(results_df, max_sharpe, min_vol):
    """Plot the efficient frontier using Plotly."""
    if results_df is None or max_sharpe is None or min_vol is None:
        st.warning("Efficient frontier plot unavailable due to insufficient data.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df['Volatility'], y=results_df['Return'], mode='markers',
        marker=dict(color=results_df['Sharpe'], colorscale='Viridis'), name='Portfolios'
    ))
    fig.add_trace(go.Scatter(
        x=[max_sharpe['Volatility']], y=[max_sharpe['Return']], mode='markers',
        marker=dict(color='red', size=15, symbol='star'), name='Max Sharpe'
    ))
    fig.add_trace(go.Scatter(
        x=[min_vol['Volatility']], y=[min_vol['Return']], mode='markers',
        marker=dict(color='green', size=15, symbol='star'), name='Min Volatility'
    ))
    fig.update_layout(
        title="Efficient Frontier", xaxis_title="Volatility", yaxis_title="Return",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

########################
# Main App
########################

def main():
    st.set_page_config(page_title="Stock & Portfolio Analysis", layout="wide")
    st.title("📊 Stock Analysis & Portfolio Optimization")
    st.markdown("**Note**: Run this app with `streamlit run stock_app.py` for local testing.")

    # Initialize session state
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "AAPL"

    # Sidebar configuration
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Stock Symbol", value=st.session_state.symbol, key="symbol_input").upper().strip()
    st.session_state.symbol = symbol
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    indicators = st.sidebar.multiselect("Technical Indicators", ["SMA", "RSI"], ["SMA", "RSI"])
    num_simulations = st.sidebar.slider("Portfolio Simulations", 500, 2000, 1000)

    # Input validation
    if not symbol:
        st.error("Please enter a valid stock symbol.")
        return
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return
    if end_date > datetime.now().date():
        st.error("End date cannot be in the future.")
        return

    # Fetch single stock data
    with st.spinner("Fetching stock data..."):
        hist, info = fetch_stock_data(symbol, start_date, end_date)
    if hist is None:
        st.error(f"No data found for {symbol}. Please check the symbol or try again later.")
        return

    # Stock Information
    st.subheader("Stock Information")
    metrics_display = {
        "Current Price": info.get('currentPrice', 'N/A'),
        "Market Cap": info.get('marketCap', 'N/A'),
        "Forward P/E": info.get('forwardPE', 'N/A'),
        "52w High": info.get('fiftyTwoWeekHigh', 'N/A'),
        "52w Low": info.get('fiftyTwoWeekLow', 'N/A')
    }
    st.table(pd.DataFrame(metrics_display.items(), columns=["Metric", "Value"]))
    with st.expander("Business Summary"):
        st.write(info.get('longBusinessSummary', 'No information available'))

    # Technical Analysis
    with st.expander("1. Technical Analysis", expanded=True):
        hist = calculate_technical_indicators(hist, indicators)
        hist_plot = hist.iloc[::5] if len(hist) > 1000 else hist

        if all(col in hist_plot.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig = go.Figure(go.Candlestick(
                x=hist_plot['Date'], open=hist_plot['Open'], high=hist_plot['High'],
                low=hist_plot['Low'], close=hist_plot['Close'], name=symbol
            ))
            fig.update_layout(title=f"{symbol} Price Chart", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Candlestick chart unavailable due to missing data columns.")

        if 'SMA' in indicators:
            y_columns = [col for col in ['Close', 'SMA_20', 'SMA_50'] if col in hist_plot.columns and not hist_plot[col].isna().all()]
            if y_columns:
                fig_ma = px.line(hist_plot, x='Date', y=y_columns, title="Moving Averages")
                st.plotly_chart(fig_ma, use_container_width=True)
            else:
                st.warning("Moving averages plot unavailable: insufficient data or missing columns.")

        if 'RSI' in indicators and 'RSI' in hist_plot.columns and not hist_plot['RSI'].isna().all():
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=hist_plot['Date'], y=hist_plot['RSI'], name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="Relative Strength Index")
            st.plotly_chart(fig_rsi, use_container_width=True)
        elif 'RSI' in indicators:
            st.warning("RSI plot unavailable: insufficient data or missing RSI column.")

    # Price Forecast
    with st.expander("2. Price Forecast with Prophet"):
        if not prophet_available:
            st.warning("Prophet library is not installed. Please install it to enable forecasting.")
        else:
            try:
                df_prophet = hist[['Date', 'Close']].reset_index(drop=True).rename(columns={"Date": "ds", "Close": "y"})
                if pd.api.types.is_datetime64tz_dtype(df_prophet['ds']):
                    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                with st.spinner("Generating forecast..."):
                    m = Prophet(daily_seasonality=True)
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=30)
                    forecast = m.predict(future)

                fig_forecast = px.line(df_prophet, x='ds', y='y', title="30-Day Price Forecast")
                fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast")
                fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], line=dict(dash='dash'), name="Upper Bound")
                fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], line=dict(dash='dash'), name="Lower Bound")
                st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"Error in forecasting: {str(e)}")

    # Profit Estimation
    with st.expander("3. Profit Estimation (Buy & Hold)"):
        if len(hist) >= 2:
            first_price = hist['Close'].iloc[0]
            last_price = hist['Close'].iloc[-1]
            profit_pct = (last_price - first_price) / first_price * 100
            st.write(
                f"If you **bought {symbol}** on **{start_date}** at **{first_price:.2f} USD** "
                f"and sold on **{end_date}** at **{last_price:.2f} USD**, "
                f"your return would be **{profit_pct:.2f}%**."
            )
        else:
            st.warning("Insufficient data for profit estimation.")

    # Portfolio Analysis
    with st.expander("4. Portfolio Performance & Optimization"):
        other_symbols = ["MSFT", "GOOGL", "AMZN"]
        all_symbols = [symbol] + other_symbols
        with st.spinner("Fetching portfolio data..."):
            combined_data, _ = fetch_stock_data(all_symbols, start_date, end_date)
        if combined_data is None:
            st.error("Failed to fetch portfolio data for some symbols.")
            return

        combined_df = combined_data.set_index('Date').dropna()
        if combined_df.empty or len(combined_df) < 2:
            st.error("No valid portfolio data available.")
            return

        combined_returns = combined_df.pct_change().dropna()
        if combined_returns.empty:
            st.error("Insufficient data for portfolio calculations.")
            return

        st.write("**Portfolio Stocks:**", all_symbols)
        st.line_chart(combined_df, use_container_width=True)

        st.subheader("Portfolio Performance Metrics")
        equal_weights = np.array([1/len(all_symbols)] * len(all_symbols))
        weighted_returns = combined_returns.dot(equal_weights)
        metrics_dict = compute_portfolio_metrics(weighted_returns)
        plot_portfolio_metrics(metrics_dict)

        st.subheader("Portfolio Optimization")
        with st.spinner("Optimizing portfolio..."):
            results_df, max_sharpe, min_vol = optimize_portfolio(combined_returns, num_portfolios=num_simulations)
        if results_df is not None:
            st.write("**Max Sharpe Portfolio Weights**", max_sharpe[3:] if max_sharpe is not None else "N/A")
            st.write("**Min Volatility Portfolio Weights**", min_vol[3:] if min_vol is not None else "N/A")
            plot_efficient_frontier(results_df, max_sharpe, min_vol)
        else:
            st.warning("Portfolio optimization unavailable due to insufficient data.")

    # Performance Evaluation
    with st.expander("5. Performance Evaluation & Comparison"):
        if not combined_returns.empty:
            cumulative_returns_df = (1 + combined_returns).cumprod()
            fig_cumulative = px.line(
                cumulative_returns_df, x=cumulative_returns_df.index, y=cumulative_returns_df.columns,
                title="Cumulative Returns Comparison"
            )
            fig_cumulative.update_layout(yaxis_title="Cumulative Returns (1 = 100%)")
            st.plotly_chart(fig_cumulative, use_container_width=True)
            st.write("**Note**: Shows relative performance of portfolio stocks over time.")
        else:
            st.warning("Cumulative returns comparison unavailable due to insufficient data.")

if __name__ == "__main__":
    main()
