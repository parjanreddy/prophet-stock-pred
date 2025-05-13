import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import warnings
import requests
from io import StringIO

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
def fetch_stock_data(symbols, start_date, end_date):
    """Fetch stock data in a single API call and normalize structure."""
    try:
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
        if data.empty:
            return None, None
        # Normalize DataFrame structure
        if isinstance(symbols, str):
            # Single symbol: return flat DataFrame with 'Date', 'Open', 'High', 'Low', 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                data = data[symbols]
            data = data.reset_index()
            # Select only required columns if they exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
            available_columns = [col for col in required_columns if col in data.columns]
            if not available_columns:
                raise ValueError("No expected columns ('Date', 'Open', 'High', 'Low', 'Close') found in data")
            data = data[available_columns]
            info = yf.Ticker(symbols).info
            return data, info
        else:
            # Multiple symbols: return DataFrame with 'Close' prices for each symbol
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
        st.error(f"Error fetching data: {str(e)}")
        return None, None

@st.cache_data
def fetch_portfolio_from_url(url):
    """Fetch and parse portfolio CSV from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Relaxed content-type check: accept common types if content is parseable
        content_type = response.headers.get('content-type', '').lower()
        if not any(ct in content_type for ct in ['text/csv', 'application/octet-stream', 'text/plain']):
            raise ValueError("URL does not point to a CSV file (unsupported content type)")
        # Parse CSV
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        # Validate columns
        required_columns = ['Symbol', 'Weight']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV must contain 'Symbol' and 'Weight' columns")
        # Validate data
        df['Symbol'] = df['Symbol'].str.upper().str.strip()
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        if df['Weight'].isna().any():
            raise ValueError("All weights must be numeric")
        if df['Weight'].sum() <= 0:
            raise ValueError("Sum of weights must be positive")
        # Normalize weights
        df['Weight'] = df['Weight'] / df['Weight'].sum()
        return df[['Symbol', 'Weight']]
    except Exception as e:
        st.error(f"Error fetching portfolio from URL: {str(e)}")
        return None

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
    if 'SMA' in indicators and len(df) >= 50:  # Ensure enough data for SMA_50
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
    if 'RSI' in indicators and len(df) >= 14:  # Ensure enough data for RSI
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

@st.cache_data
def get_recommendations():
    """Get recommended stocks based on different categories."""
    return {
        "Growth Stocks": ["TSLA", "NVDA", "AMD", "SHOP", "SE"],
        "Value Stocks": ["BRK-B", "JNJ", "PG", "VZ", "T"],
        "Dividend Stocks": ["O", "T", "VZ", "PFE", "MO"],
        "Tech Leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "Blue Chips": ["JPM", "V", "WMT", "HD", "DIS"]
    }

@st.cache_data
def get_budget_allocation(budget, stocks, rec_info):
    """Calculate budget allocation for budget-friendly stocks."""
    budget_stocks = [
        s for s in stocks
        if isinstance(rec_info[s].get('currentPrice'), (int, float)) and rec_info[s].get('currentPrice') < 50
    ]
    if not budget_stocks:
        return None
    # Equal allocation across budget-friendly stocks
    allocation_per_stock = budget / len(budget_stocks)
    allocation_data = []
    for s in budget_stocks:
        price = rec_info[s].get('currentPrice')
        shares = np.floor(allocation_per_stock / price) if price > 0 else 0
        allocated_amount = shares * price
        allocation_data.append({
            'Symbol': s,
            'Name': rec_info[s].get('shortName', 'N/A'),
            'Price': f"${price:.2f}" if isinstance(price, (int, float)) else 'N/A',
            'Shares': int(shares),
            'Allocated Amount': f"${allocated_amount:.2f}"
        })
    return pd.DataFrame(allocation_data)

@st.cache_data
def calculate_portfolio_value(data, weights):
    """Calculate weighted portfolio value."""
    weighted_values = data * weights
    portfolio_value = weighted_values.sum(axis=1)
    return portfolio_value

@st.cache_data
def forecast_stock(symbol, hist_data, periods=30):
    """Forecast individual stock price using Prophet."""
    if not prophet_available:
        return None, None
    try:
        df_prophet = hist_data[['Date', 'Close']].reset_index(drop=True).rename(columns={"Date": "ds", "Close": "y"})
        if pd.api.types.is_datetime64tz_dtype(df_prophet['ds']):
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        m = Prophet(daily_seasonality=True)
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return df_prophet, forecast
    except Exception as e:
        st.error(f"Error forecasting {symbol}: {str(e)}")
        return None, None

########################
# Main App
########################

def main():
    st.set_page_config(page_title="Stock & Portfolio Analysis", layout="wide")
    st.title("📊 Stock Analysis & Portfolio Optimization")
    st.markdown("**Note**: Run this app with `streamlit run stock_app.py` to view it in a browser.")
    # Initialize session state
    if 'symbol' not in st.session_state:
        st.session_state.symbol = "AAPL"
    # Sidebar configuration
    st.sidebar.header("Configuration")
    symbol = st.sidebar.text_input("Stock Symbol", value=st.session_state.symbol, key="symbol_input").upper().strip()
    st.session_state.symbol = symbol
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    indicators = st.sidebar.multiselect("Technical Indicators", ["SMA", 'RSI'], ["SMA", 'RSI'])
    num_simulations = st.sidebar.slider("Portfolio Simulations", 500, 2000, 1000)
    budget = st.sidebar.number_input("Investment Budget (USD)", min_value=0.0, value=1000.0, step=100.0)
    portfolio_url = st.sidebar.text_input("Portfolio CSV URL (Optional)", placeholder="https://example.com/portfolio.csv")
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
    if budget <= 0:
        st.error("Investment budget must be greater than zero.")
        return
    # Fetch single stock data
    with st.spinner("Fetching stock data..."):
        hist, info = fetch_stock_data(symbol, start_date, end_date)
    if hist is None:
        st.error(f"No data found for {symbol}. Please check the symbol or date range.")
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
        # Calculate technical indicators before downsampling
        hist = calculate_technical_indicators(hist, indicators)
        # Downsample for plotting if large dataset
        hist_plot = hist.iloc[::5] if len(hist) > 1000 else hist
        # Candlestick chart
        if all(col in hist_plot.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig = go.Figure(go.Candlestick(
                x=hist_plot['Date'], open=hist_plot['Open'], high=hist_plot['High'],
                low=hist_plot['Low'], close=hist_plot['Close'], name=symbol
            ))
            fig.update_layout(title=f"{symbol} Price Chart", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Candlestick chart unavailable due to missing data columns.")
        # Plot moving averages
        if 'SMA' in indicators:
            y_columns = [col for col in ['Close', 'SMA_20', 'SMA_50'] if col in hist_plot.columns and not hist_plot[col].isna().all()]
            if y_columns:
                fig_ma = px.line(hist_plot, x='Date', y=y_columns, title="Moving Averages")
                st.plotly_chart(fig_ma, use_container_width=True)
            else:
                st.warning("Moving averages plot unavailable: insufficient data or missing columns.")
        # Plot RSI
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
            st.warning("Prophet library is not installed. Please install it using `pip install prophet` to enable forecasting.")
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
    # Stock Recommendations
    with st.expander("4. Stock Recommendations"):
        st.write("Here are some recommended stocks based on different investment strategies:")
        recommendations = get_recommendations()
        # Create tabs for different recommendation categories
        tabs = st.tabs(list(recommendations.keys()))
        for tab, (category, stocks) in zip(tabs, recommendations.items()):
            with tab:
                st.subheader(category)
                # Fetch current prices for these stocks
                with st.spinner(f"Fetching data for {category}..."):
                    rec_data, rec_info = fetch_stock_data(stocks, datetime.now() - timedelta(days=7), datetime.now())
                if rec_data is not None and rec_info is not None:
                    # Create a DataFrame with relevant information
                    rec_df = pd.DataFrame([{
                        'Symbol': s,
                        'Name': rec_info[s].get('shortName', 'N/A'),
                        'Price': rec_info[s].get('currentPrice', 'N/A'),
                        'Change (1D)': rec_data[s].pct_change().iloc[-1] * 100 if len(rec_data) > 1 else 'N/A',
                        'Market Cap': rec_info[s].get('marketCap', 'N/A'),
                        'Budget Friendly': 'Yes' if isinstance(rec_info[s].get('currentPrice'), (int, float)) and rec_info[s].get('currentPrice') < 50 else 'No'
                    } for s in stocks])
                    # Format the DataFrame
                    rec_df['Change (1D)'] = rec_df['Change (1D)'].apply(
                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                    )
                    rec_df['Market Cap'] = rec_df['Market Cap'].apply(
                        lambda x: f"${x/1e9:.2f}B" if isinstance(x, (int, float)) else x
                    )
                    rec_df['Price'] = rec_df['Price'].apply(
                        lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x
                    )
                    # Display the table
                    st.table(rec_df)
                    # Add a performance chart
                    if len(rec_data) > 1:
                        fig_rec = px.line(rec_data.set_index('Date'), title=f"{category} Recent Performance")
                        st.plotly_chart(fig_rec, use_container_width=True)
                else:
                    st.warning(f"Could not fetch data for {category} recommendations.")
    # Budget Allocation
    with st.expander("5. Budget Allocation"):
        st.write(f"Recommended allocation of your **${budget:.2f}** budget across budget-friendly stocks (price < $50):")
        # Collect all recommended stocks
        all_recommended_stocks = []
        for stocks in recommendations.values():
            all_recommended_stocks.extend(stocks)
        all_recommended_stocks = list(set(all_recommended_stocks))  # Remove duplicates
        # Fetch data for all recommended stocks
        with st.spinner("Fetching data for budget allocation..."):
            _, rec_info = fetch_stock_data(all_recommended_stocks, datetime.now() - timedelta(days=7), datetime.now())
        if rec_info is not None:
            # Calculate budget allocation
            allocation_df = get_budget_allocation(budget, all_recommended_stocks, rec_info)
            if allocation_df is not None and not allocation_df.empty:
                st.table(allocation_df)
                st.write("**Note**: Allocation is equally distributed across budget-friendly stocks. Shares are rounded down to the nearest whole number.")
            else:
                st.warning("No budget-friendly stocks (price < $50) found for allocation.")
        else:
            st.warning("Could not fetch data for budget allocation.")
    # Portfolio Analysis
    with st.expander("6. Portfolio Performance & Optimization"):
        # Initialize symbols for portfolio analysis
        other_symbols = ["MSFT", "GOOGL", "AMZN"]
        all_symbols = [symbol] + other_symbols
        portfolio_weights = None
        portfolio_source = "Default portfolio"
        
        # Check if portfolio URL is provided
        if portfolio_url:
            portfolio_df = fetch_portfolio_from_url(portfolio_url)
            if portfolio_df is not None:
                all_symbols = portfolio_df['Symbol'].tolist()
                portfolio_weights = portfolio_df['Weight'].values
                portfolio_source = "Uploaded portfolio (URL)"
                st.write(f"**Portfolio Source**: {portfolio_source}")
                st.write("**Portfolio Composition**:")
                st.table(portfolio_df)
        
        # Fetch portfolio data
        with st.spinner("Fetching portfolio data..."):
            combined_data, _ = fetch_stock_data(all_symbols, start_date, end_date)
        if combined_data is None:
            st.error("Failed to fetch portfolio data.")
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
        # Portfolio Metrics
        st.subheader("Portfolio Performance Metrics")
        if portfolio_weights is None:
            # Use equal weights if no portfolio URL is provided
            portfolio_weights = np.array([1/len(all_symbols)] * len(all_symbols))
        weighted_returns = combined_returns.dot(portfolio_weights)
        metrics_dict = compute_portfolio_metrics(weighted_returns)
        plot_portfolio_metrics(metrics_dict)
        # Portfolio Optimization
        st.subheader("Portfolio Optimization")
        with st.spinner("Optimizing portfolio..."):
            results_df, max_sharpe, min_vol = optimize_portfolio(combined_returns, num_portfolios=num_simulations)
        if results_df is not None:
            st.write("**Max Sharpe Portfolio Weights**", max_sharpe[3:] if max_sharpe is not None else "N/A")
            st.write("**Min Volatility Portfolio Weights**", min_vol[3:] if min_vol is not None else "N/A")
            plot_efficient_frontier(results_df, max_sharpe, min_vol)
        else:
            st.warning("Portfolio optimization unavailable due to insufficient data.")
        # Portfolio and Stock Forecasts
        if prophet_available:
            st.subheader("Portfolio and Stock Forecasts with Prophet")
            # Portfolio Value Forecast
            try:
                # Calculate portfolio value
                portfolio_value = calculate_portfolio_value(combined_df, portfolio_weights)
                df_prophet = pd.DataFrame({
                    'ds': portfolio_value.index,
                    'y': portfolio_value.values
                })
                if pd.api.types.is_datetime64tz_dtype(df_prophet['ds']):
                    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                with st.spinner("Generating portfolio forecast..."):
                    m = Prophet(daily_seasonality=True)
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=30)
                    forecast = m.predict(future)
                fig_forecast = px.line(df_prophet, x='ds', y='y', title="30-Day Portfolio Value Forecast")
                fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast")
                fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], line=dict(dash='dash'), name="Upper Bound")
                fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], line=dict(dash='dash'), name="Lower Bound")
                fig_forecast.update_layout(yaxis_title="Portfolio Value (USD)")
                st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"Error in portfolio forecasting: {str(e)}")
            # Individual Stock Forecasts
            st.subheader("Individual Stock Forecasts")
            tabs = st.tabs(all_symbols)
            for tab, s in zip(tabs, all_symbols):
                with tab:
                    with st.spinner(f"Fetching and forecasting data for {s}..."):
                        stock_hist, _ = fetch_stock_data(s, start_date, end_date)
                    if stock_hist is not None:
                        df_stock, stock_forecast = forecast_stock(s, stock_hist)
                        if df_stock is not None and stock_forecast is not None:
                            fig_stock = px.line(df_stock, x='ds', y='y', title=f"30-Day Price Forecast for {s}")
                            fig_stock.add_scatter(x=stock_forecast['ds'], y=stock_forecast['yhat'], name="Forecast")
                            fig_stock.add_scatter(x=stock_forecast['ds'], y=stock_forecast['yhat_upper'], line=dict(dash='dash'), name="Upper Bound")
                            fig_stock.add_scatter(x=stock_forecast['ds'], y=stock_forecast['yhat_lower'], line=dict(dash='dash'), name="Lower Bound")
                            fig_stock.update_layout(yaxis_title="Price (USD)")
                            st.plotly_chart(fig_stock, use_container_width=True)
                        else:
                            st.warning(f"Forecast unavailable for {s}.")
                    else:
                        st.warning(f"Could not fetch data for {s}.")
        else:
            st.warning("Prophet library is not installed. Please install it using `pip install prophet` to enable portfolio and stock forecasting.")
    # Performance Evaluation
    with st.expander("7. Performance Evaluation & Comparison"):
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
