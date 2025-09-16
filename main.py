"""
Crypto Backtester - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import our modules
from data_handler import DataHandler
from strategies import TrendFollowingStrategy, MeanReversionStrategy
from backtester import Backtester
from performance import PerformanceAnalyzer
from visualizations import Visualizer

# Page configuration
st.set_page_config(
    page_title="Crypto Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    border: none;
    padding: 0.5rem 1rem;
    width: 100%;
}
.stButton>button:hover {
    background-color: #145a8d;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🪙Crypto Backtesting Platform")
st.markdown("Backtest trend following and mean reversion strategies on Bitcoin and Ethereum")

# Sidebar for user inputs
st.sidebar.header("⚙️Backtest Settings")

# Basic Settings
st.sidebar.subheader("Basic Settings")

# Initial Capital
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Starting capital for the backtest"
)

# Coin Selection
coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    options=["BTC", "ETH"],
    help="Choose which cryptocurrency to backtest"
)

# Strategy Selection
strategy_type = st.sidebar.selectbox(
    "Select Strategy",
    options=["Trend Following", "Mean Reversion"],
    help="Choose the trading strategy"
)

# Date Range
st.sidebar.subheader("Date Range")

# Default date range (2 years)
default_end = datetime.now().date()
default_start = default_end - timedelta(days=730)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=default_start,
        min_value=datetime(2018, 1, 1).date(),
        max_value=default_end,
        help="Start date for backtest"
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
        max_value=default_end,
        help="End date for backtest"
    )

# Advanced Settings (Expandable)
with st.sidebar.expander("📊 Advanced Settings", expanded=False):
    st.subheader("Strategy Parameters")
    
    if strategy_type == "Trend Following":
        sma_short = st.number_input("SMA Short Period", value=50, min_value=10, max_value=100)
        sma_long = st.number_input("SMA Long Period", value=200, min_value=100, max_value=300)
        strategy_params = {'sma_short': sma_short, 'sma_long': sma_long}
    else:  # Mean Reversion
        rsi_period = st.number_input("RSI Period", value=14, min_value=5, max_value=30)
        rsi_buy = st.number_input("RSI Buy Threshold", value=30, min_value=10, max_value=40)
        rsi_exit = st.number_input("RSI Exit Threshold", value=50, min_value=40, max_value=70)
        strategy_params = {
            'rsi_period': rsi_period,
            'buy_threshold': rsi_buy,
            'exit_threshold': rsi_exit
        }
    
    st.subheader("Trading Costs")
    trading_fee = st.number_input(
        "Trading Fee (%)",
        value=0.1,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Percentage fee per trade"
    ) / 100
    
    slippage = st.number_input(
        "Slippage (%)",
        value=0.05,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Expected slippage per trade"
    ) / 100

# Run Backtest Button
run_backtest = st.sidebar.button("🚀 Run Backtest", use_container_width=True)

# Main content area
if run_backtest:
    # Date validation
    days_difference = (end_date - start_date).days
    
    min_days_required = {
        "Trend Following": 200,
        "Mean Reversion": 14
    }
    
    required_days = min_days_required[strategy_type]
    
    if days_difference < required_days:
        st.error(f"""
        ❌ **Date range too short!**
        
        You selected {days_difference} days, but {strategy_type} needs at least {required_days} days.
        
        **Why?** {strategy_type} uses indicators that need historical data:
        - Trend Following: Uses 200-day moving average
        - Mean Reversion: Uses 14-day RSI
        
        Please select a date range of at least {required_days} days.
        """)
        st.stop()
    
    try:
        # Show loading spinner
        with st.spinner(f"Running backtest for {coin} using {strategy_type} strategy..."):
            
            # Initialize components
            data_handler = DataHandler()
            visualizer = Visualizer()
            analyzer = PerformanceAnalyzer()
            
            # Fetch data
            st.info(f"📊 Fetching {coin} data from {start_date} to {end_date}...")
            price_data = data_handler.fetch_data(
                coin, 
                pd.Timestamp(start_date), 
                pd.Timestamp(end_date)
            )
            
            # Add indicators based on strategy
            if strategy_type == "Trend Following":
                data_with_indicators = data_handler.add_indicators(price_data, 'trend')
                strategy = TrendFollowingStrategy(
                    sma_short=strategy_params.get('sma_short', 50),
                    sma_long=strategy_params.get('sma_long', 200)
                )
            else:  # Mean Reversion
                data_with_indicators = data_handler.add_indicators(price_data, 'mean_reversion')
                strategy = MeanReversionStrategy(
                    rsi_period=strategy_params.get('rsi_period', 14),
                    buy_threshold=strategy_params.get('buy_threshold', 30),
                    exit_threshold=strategy_params.get('exit_threshold', 50)
                )
            
            # Generate signals
            data_with_signals = strategy.generate_signals(data_with_indicators)
            
            # Run backtest
            backtester = Backtester(
                initial_capital=initial_capital,
                trading_fee=trading_fee,
                slippage=slippage
            )
            results, trades = backtester.run_backtest(data_with_signals)
            
            # Calculate performance metrics
            metrics, drawdown_series = analyzer.calculate_metrics(results, initial_capital)
            
            # Calculate buy & hold comparison
            buy_hold = backtester.calculate_buy_and_hold(data_with_signals)
            buy_hold_return = (buy_hold.iloc[-1] / initial_capital - 1) * 100
            
        # Display Results
        st.success("✅ Backtest completed successfully!")
        
        # Summary Metrics Row
        st.markdown("### 📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics['Total Return (%)']}%",
                delta=f"{metrics['Total Return (%)'] - buy_hold_return:.2f}% vs Buy & Hold"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['Sharpe Ratio']}",
                help="Risk-adjusted returns (higher is better)"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics['Max Drawdown (%)']}%",
                help="Largest peak-to-trough decline"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{metrics['Win Rate (%)']}%",
                help="Percentage of profitable days"
            )
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Equity Curve", 
            "📉 Drawdown", 
            "📊 Returns Distribution",
            "🎯 Signals & Price",
            "📋 Detailed Metrics"
        ])
        
        with tab1:
            # Equity Curve
            equity_fig = visualizer.plot_equity_curve(results, strategy.name, coin)
            st.plotly_chart(equity_fig, use_container_width=True)
            
            # Add comparison with buy & hold
            st.markdown("#### Strategy vs Buy & Hold Comparison")
            comparison_data = pd.DataFrame({
                'Strategy': results['portfolio_value'],
                'Buy & Hold': buy_hold
            })
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                x=comparison_data.index,
                y=comparison_data['Strategy'],
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ))
            fig_comparison.add_trace(go.Scatter(
                x=comparison_data.index,
                y=comparison_data['Buy & Hold'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=2, dash='dash')
            ))
            fig_comparison.update_layout(
                title='Strategy vs Buy & Hold Performance',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with tab2:
            # Drawdown Chart
            drawdown_fig = visualizer.plot_drawdown(drawdown_series, strategy.name, coin)
            st.plotly_chart(drawdown_fig, use_container_width=True)
        
        with tab3:
            # Returns Distribution
            returns_fig = visualizer.plot_returns_distribution(results['returns'], strategy.name, coin)
            st.plotly_chart(returns_fig, use_container_width=True)
        
        with tab4:
            # Price and Signals
            signals_fig = visualizer.plot_price_and_signals(data_with_signals, strategy.name, coin)
            st.plotly_chart(signals_fig, use_container_width=True)
        
        with tab5:
            # Detailed Metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Performance Metrics")
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                st.dataframe(metrics_df, use_container_width=True)
            
            with col2:
                st.markdown("#### Trade Log")
                if not trades.empty:
                    trades_display = trades.copy()
                    trades_display['date'] = pd.to_datetime(trades_display['date']).dt.date
                    trades_display['price'] = trades_display['price'].round(2)
                    trades_display['value'] = trades_display['value'].round(2)
                    trades_display['cost'] = trades_display['cost'].round(2)
                    st.dataframe(trades_display, use_container_width=True)
                else:
                    st.info("No trades executed during this period")
        
        # Download Results
        st.markdown("### 💾 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare results CSV
            results_csv = results.to_csv()
            st.download_button(
                label="📥 Download Backtest Results (CSV)",
                data=results_csv,
                file_name=f"{coin}_{strategy_type.replace(' ', '_')}_backtest_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Prepare trades CSV
            if not trades.empty:
                trades_csv = trades.to_csv(index=False)
                st.download_button(
                    label="📥 Download Trade Log (CSV)",
                    data=trades_csv,
                    file_name=f"{coin}_{strategy_type.replace(' ', '_')}_trades.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Error running backtest: {str(e)}")
        st.info("Please check your settings and try again.")

else:
    # Welcome screen when no backtest is running
    st.markdown("## 👋 Welcome to the Crypto Backtester!")
    st.markdown("""
    This platform allows you to backtest two popular trading strategies on Bitcoin and Ethereum:
    
    ### 📈 Available Strategies:
    
    **1. Trend Following (SMA Crossover)**
    - Uses Simple Moving Average crossovers to identify trends
    - Enters when short-term SMA crosses above long-term SMA
    - Exits when short-term SMA crosses below long-term SMA
    
    **2. Mean Reversion (RSI-based)**
    - Uses Relative Strength Index to identify oversold conditions
    - Enters when RSI drops below the buy threshold
    - Exits when RSI rises above the exit threshold
    
    ### How to use:
    1. Select your initial capital
    2. Choose a cryptocurrency (BTC or ETH)
""")