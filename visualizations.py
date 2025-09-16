"""
Visualization module for charts and plots
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class Visualizer:
    def __init__(self):
        # Define color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def plot_equity_curve(self, backtest_results, strategy_name, coin):
        """
        Create equity curve plot showing portfolio value over time
        
        Parameters:
        backtest_results: DataFrame with portfolio values
        strategy_name: Name of the strategy
        coin: Cryptocurrency name
        
        Returns:
        Plotly figure object
        """
        fig = go.Figure()
        
        # Add portfolio value line
        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color=self.colors['primary'], width=2),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add buy/sell markers
        buys = backtest_results[backtest_results['entry'] == 1]
        sells = backtest_results[backtest_results['exit'] == 1]
        
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index,
                y=buys['portfolio_value'],
                mode='markers',
                name='Buy',
                marker=dict(
                    color=self.colors['success'],
                    size=10,
                    symbol='triangle-up'
                ),
                hovertemplate='BUY<br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
        
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index,
                y=sells['portfolio_value'],
                mode='markers',
                name='Sell',
                marker=dict(
                    color=self.colors['danger'],
                    size=10,
                    symbol='triangle-down'
                ),
                hovertemplate='SELL<br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{coin} - {strategy_name} - Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_drawdown(self, drawdown_series, strategy_name, coin):
        """
        Create drawdown plot
        
        Parameters:
        drawdown_series: Series of drawdown percentages
        strategy_name: Name of the strategy
        coin: Cryptocurrency name
        
        Returns:
        Plotly figure object
        """
        fig = go.Figure()
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color=self.colors['danger'], width=1),
            name='Drawdown',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{coin} - {strategy_name} - Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def plot_returns_distribution(self, returns, strategy_name, coin):
        """
        Create returns distribution histogram
        
        Parameters:
        returns: Series of daily returns
        strategy_name: Name of the strategy
        coin: Cryptocurrency name
        
        Returns:
        Plotly figure object
        """
        # Convert returns to percentage
        returns_pct = returns * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name='Daily Returns',
            marker_color=self.colors['primary'],
            hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
        ))
        
        # Add vertical line at mean
        mean_return = returns_pct.mean()
        fig.add_vline(
            x=mean_return,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_return:.2f}%"
        )
        
        fig.update_layout(
            title=f'{coin} - {strategy_name} - Returns Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_price_and_signals(self, data_with_signals, strategy_name, coin):
        """
        Create price chart with strategy signals and indicators
        
        Parameters:
        data_with_signals: DataFrame with price and signal data
        strategy_name: Name of the strategy
        coin: Cryptocurrency name
        
        Returns:
        Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{coin} Price & Indicators', 'Position Status'),
            row_heights=[0.7, 0.3]
        )
        
        # Plot 1: Price and indicators
        fig.add_trace(
            go.Scatter(
                x=data_with_signals.index,
                y=data_with_signals['Close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=1)
            ),
            row=1, col=1
        )
        
        # Add strategy-specific indicators
        if 'SMA_50' in data_with_signals.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_with_signals.index,
                    y=data_with_signals['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color=self.colors['primary'], width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data_with_signals.index,
                    y=data_with_signals['SMA_200'],
                    mode='lines',
                    name='SMA 200',
                    line=dict(color=self.colors['secondary'], width=1)
                ),
                row=1, col=1
            )
        
        # Add buy/sell markers
        buys = data_with_signals[data_with_signals['entry'] == 1]
        sells = data_with_signals[data_with_signals['exit'] == 1]
        
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        color=self.colors['success'],
                        size=12,
                        symbol='triangle-up'
                    )
                ),
                row=1, col=1
            )
        
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        color=self.colors['danger'],
                        size=12,
                        symbol='triangle-down'
                    )
                ),
                row=1, col=1
            )
        
        # Plot 2: Position status
        fig.add_trace(
            go.Scatter(
                x=data_with_signals.index,
                y=data_with_signals['signal'],
                mode='lines',
                name='Position',
                line=dict(color=self.colors['info'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 255, 0.2)'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{coin} - {strategy_name} - Signals Overview',
            template='plotly_white',
            height=800,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='Position', row=2, col=1)
        
        return fig
    
    def create_metrics_table(self, metrics_dict):
        """
        Create a formatted metrics table
        
        Parameters:
        metrics_dict: Dictionary of performance metrics
        
        Returns:
        Plotly figure object
        """
        # Prepare data for table
        metrics_list = []
        values_list = []
        
        for metric, value in metrics_dict.items():
            metrics_list.append(metric)
            if isinstance(value, float):
                values_list.append(f"{value:.2f}")
            else:
                values_list.append(str(value))
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color=self.colors['primary'],
                font=dict(color='white', size=14),
                align='left'
            ),
            cells=dict(
                values=[metrics_list, values_list],
                fill_color=[['lightgray', 'white'] * len(metrics_list)],
                align='left',
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title='Performance Metrics',
            height=400
        )
        
        return fig

# Test function
def test_visualizer():
    """Test the visualization module"""
    from data_handler import DataHandler
    from strategies import TrendFollowingStrategy
    from backtester import Backtester
    from performance import PerformanceAnalyzer
    
    print("Testing Visualizer...")
    
    # Initialize components
    handler = DataHandler()
    visualizer = Visualizer()
    
    # Get sample data and run backtest
    btc_data = handler.get_latest_data('BTC', days_back=365)
    btc_with_indicators = handler.add_indicators(btc_data, 'trend')
    strategy = TrendFollowingStrategy()
    btc_with_signals = strategy.generate_signals(btc_with_indicators)
    
    backtester = Backtester(initial_capital=10000)
    results, trades = backtester.run_backtest(btc_with_signals)
    
    # Calculate performance metrics
    analyzer = PerformanceAnalyzer()
    metrics, drawdown_series = analyzer.calculate_metrics(results, 10000)
    
    # Test each visualization
    print("✅ Creating equity curve...")
    equity_fig = visualizer.plot_equity_curve(results, "Trend Following", "BTC")
    
    print("✅ Creating drawdown chart...")
    drawdown_fig = visualizer.plot_drawdown(drawdown_series, "Trend Following", "BTC")
    
    print("✅ Creating returns distribution...")
    returns_fig = visualizer.plot_returns_distribution(results['returns'], "Trend Following", "BTC")
    
    print("✅ Creating price and signals chart...")
    signals_fig = visualizer.plot_price_and_signals(btc_with_signals, "Trend Following", "BTC")
    
    print("✅ Creating metrics table...")
    metrics_fig = visualizer.create_metrics_table(metrics)
    
    print("\n✅ All visualizations created successfully!")
    print("Note: Visualizations will be displayed in the Streamlit UI")

if __name__ == "__main__":
    test_visualizer()