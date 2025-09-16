"""
Performance metrics calculation
"""
import pandas as pd
import numpy as np
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, risk_free_rate=0.02):
        """
        Initialize performance analyzer
        
        Parameters:
        risk_free_rate: Annual risk-free rate (default 2% = 0.02)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 365  # Crypto trades 365 days
    
    def calculate_metrics(self, backtest_results, initial_capital):
        """
        Calculate all performance metrics
        
        Parameters:
        backtest_results: DataFrame from backtester with portfolio values
        initial_capital: Starting capital
        
        Returns:
        Dictionary with all metrics
        """
        # Basic returns
        total_return = (backtest_results['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        
        # Calculate daily returns
        daily_returns = backtest_results['returns']
        
        # Calculate annualized metrics
        days_traded = len(backtest_results)
        years_traded = days_traded / self.trading_days_per_year
        
        # Annualized return
        annualized_return = ((backtest_results['portfolio_value'].iloc[-1] / initial_capital) ** (1/years_traded) - 1) * 100
        
        # Sharpe Ratio
        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns)
        
        # Drawdown analysis
        drawdown_series, max_drawdown, max_drawdown_duration = self.calculate_drawdown(
            backtest_results['portfolio_value']
        )
        
        # Win rate (for days, not trades)
        positive_days = (daily_returns > 0).sum()
        negative_days = (daily_returns < 0).sum()
        total_days = positive_days + negative_days
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        
        # Volatility
        daily_volatility = daily_returns.std()
        annual_volatility = daily_volatility * np.sqrt(self.trading_days_per_year) * 100
        
        # Best and worst days
        best_day = daily_returns.max() * 100
        worst_day = daily_returns.min() * 100
        
        # Create metrics dictionary
        metrics = {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Max Drawdown Duration (days)': max_drawdown_duration,
            'Win Rate (%)': round(win_rate, 2),
            'Annual Volatility (%)': round(annual_volatility, 2),
            'Best Day (%)': round(best_day, 2),
            'Worst Day (%)': round(worst_day, 2),
            'Days Traded': days_traded,
            'Final Portfolio Value': round(backtest_results['portfolio_value'].iloc[-1], 2)
        }
        
        return metrics, drawdown_series
    
    def calculate_sharpe_ratio(self, returns):
        """
        Calculate Sharpe Ratio
        
        Parameters:
        returns: Series of daily returns
        
        Returns:
        Sharpe Ratio (annualized)
        """
        # Calculate excess returns
        daily_risk_free = self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - daily_risk_free
        
        # Calculate Sharpe Ratio
        if returns.std() == 0:
            return 0
        
        sharpe = np.sqrt(self.trading_days_per_year) * excess_returns.mean() / returns.std()
        
        return sharpe
    
    def calculate_drawdown(self, portfolio_values):
        """
        Calculate drawdown series and maximum drawdown
        
        Parameters:
        portfolio_values: Series of portfolio values
        
        Returns:
        Tuple of (drawdown_series, max_drawdown_percentage, max_duration_days)
        """
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown series
        drawdown_series = (portfolio_values - running_max) / running_max * 100
        
        # Maximum drawdown
        max_drawdown = drawdown_series.min()
        
        # Calculate drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i in range(len(portfolio_values)):
            if portfolio_values.iloc[i] < running_max.iloc[i]:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        # Check last drawdown
        if current_duration > max_duration:
            max_duration = current_duration
        
        return drawdown_series, max_drawdown, max_duration
    
    def compare_strategies(self, results_dict):
        """
        Compare multiple strategy results
        
        Parameters:
        results_dict: Dictionary with strategy names as keys and results as values
        
        Returns:
        DataFrame with comparison
        """
        comparison_data = {}
        
        for strategy_name, (results, initial_capital) in results_dict.items():
            metrics, _ = self.calculate_metrics(results, initial_capital)
            comparison_data[strategy_name] = metrics
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        return comparison_df

# Test function
def test_performance_analyzer():
    """Test the performance analyzer"""
    from data_handler import DataHandler
    from strategies import TrendFollowingStrategy, MeanReversionStrategy
    from backtester import Backtester
    
    # Initialize components
    handler = DataHandler()
    analyzer = PerformanceAnalyzer()
    initial_capital = 10000
    
    print("="*60)
    print("PERFORMANCE ANALYSIS - BTC TREND FOLLOWING")
    print("="*60)
    
    # Run backtest for BTC Trend Following
    btc_data = handler.get_latest_data('BTC', days_back=730)
    btc_with_indicators = handler.add_indicators(btc_data, 'trend')
    strategy = TrendFollowingStrategy()
    btc_with_signals = strategy.generate_signals(btc_with_indicators)
    
    backtester = Backtester(initial_capital=initial_capital)
    btc_results, _ = backtester.run_backtest(btc_with_signals)
    
    # Calculate metrics
    metrics, drawdown_series = analyzer.calculate_metrics(btc_results, initial_capital)
    
    print("\n📊 Performance Metrics:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value}")
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS - ETH MEAN REVERSION")
    print("="*60)
    
    # Run backtest for ETH Mean Reversion
    eth_data = handler.get_latest_data('ETH', days_back=730)
    eth_with_indicators = handler.add_indicators(eth_data, 'mean_reversion')
    mr_strategy = MeanReversionStrategy()
    eth_with_signals = mr_strategy.generate_signals(eth_with_indicators)
    
    eth_results, _ = backtester.run_backtest(eth_with_signals)
    
    # Calculate metrics
    eth_metrics, eth_drawdown = analyzer.calculate_metrics(eth_results, initial_capital)
    
    print("\n📊 Performance Metrics:")
    for metric, value in eth_metrics.items():
        print(f"- {metric}: {value}")
    
    # Compare strategies
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    comparison_data = {
        'BTC Trend Following': (btc_results, initial_capital),
        'ETH Mean Reversion': (eth_results, initial_capital)
    }
    
    comparison_df = analyzer.compare_strategies(comparison_data)
    print("\n", comparison_df)
    
    print("\n✅ Performance analyzer tested successfully!")

if __name__ == "__main__":
    test_performance_analyzer()