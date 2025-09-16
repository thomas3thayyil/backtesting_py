"""
Backtesting engine
"""
import pandas as pd
import numpy as np
from datetime import datetime

class Backtester:
    def __init__(self, initial_capital=10000, trading_fee=0.001, slippage=0.0005):
        """
        Initialize backtester
        
        Parameters:
        initial_capital: Starting capital in USD
        trading_fee: Fee per trade (0.001 = 0.1%)
        slippage: Slippage per trade (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.total_cost = trading_fee + slippage  # Total cost per trade
        
    def run_backtest(self, data_with_signals):
        """
        Run the backtest with given signals
        
        Parameters:
        data_with_signals: DataFrame with price data and signal column
        
        Returns:
        DataFrame with backtest results including portfolio value
        """
        df = data_with_signals.copy()
        
        # Initialize tracking variables
        cash = self.initial_capital
        position = 0  # Number of coins held
        portfolio_value = self.initial_capital
        
        # Lists to track values
        cash_values = []
        position_values = []
        portfolio_values = []
        trades = []
        
        # Iterate through each day
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            current_signal = df['signal'].iloc[i]
            previous_signal = df['signal'].iloc[i-1] if i > 0 else 0
            
            # Check if we need to trade
            if current_signal != previous_signal:
                if current_signal == 1:  # Buy signal
                    # Calculate how many coins we can buy
                    trade_cost = cash * self.total_cost
                    available_cash = cash - trade_cost
                    position = available_cash / current_price
                    
                    # Record trade
                    trades.append({
                        'date': df.index[i],
                        'type': 'BUY',
                        'price': current_price,
                        'coins': position,
                        'value': available_cash,
                        'cost': trade_cost
                    })
                    
                    cash = 0  # All cash is now in position
                    print(f"🟢 BUY on {df.index[i].date()}: {position:.6f} coins at ${current_price:,.2f}")
                    
                elif current_signal == 0 and position > 0:  # Sell signal
                    # Sell all position
                    gross_value = position * current_price
                    trade_cost = gross_value * self.total_cost
                    cash = gross_value - trade_cost
                    
                    # Record trade
                    trades.append({
                        'date': df.index[i],
                        'type': 'SELL',
                        'price': current_price,
                        'coins': position,
                        'value': cash,
                        'cost': trade_cost
                    })
                    
                    print(f"🔴 SELL on {df.index[i].date()}: {position:.6f} coins at ${current_price:,.2f}")
                    position = 0
            
            # Calculate portfolio value
            position_value = position * current_price if position > 0 else 0
            portfolio_value = cash + position_value
            
            # Store values
            cash_values.append(cash)
            position_values.append(position_value)
            portfolio_values.append(portfolio_value)
        
        # Add results to dataframe
        df['cash'] = cash_values
        df['position_value'] = position_values
        df['portfolio_value'] = portfolio_values
        df['returns'] = df['portfolio_value'].pct_change().fillna(0)
        df['cumulative_returns'] = (df['portfolio_value'] / self.initial_capital) - 1
        
        # Create trades dataframe
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate final statistics
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        num_trades = len(trades)
        
        print(f"\n📊 Backtest Complete:")
        print(f"- Initial Capital: ${self.initial_capital:,.2f}")
        print(f"- Final Portfolio Value: ${portfolio_value:,.2f}")
        print(f"- Total Return: {total_return:.2f}%")
        print(f"- Number of Trades: {num_trades}")
        
        return df, trades_df
    
    def calculate_buy_and_hold(self, data):
        """
        Calculate buy and hold returns for comparison
        
        Parameters:
        data: DataFrame with price data
        
        Returns:
        Series with buy and hold portfolio values
        """
        initial_price = data['Close'].iloc[0]
        coins_bought = self.initial_capital * (1 - self.total_cost) / initial_price
        
        buy_hold_values = coins_bought * data['Close']
        
        return buy_hold_values

# Test function
def test_backtester():
    """Test the backtester with sample strategy"""
    from data_handler import DataHandler
    from strategies import TrendFollowingStrategy, MeanReversionStrategy
    
    # Initialize components
    handler = DataHandler()
    backtester = Backtester(initial_capital=10000)
    
    print("="*60)
    print("TESTING BACKTESTER WITH BTC TREND FOLLOWING")
    print("="*60)
    
    # Get BTC data for last 2 years
    btc_data = handler.get_latest_data('BTC', days_back=730)
    
    # Add indicators and generate signals
    btc_with_indicators = handler.add_indicators(btc_data, 'trend')
    strategy = TrendFollowingStrategy()
    btc_with_signals = strategy.generate_signals(btc_with_indicators)
    
    # Run backtest
    results, trades = backtester.run_backtest(btc_with_signals)
    
    # Show sample results
    print("\n📈 Last 5 days of portfolio values:")
    print(results[['Close', 'signal', 'portfolio_value', 'cumulative_returns']].tail())
    
    if not trades.empty:
        print("\n📋 Last 5 trades:")
        print(trades.tail())
    
    # Compare with buy and hold
    buy_hold = backtester.calculate_buy_and_hold(btc_with_signals)
    buy_hold_return = (buy_hold.iloc[-1] / backtester.initial_capital - 1) * 100
    
    print(f"\n🎯 Comparison:")
    print(f"- Strategy Return: {results['cumulative_returns'].iloc[-1]*100:.2f}%")
    print(f"- Buy & Hold Return: {buy_hold_return:.2f}%")
    
    print("\n" + "="*60)
    print("TESTING BACKTESTER WITH ETH MEAN REVERSION")
    print("="*60)
    
    # Test with ETH mean reversion
    eth_data = handler.get_latest_data('ETH', days_back=730)
    eth_with_indicators = handler.add_indicators(eth_data, 'mean_reversion')
    mr_strategy = MeanReversionStrategy()
    eth_with_signals = mr_strategy.generate_signals(eth_with_indicators)
    
    # Run backtest
    eth_results, eth_trades = backtester.run_backtest(eth_with_signals)
    
    print("\n✅ Backtester tested successfully!")

if __name__ == "__main__":
    test_backtester()