"""
Trading strategy implementations
"""
import pandas as pd
import numpy as np

class TrendFollowingStrategy:
    """
    Trend Following Strategy using SMA crossover
    - Entry: SMA_50 crosses above SMA_200 (Golden Cross)
    - Exit: SMA_50 crosses below SMA_200 (Death Cross)
    """
    def __init__(self, sma_short=50, sma_long=200):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.name = f"Trend Following (SMA {sma_short}/{sma_long})"
    
    def generate_signals(self, data):
        """
        Generate trading signals based on SMA crossover
        
        Parameters:
        data: DataFrame with price data and SMA indicators
        
        Returns:
        DataFrame with signals column (1 = in position, 0 = out of position)
        """
        df = data.copy()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Calculate when SMA_short is above SMA_long
        df['sma_position'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
        
        # Generate signals on crossover points
        # Entry signal: when position changes from 0 to 1
        # Exit signal: when position changes from 1 to 0
        df['signal'] = df['sma_position']
        
        # Add entry/exit points for visualization
        df['entry'] = ((df['signal'] == 1) & (df['signal'].shift(1) == 0)).astype(int)
        df['exit'] = ((df['signal'] == 0) & (df['signal'].shift(1) == 1)).astype(int)
        
        print(f"📊 Generated {df['entry'].sum()} entry signals and {df['exit'].sum()} exit signals")
        
        return df

class MeanReversionStrategy:
    """
    Mean Reversion Strategy using RSI
    - Entry: RSI falls below buy threshold (oversold)
    - Exit: RSI rises above exit threshold
    """
    def __init__(self, rsi_period=14, buy_threshold=30, exit_threshold=50):
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.exit_threshold = exit_threshold
        self.name = f"Mean Reversion (RSI {rsi_period}, Buy<{buy_threshold}, Exit>{exit_threshold})"
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI levels
        
        Parameters:
        data: DataFrame with price data and RSI indicator
        
        Returns:
        DataFrame with signals column (1 = in position, 0 = out of position)
        """
        df = data.copy()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Track position state
        in_position = False
        
        for i in range(len(df)):
            if i == 0:
                continue
                
            current_rsi = df['RSI'].iloc[i]
            
            # Entry logic: RSI below buy threshold and not in position
            if current_rsi < self.buy_threshold and not in_position:
                df.loc[df.index[i], 'signal'] = 1
                in_position = True
            
            # Exit logic: RSI above exit threshold and in position
            elif current_rsi > self.exit_threshold and in_position:
                df.loc[df.index[i], 'signal'] = 0
                in_position = False
            
            # Stay in current position
            else:
                df.loc[df.index[i], 'signal'] = 1 if in_position else 0
        
        # Add entry/exit points for visualization
        df['entry'] = ((df['signal'] == 1) & (df['signal'].shift(1) == 0)).astype(int)
        df['exit'] = ((df['signal'] == 0) & (df['signal'].shift(1) == 1)).astype(int)
        
        print(f"📊 Generated {df['entry'].sum()} entry signals and {df['exit'].sum()} exit signals")
        
        return df

# Test function
def test_strategies():
    """Test both strategies with sample data"""
    from data_handler import DataHandler
    
    handler = DataHandler()
    
    # Test with BTC data
    print("="*60)
    print("TESTING STRATEGIES WITH BTC DATA")
    print("="*60)
    
    # Get data
    btc_data = handler.get_latest_data('BTC', days_back=730)
    
    # Test Trend Following
    print("\n1. Testing Trend Following Strategy")
    print("-"*40)
    btc_trend_data = handler.add_indicators(btc_data, 'trend')
    trend_strategy = TrendFollowingStrategy()
    btc_trend_signals = trend_strategy.generate_signals(btc_trend_data)
    
    # Show last few signals
    print("\nLast 5 days of trend signals:")
    print(btc_trend_signals[['Close', 'SMA_50', 'SMA_200', 'signal', 'entry', 'exit']].tail())
    
    # Test Mean Reversion
    print("\n2. Testing Mean Reversion Strategy")
    print("-"*40)
    btc_rsi_data = handler.add_indicators(btc_data, 'mean_reversion')
    mean_rev_strategy = MeanReversionStrategy()
    btc_rsi_signals = mean_rev_strategy.generate_signals(btc_rsi_data)
    
    # Show last few signals
    print("\nLast 5 days of mean reversion signals:")
    print(btc_rsi_signals[['Close', 'RSI', 'signal', 'entry', 'exit']].tail())
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Trend following stats
    trend_days_in = btc_trend_signals['signal'].sum()
    trend_days_out = len(btc_trend_signals) - trend_days_in
    trend_pct_in = (trend_days_in / len(btc_trend_signals)) * 100
    
    print(f"\nTrend Following Strategy:")
    print(f"- Days in position: {trend_days_in} ({trend_pct_in:.1f}%)")
    print(f"- Days out of position: {trend_days_out} ({100-trend_pct_in:.1f}%)")
    print(f"- Total trades: {btc_trend_signals['entry'].sum()}")
    
    # Mean reversion stats
    mr_days_in = btc_rsi_signals['signal'].sum()
    mr_days_out = len(btc_rsi_signals) - mr_days_in
    mr_pct_in = (mr_days_in / len(btc_rsi_signals)) * 100
    
    print(f"\nMean Reversion Strategy:")
    print(f"- Days in position: {mr_days_in} ({mr_pct_in:.1f}%)")
    print(f"- Days out of position: {mr_days_out} ({100-mr_pct_in:.1f}%)")
    print(f"- Total trades: {btc_rsi_signals['entry'].sum()}")
    
    print("\n✅ Strategies tested successfully!")

if __name__ == "__main__":
    test_strategies()