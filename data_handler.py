"""
Data fetching and preprocessing module
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ta  # Technical Analysis library

class DataHandler:
    def __init__(self):
        # Map our symbols to Yahoo Finance symbols
        self.symbol_map = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD'
        }
        # Create data directories
        self.data_dir = "data"
        self.price_data_dir = os.path.join(self.data_dir, "price_data")
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.price_data_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def fetch_data(self, coin, start_date, end_date):
        """
        Fetch historical data for given coin
        
        Parameters:
        coin: 'BTC' or 'ETH'
        start_date: datetime object
        end_date: datetime object
        
        Returns:
        DataFrame with OHLCV data
        """
        print(f"📊 Downloading {coin} data...")
        
        # Get Yahoo Finance symbol
        symbol = self.symbol_map[coin]
        
        # Download data
        data = yf.download(
            symbol, 
            start=start_date, 
            end=end_date,
            progress=False
        )
        
        # Make sure we have data
        if data.empty:
            raise ValueError(f"No data found for {coin}")
        
        print(f"✅ Downloaded {len(data)} days of {coin} data")
        return data
    
    def add_indicators(self, data, strategy_type):
        """
        Add technical indicators based on strategy type
        
        Parameters:
        data: DataFrame with OHLCV data
        strategy_type: 'trend' or 'mean_reversion'
        
        Returns:
        DataFrame with indicators added
        """
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Flatten column names if they're multi-level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if strategy_type == 'trend':
            # Add SMAs for trend following
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            print("✅ Added SMA indicators")
            
        elif strategy_type == 'mean_reversion':
            # Add RSI for mean reversion
            # Make sure we're passing a Series, not a DataFrame
            close_prices = df['Close'].squeeze()  # Convert to Series if needed
            df['RSI'] = ta.momentum.RSIIndicator(
                close=close_prices, 
                window=14
            ).rsi()
            print("✅ Added RSI indicator")
        
        # Drop any NaN values created by indicators
        df = df.dropna()
        
        return df
    
    def get_latest_data(self, coin, days_back=365):
        """
        Quick function to get recent data
        
        Parameters:
        coin: 'BTC' or 'ETH'
        days_back: number of days of historical data
        
        Returns:
        DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return self.fetch_data(coin, start_date, end_date)
    
    def get_data_from_2018(self, coin):
        """
        Get all data from January 1, 2018 to today
        
        Parameters:
        coin: 'BTC' or 'ETH'
        
        Returns:
        DataFrame with all historical data since 2018
        """
        start_date = datetime(2018, 1, 1)
        end_date = datetime.now()
        
        return self.fetch_data(coin, start_date, end_date)

# Test function to make sure everything works
def test_data_handler():
    """Test if data handler is working properly"""
    handler = DataHandler()
    
    # Test both coins
    for coin in ['BTC', 'ETH']:
        print(f"\n{'='*50}")
        print(f"Testing {coin}")
        print('='*50)
        
        try:
            # Get 2 years of data (enough for 200-day SMA)
            data = handler.get_latest_data(coin, days_back=730)
            print(f"\n📈 {coin} Data Sample (last 3 days):")
            print(data.tail(3))
            
            # Test trend following indicators
            data_with_trend = handler.add_indicators(data, 'trend')
            print(f"\n📊 {coin} with SMA indicators (last 3 days):")
            print(data_with_trend[['Close', 'SMA_50', 'SMA_200']].tail(3))
            
            # Test mean reversion indicators
            data_with_rsi = handler.add_indicators(data, 'mean_reversion')
            print(f"\n📊 {coin} with RSI indicator (last 3 days):")
            print(data_with_rsi[['Close', 'RSI']].tail(3))
            
        except Exception as e:
            print(f"\n❌ Error with {coin}: {str(e)}")
            return False
    
    # Check 2018 data availability
    print("\n" + "="*50)
    print("🔍 Checking 2018 data availability...")
    print("="*50)
    try:
        btc_2018 = handler.get_data_from_2018('BTC')
        eth_2018 = handler.get_data_from_2018('ETH')
        print(f"✅ BTC data from 2018: {len(btc_2018)} days")
        print(f"✅ ETH data from 2018: {len(eth_2018)} days")
        print(f"📅 BTC date range: {btc_2018.index[0].date()} to {btc_2018.index[-1].date()}")
        print(f"📅 ETH date range: {eth_2018.index[0].date()} to {eth_2018.index[-1].date()}")
    except Exception as e:
        print(f"❌ Error fetching 2018 data: {str(e)}")
        return False
    
    print("\n✅ Data handler is working correctly for both BTC and ETH!")
    return True

def save_price_data(self, data, coin, start_date, end_date):
        """Save price data to CSV"""
        filename = f"{coin}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.price_data_dir, filename)
        data.to_csv(filepath)
        print(f"💾 Saved price data to {filepath}")
        return filepath

def load_price_data(self, coin, start_date, end_date):
        """Try to load price data from CSV"""
        filename = f"{coin}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.price_data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"📂 Loading cached data from {filepath}")
            return pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return None

def fetch_data(self, coin, start_date, end_date):
        """Updated fetch_data with caching"""
        # Try to load from cache first
        cached_data = self.load_price_data(coin, start_date, end_date)
        if cached_data is not None:
            return cached_data
        
        # If not cached, download as before
        print(f"📊 Downloading {coin} data...")
        symbol = self.symbol_map[coin]
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for {coin}")
        
        # Save to cache
        self.save_price_data(data, coin, start_date, end_date)
        
        print(f"✅ Downloaded {len(data)} days of {coin} data")
        return data

# Run test when file is executed directly
if __name__ == "__main__":
    test_data_handler()