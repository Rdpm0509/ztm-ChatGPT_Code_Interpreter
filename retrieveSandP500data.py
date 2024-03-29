import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the ticker symbol for the S&P 500
ticker_symbol = "^GSPC"

# Define the start and end dates
start_date = "2000-01-01"
end_date = "2024-03-24"  # Adjust this date to your current date or the end date you're interested in

downloadData=False
if downloadData:
    # Download the historical data
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Display the first few rows of the dataframe
    print(sp500_data.head())

    sp500_data.to_csv("S&P500.csv")

# Trend Analysis Output 
sp500_df = pd.read_csv("S&P500.csv")

# Convert Date column to datetime format
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])

# Set Date as the index
sp500_df.set_index('Date', inplace=True)

# Calculate the rolling mean (or moving average) for 200 days to smooth out short-term fluctuations and highlight longer-term trends
sp500_df['200d'] = sp500_df['Adj Close'].rolling(window=200).mean()

# Plotting the Adjusted Close price and the 200-day moving average
plt.figure(figsize=(14, 7))
plt.plot(sp500_df.index, sp500_df['Adj Close'], label='Adjusted Close')
plt.plot(sp500_df.index, sp500_df['200d'], label='200-Day Moving Average', color='orange')

plt.title('Trend Analysis of S&P 500 (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()


# Seasonal and Cyclical Analysis 

# Correcting the approach for calculating monthly returns
# First, calculate daily returns
sp500_df['Daily Return'] = sp500_df['Adj Close'].pct_change()

# Then, calculate average monthly returns by resampling
monthly_returns = sp500_df['Daily Return'].resample('M').agg(lambda x: (1 + x).prod() - 1) * 100

# Now, calculate the average returns for each month across all years
average_monthly_returns_corrected = monthly_returns.groupby(monthly_returns.index.month).mean()

# Plotting corrected average monthly returns
plt.figure(figsize=(10, 6))
average_monthly_returns_corrected.plot(kind='bar', color='skyblue')
plt.title('Corrected Average Monthly Returns of S&P 500 (2000-2023)')
plt.xlabel('Month')
plt.ylabel('Average Return (%)')
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.axhline(0, color='grey', linestyle='--')
plt.show()

average_monthly_returns_corrected

# Technical Analysis Output
# Calculating the MACD and Signal Line
sp500_df['12d_EMA'] = sp500_df['Adj Close'].ewm(span=12, adjust=False).mean()
sp500_df['26d_EMA'] = sp500_df['Adj Close'].ewm(span=26, adjust=False).mean()
sp500_df['MACD'] = sp500_df['12d_EMA'] - sp500_df['26d_EMA']
sp500_df['Signal_Line'] = sp500_df['MACD'].ewm(span=9, adjust=False).mean()

# Plotting MACD and Signal Line
plt.figure(figsize=(14, 7))
plt.plot(sp500_df.index, sp500_df['MACD'], label='MACD', color='blue')
plt.plot(sp500_df.index, sp500_df['Signal_Line'], label='Signal Line', color='red')
plt.title('MACD for S&P 500 (2000-2023)')
plt.xlabel('Year')
plt.ylabel('MACD Value')
plt.legend()
plt.show()



# ----------
# Manual calculation of RSI (as talib is not available)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculating Bollinger Bands without talib
def calculate_bollinger_bands(data, window=20, num_of_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    
    return upper_band, rolling_mean, lower_band

# Calculating Bollinger Bands
sp500_df['20d_SMA'] = sp500_df['Adj Close'].rolling(window=20).mean()
sp500_df['SD20'] = sp500_df['Adj Close'].rolling(window=20).std()
sp500_df['Upper_Band'] = sp500_df['20d_SMA'] + (sp500_df['SD20'] * 2)
sp500_df['Lower_Band'] = sp500_df['20d_SMA'] - (sp500_df['SD20'] * 2)

# Calculating RSI
delta = sp500_df['Adj Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
sp500_df['RSI'] = 100 - (100 / (1 + rs))

# Extracting the last row for current (most recent) values of the indicators
current_values = sp500_df.iloc[-1]

# Plotting Bollinger Bands
plt.figure(figsize=(14, 7))
plt.plot(sp500_df.index, sp500_df['Adj Close'], label='Adjusted Close', color='blue')
plt.plot(sp500_df.index, sp500_df['Upper_Band'], label='Upper Bollinger Band', color='red')
plt.plot(sp500_df.index, sp500_df['20d_SMA'], label='20-Day SMA', color='green')
plt.plot(sp500_df.index, sp500_df['Lower_Band'], label='Lower Bollinger Band', color='cyan')
plt.title('Bollinger Bands for S&P 500 (2000-2023)')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.show()

current_values[['Adj Close', 'MACD', 'Signal_Line', 'Upper_Band', '20d_SMA', 'Lower_Band', 'RSI']]

# Ensuring the dataframe is focused on the last 12 months
last_12_months_df = sp500_df.last('12M')

# Recalculating the 50d and 200d moving averages, RSI, and Bollinger Bands for the last 12 months
last_12_months_df['50d_MA'] = last_12_months_df['Adj Close'].rolling(window=50).mean()
last_12_months_df['200d_MA'] = last_12_months_df['Adj Close'].rolling(window=200).mean()
last_12_months_df['RSI'] = calculate_rsi(last_12_months_df['Adj Close'])
last_12_months_df['upper_band'], last_12_months_df['middle_band'], last_12_months_df['lower_band'] = calculate_bollinger_bands(last_12_months_df['Adj Close'])

# Plotting again with corrected data
fig, axs = plt.subplots(3, 1, figsize=(14, 15))

# Adjusted Close Price with Moving Averages
axs[0].plot(last_12_months_df.index, last_12_months_df['Adj Close'], label='Adjusted Close', color='blue')
axs[0].plot(last_12_months_df.index, last_12_months_df['50d_MA'], label='50-Day MA', color='green')
axs[0].plot(last_12_months_df.index, last_12_months_df['200d_MA'], label='200-Day MA', color='red')
axs[0].set_title('Adjusted Close and Moving Averages')
axs[0].legend()

# RSI Plot
axs[1].plot(last_12_months_df.index, last_12_months_df['RSI'], color='purple')
axs[1].axhline(70, linestyle='--', color='red', label='Overbought')
axs[1].axhline(30, linestyle='--', color='green', label='Oversold')
axs[1].set_title('Relative Strength Index (RSI)')
axs[1].legend()

# Bollinger Bands Plot
axs[2].plot(last_12_months_df.index, last_12_months_df['Adj Close'], label='Adjusted Close', color='blue')
axs[2].plot(last_12_months_df.index, last_12_months_df['upper_band'], label='Upper Band', linestyle='--', color='grey')
axs[2].plot(last_12_months_df.index, last_12_months_df['middle_band'], label='Middle Band', color='red')
axs[2].plot(last_12_months_df.index, last_12_months_df['lower_band'], label='Lower Band', linestyle='--', color='grey')
axs[2].set_title('Bollinger Bands')
axs[2].legend()

plt.tight_layout()
plt.show()
