import yfinance as yf
import pandas as pd

portfolio_tickers = ["NVDA", "MSFT", "VOO", "GOOG", "JPM"]


def fetch_portfolio_data(tickers):
    all_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max")
        all_data[ticker] = hist
    return all_data


portfolio_data = fetch_portfolio_data(portfolio_tickers)

for ticker, data in portfolio_data.items():
    print(f"\n{ticker}:\n", data.tail())


def prepare_data(data):
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data


prepared_data = {ticker: prepare_data(data) for ticker, data in portfolio_data.items()}


def create_features(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Close'].rolling(window=50).std()
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    data['Momentum_30'] = data['Close'] / data['Close'].shift(30) - 1
    data['Volume_Change'] = data['Volume'].pct_change()
    return data


feature_data = {ticker: create_features(data) for ticker, data in prepared_data.items()}
