import yfinance as yf
import pandas_datareader.data as web
import datetime


def get_historical_prices(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]


def get_dividends_and_earnings(ticker):
    stock = yf.Ticker(ticker)
    try:
        dividends = stock.dividends
        earnings = stock.earnings
        return dividends
    except Exception as e:
        print(e)


def get_economic_indicators(indicator_id):
    start = datetime.datetime(1992, 1, 1)
    end = datetime.datetime.now()
    data = web.DataReader(indicator_id, 'fred', start, end)
    return data
