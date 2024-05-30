from data_collection.collect_num_data import get_economic_indicators, get_historical_prices, get_dividends_and_earnings
from data_collection.sentiment_analyzer import fetch_and_analyze_sentiment
from model import train_and_evaluate_model
import pandas as pd

api_key = 'fd756b8b451b492aa863abe017d5192c'


def get_portfolio_data(path):
    return pd.read_csv(path)

def make_tz_naive(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


if __name__ == "__main__":
    indicators = [
            "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "GS10",
        "INDPRO", "PPIACO", "RSXFS", "HOUST", "PSAVERT"
    ]

    portfolio_csv = "portfolio_files/Apr-2025 portfolio.csv"
    portfolio_symbols = get_portfolio_data(portfolio_csv)["Symbol"]

    historical_data = {}
    dividends_data = {}
    earnings_data = {}
    sentiment_scores = {}
    economic_indicators = {}

    for symbol in portfolio_symbols:
        hist_data = get_historical_prices(symbol)
        historical_data[symbol] = make_tz_naive(hist_data)

        sentiment_scores[symbol] = fetch_and_analyze_sentiment(symbol, api_key)
    # Collecting economic indicators
    for indicator in indicators:
        econ_data = get_economic_indicators(indicator)
        econ_data.to_csv(f'./data/economic_indicators/{indicator}')

    merged_data = pd.DataFrame()
    for symbol in portfolio_symbols:
        data = historical_data[symbol]
        data['Sentiment'] = sentiment_scores[symbol]
        merged_data = pd.concat([merged_data, data], axis=0)

    econ_indic_df = pd.concat(economic_indicators.values(), axis=1)
    merged_data = pd.concat([merged_data, econ_indic_df], axis=1)

    train_and_evaluate_model(merged_data)