from data_collection.collect_num_data import get_economic_indicators
from data_collection.sentiment_analyzer import fetch_and_analyze_sentiment

if __name__ == "__main__":
    indicators = [
        "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS", "GS10",
        "INDPRO", "PPIACO", "RSXFS", "HOUST", "PSAVERT"
    ]

    for indicator in indicators:
        data = get_economic_indicators(indicator)
        print(f"{indicator}:")
        print(data.tail())

    api_key = '<api_key>'
    sentiments = fetch_and_analyze_sentiment('nvda', api_key)
    for headline, vader_score, final_score in sentiments:
        print(
            f"Headline: {headline} | VADER Score: {vader_score}  | Final Score: {final_score}")
