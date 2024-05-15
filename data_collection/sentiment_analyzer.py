from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()


def analyze_with_vader(text):
    sentiment_scores = vader_analyzer.polarity_scores(text)
    return sentiment_scores['compound']


# Fetch and analyze sentiment
def fetch_and_analyze_sentiment(query, api_key):
    news_api = NewsApiClient(api_key=api_key)
    all_articles = news_api.get_everything(q=query, language='en', sort_by='publishedAt', page_size=100)

    sentiments = []
    for article in all_articles['articles']:
        headline = article['title']
        vader_score = analyze_with_vader(headline)
        final_score = np.mean([vader_score])
        sentiments.append((headline, vader_score, final_score))

    return sentiments
