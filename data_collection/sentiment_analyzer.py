from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import numpy as np

nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()


def analyze_with_vader(text):
    sentiment_scores = vader_analyzer.polarity_scores(text)
    return sentiment_scores['compound']


def fetch_and_analyze_sentiment(query, api_key):
    news_api = NewsApiClient(api_key=api_key)
    all_articles = news_api.get_everything(q=query, language='en', sort_by='publishedAt', page_size=100)

    sentiments = []
    sentiment_scores = []
    for article in all_articles['articles']:
        headline = article['title']
        vader_score = analyze_with_vader(headline)
        final_score = np.mean([vader_score])
        sentiment_scores.append(np.mean([vader_score]))
        sentiments.append((headline, vader_score, final_score))
    return np.mean(sentiment_scores)
