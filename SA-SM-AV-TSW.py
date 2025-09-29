import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ====== API Keys ======

api_key = '08cbfd9ce468468688c55507dfc9b27b' 
ALPHA_VANTAGE_API_KEY = 'HY8C9U9LDF79OPG9'

# ====== Data Collection ======

def collect_news(api_key):
    url = f'https://newsapi.org/v2/everything?q=stock market&from=2024-11-01&sortBy=popularity&pageSize=100&apiKey={api_key}'
    response = requests.get(url)
    return response.json().get('articles', [])

def collect_stock_prices(symbol, start_date='2024-11-01'):
    """
    Fetches weekly stock prices for a given symbol from Alpha Vantage using the TIME_SERIES_WEEKLY function.
    :param symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOG')
    :param start_date: The start date for the data collection in 'YYYY-MM-DD' format
    :return: DataFrame containing stock prices (weekly)
    """
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_WEEKLY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'datatype': 'json'  # You can change to 'csv' if you prefer to download CSV
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Check if the response contains the time series data
    if 'Weekly Time Series' in data:
        time_series = data['Weekly Time Series']
        
        stock_data = []
        for date, values in time_series.items():
            if date >= start_date:  # Filter by the start date
                stock_data.append({
                    'date': date,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume'])
                })
        
        # Convert to DataFrame
        stock_df = pd.DataFrame(stock_data)
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        stock_df = stock_df.sort_values(by='date')
        return stock_df
    else:
        raise ValueError("Error fetching data from Alpha Vantage.")

# ====== Data Preprocessing ======

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    text = text.lower()                   # Convert to lowercase
    tokens = word_tokenize(text)          # Tokenize
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
    return ' '.join(tokens)  # Join tokens back to a string

# ====== Sentiment Analysis ======

def analyze_sentiment(articles, stock_prices):
    """
    Analyzes sentiment of news articles and matches them with stock prices (weekly).
    :param articles: List of articles with their sentiment scores
    :param stock_prices: DataFrame containing weekly stock price data
    :return: DataFrame with sentiment data and associated stock prices
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    
    for article in articles:
        cleaned_title = preprocess_text(article['title'])
        sentiment_score = analyzer.polarity_scores(cleaned_title)['compound']
        
        # Convert article date to datetime
        article_date = pd.to_datetime(article['publishedAt'])
        
        # Make both stock_prices['date'] and article_date timezone-naive (if they are timezone-aware)
        if article_date.tzinfo is not None:  # Check if article_date has timezone info
            article_date = article_date.tz_localize(None)  # Remove timezone if present
        
        # Convert stock_prices['date'] to timezone-naive if necessary
        stock_prices['date'] = stock_prices['date'].dt.tz_localize(None)  # Remove timezone if present
        
        # Find the closest stock price date
        closest_price = stock_prices.loc[stock_prices['date'] <= article_date].iloc[-1]['close'] if not stock_prices.loc[stock_prices['date'] <= article_date].empty else np.nan
        
        sentiment_data.append({
            'title': article['title'],
            'sentiment': sentiment_score,
            'date': article['publishedAt'],
            'source': article['source']['name'],
            'closing_price': closest_price  # Using the closest weekly stock price
        })
    
    return pd.DataFrame(sentiment_data)
# ====== Feature Engineering ======

def extract_features(df):
    df['keyword_buy'] = df['title'].str.contains('buy', case=False).astype(int)
    df['keyword_sell'] = df['title'].str.contains('sell', case=False).astype(int)
    df['keyword_crash'] = df['title'].str.contains('crash', case=False).astype(int)
    df['lagged_sentiment'] = df['sentiment'].shift(1)
    
    # Define stock movement: 1 if the next closing price is greater, else 0
    df['stock_movement'] = (df['closing_price'].shift(-1) > df['closing_price']).astype(int)
    
    df.dropna(inplace=True)  # Drop rows with NaN values after shifting
    return df


# ====== Visualization ======
def plot_stock_and_sentiment(df):
    # Ensure 'date' is in datetime format and sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Apply a 3-week rolling mean to smooth data
    df['rolling_sentiment'] = df['sentiment'].rolling(window=3).mean()
    df['rolling_closing_price'] = df['closing_price'].rolling(window=3).mean()

    # Create a Plotly figure
    fig = go.Figure()

    # Add smoothed stock price trace
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_closing_price'], mode='lines', 
        name='Stock Price', line=dict(color='blue', width=2)
    ))

    # Add smoothed sentiment score trace on a secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_sentiment'], mode='lines',
        name='Sentiment Score', line=dict(color='red', width=2), yaxis='y2'
    ))

    # Update layout for better readability
    fig.update_layout(
        title='Stock Price and Sentiment Over Time (Smoothed)',
        xaxis_title='Date',
        yaxis=dict(title='Stock Price', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Sentiment Score', titlefont=dict(color='red'), tickfont=dict(color='red'),
                    overlaying='y', side='right'),
        template="plotly_white"
    )

    # Show plot
    fig.show()


def plot_stock_price(df):
    # Ensure 'date' is in datetime format and sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Apply 3-period rolling mean to closing prices
    df['rolling_closing_price'] = df['closing_price'].rolling(window=7).mean()  

    # Create a Plotly figure for stock price
    fig = go.Figure()

    # Add smoothed stock price trace (use 'rolling_closing_price' here)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_closing_price'], mode='lines',
        name='Smoothed Stock Price (3-day Rolling Avg)', line=dict(color='blue', width=2)
    ))

    # Update layout
    fig.update_layout(
        title='Smoothed Stock Price Over Time (3-day Rolling Avg)',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        template="plotly_white"
    )

    # Show stock price plot
    fig.show()

# Function to plot only the smoothed sentiment score (using EMA)
def plot_sentiment_score(df):
    # Ensure 'date' is in datetime format and sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Apply 3-period rolling mean to sentiment scores
    df['rolling_sentiment'] = df['sentiment'].rolling(window=7).mean()

    # Create a Plotly figure for sentiment score
    fig = go.Figure()

    # Add smoothed sentiment score trace (use 'rolling_sentiment' here)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_sentiment'], mode='lines',
        name='Smoothed Sentiment Score (3-day Rolling Avg)', line=dict(color='red', width=2)
    ))

    # Update layout
    fig.update_layout(
        title='Smoothed Sentiment Score Over Time (3-day Rolling Avg)',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        template="plotly_white"
    )

    # Show sentiment score plot
    fig.show()


# ====== Main Execution Flow ======

# Replace with your own API key
# api_key = 'd4a40858d71f4397a7de8ec8fa267041'

articles = collect_news(api_key)
symbol = 'TCS'  # Replace with the desired stock symbol
stock_prices = collect_stock_prices(symbol)
df = analyze_sentiment(articles, stock_prices)

# Save sentiment data to SQLite
conn = sqlite3.connect('sentiment_analysis.db')
df.to_sql('articles', conn, if_exists='replace', index=False)
conn.close()

# Feature engineering
df = extract_features(df)

plot_stock_and_sentiment(df)
plot_stock_price(df)
plot_sentiment_score(df)


