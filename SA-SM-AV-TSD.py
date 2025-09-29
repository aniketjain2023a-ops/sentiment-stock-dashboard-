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
# api_key = 'd4a40858d71f4397a7de8ec8fa267041'
ALPHA_VANTAGE_API_KEY = 'YMRLDD6F5U18OD4M'
company = 'apple'

# ====== Data Collection ======

def collect_news(api_key):
    url = f'https://newsapi.org/v2/everything?q={company}&from=2024-11-01&sortBy=popularity&pageSize=100&apiKey={api_key}'
    response = requests.get(url)
    return response.json().get('articles', [])

def collect_stock_prices(symbol, start_date='2024-11-01'):
    """
    Fetches daily stock prices for a given symbol from Alpha Vantage.
    :param symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOG')
    :param start_date: The start date for the data collection in 'YYYY-MM-DD' format
    :return: DataFrame containing stock prices
    """
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY,
        'outputsize': 'full'
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'Time Series (Daily)' in data:
        time_series = data['Time Series (Daily)']
        stock_data = []
        for date, values in time_series.items():
            if date >= start_date: 
                stock_data.append({
                    'date': date,
                    'closing_price': float(values['4. close'])
                })

        stock_df = pd.DataFrame(stock_data)
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        stock_df = stock_df.sort_values(by='date')
        return stock_df
    else:
        raise ValueError("Error fetching data from Alpha Vantage.")

# ====== Data Preprocessing ======

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)      
    text = text.lower()                 
    tokens = word_tokenize(text)          
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words] 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  
    return ' '.join(tokens) 

# ====== Sentiment Analysis ======

def analyze_sentiment(articles, stock_prices):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    
    for article in articles:
        cleaned_title = preprocess_text(article['title'])
        sentiment_score = analyzer.polarity_scores(cleaned_title)['compound']
        
        article_date = pd.to_datetime(article['publishedAt'])
        
        if article_date.tzinfo is not None: 
            article_date = article_date.tz_localize(None) 
        stock_prices['date'] = stock_prices['date'].dt.tz_localize(None)  

        closest_price = stock_prices.loc[stock_prices['date'] <= article_date].iloc[-1]['closing_price'] if not stock_prices.loc[stock_prices['date'] <= article_date].empty else np.nan
        
        sentiment_data.append({
            'title': article['title'],
            'sentiment': sentiment_score,
            'date': article['publishedAt'],
            'source': article['source']['name'],
            'closing_price': closest_price 
        })
    
    df = pd.DataFrame(sentiment_data)
    
    # Apply rolling averages
    df['rolling_sentiment'] = df['sentiment'].rolling(window=7).mean()
    df['rolling_closing_price'] = df['closing_price'].rolling(window=7).mean()
    
    return df

# ====== Feature Engineering ======

def extract_features(df):
    df['keyword_buy'] = df['title'].str.contains('buy', case=False).astype(int)
    df['keyword_sell'] = df['title'].str.contains('sell', case=False).astype(int)
    df['keyword_crash'] = df['title'].str.contains('crash', case=False).astype(int)
    df['lagged_sentiment'] = df['sentiment'].shift(1)

    df['stock_movement'] = (df['closing_price'].shift(-1) > df['closing_price']).astype(int)
    
    df.dropna(inplace=True)
    return df


# ====== Visualization ======


# def plot_stock_and_sentiment(df, rolling_window=3):
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.sort_values(by='date')

#     if 'sentiment' not in df.columns or 'closing_price' not in df.columns:
#         raise ValueError("DataFrame must contain 'sentiment' and 'closing_price' columns.")


#     df['rolling_sentiment'] = df['sentiment'].rolling(window=rolling_window).mean()
#     df['rolling_closing_price'] = df['closing_price'].rolling(window=rolling_window).mean()
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=df['date'], y=df['rolling_closing_price'], mode='lines', 
#         name='Stock Price', line=dict(color='blue', width=2)
#     ))

#     fig.add_trace(go.Scatter(
#         x=df['date'], y=df['rolling_sentiment'], mode='lines',
#         name='Sentiment Score', line=dict(color='red', width=2), yaxis='y2'
#     ))

#     fig.update_layout(
#         title='Stock Price and Sentiment Over Time (Smoothed)',
#         xaxis_title='Date',
#         yaxis=dict(title='Stock Price', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
#         yaxis2=dict(title='Sentiment Score', titlefont=dict(color='red'), tickfont=dict(color='red'),
#                     overlaying='y', side='right'),
#         template="plotly_white"
#     )

#     fig.show()

def plot_stock_and_sentiment(df, rolling_window=3):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    if 'sentiment' not in df.columns or 'closing_price' not in df.columns:
        raise ValueError("DataFrame must contain 'sentiment' and 'closing_price' columns.")

    # Calculate rolling sentiment and EMA smoothed stock price
    df['rolling_sentiment'] = df['sentiment'].rolling(window=rolling_window).mean()
    df['ema_closing_price'] = df['closing_price'].ewm(span=5, adjust=False).mean()
    
    fig = go.Figure()

    # Plot EMA-smoothed stock price instead of rolling closing price
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ema_closing_price'], mode='lines', 
        name='EMA Smoothed Stock Price', line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_sentiment'], mode='lines',
        name='Sentiment Score', line=dict(color='red', width=2), yaxis='y2'
    ))

    fig.update_layout(
        title='Stock Price (EMA Smoothed) and Sentiment Over Time',
        xaxis_title='Date',
        yaxis=dict(title='Stock Price', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Sentiment Score', titlefont=dict(color='red'), tickfont=dict(color='red'),
                    overlaying='y', side='right'),
        template="plotly_white"
    )

    fig.show()


def plot_stock_price(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['closing_price'], mode='lines',
        name='Raw Stock Price', line=dict(color='gray', width=1), opacity=0.5
    ))

    df['ema_closing_price'] = df['closing_price'].ewm(span=5, adjust=False).mean()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ema_closing_price'], mode='lines',
        name='EMA Smoothed Stock Price', line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title='Stock Price Over Time with EMA Smoothing',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        template="plotly_white"
    )
    fig.show()


def plot_sentiment_score(df, rolling_window=3):
    """
    Plots the smoothed sentiment score over time using a rolling average.

    Parameters:
    - df: pandas DataFrame containing 'date' and 'sentiment' columns.
    - rolling_window: int, optional, default is 3. The window size for the rolling mean.

    Raises:
    - ValueError: If the DataFrame does not contain the required columns.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    if 'sentiment' not in df.columns:
        raise ValueError("DataFrame must contain a 'sentiment' column.")

    df['rolling_sentiment'] = df['sentiment'].rolling(window=rolling_window).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_sentiment'], mode='lines',
        name=f'Smoothed Sentiment Score ({rolling_window}-day Rolling Avg)', line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=f'Smoothed Sentiment Score Over Time ({rolling_window}-day Rolling Avg)',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        template="plotly_white"
    )
    fig.show()



# # ====== Correlation Calculation ======

def calculate_rolling_correlation(df, window=14):

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    if 'rolling_sentiment' not in df.columns:
        df['rolling_sentiment'] = df['sentiment'].rolling(window=window).mean()
    if 'rolling_closing_price' not in df.columns:
        df['rolling_closing_price'] = df['closing_price'].rolling(window=window).mean()

    df['rolling_corr'] = df['rolling_sentiment'].rolling(window=window).corr(df['rolling_closing_price'])
    
    return df[['date', 'rolling_sentiment', 'rolling_closing_price', 'rolling_corr']]

# ====== Main Execution Flow ======


articles = collect_news(api_key)
symbol = 'AAPL'
stock_prices = collect_stock_prices(symbol)
df = analyze_sentiment(articles, stock_prices)


df = extract_features(df)
rolling_correlation_df = calculate_rolling_correlation(df, window=14)
df = pd.merge(df, rolling_correlation_df[['date', 'rolling_corr']], on='date', how='left')
df['rolling_corr'].fillna(0, inplace=True)

conn = sqlite3.connect('sentiment_analysis.db')
df.to_sql('articles', conn, if_exists='replace', index=False)
conn.close()

plot_stock_and_sentiment(df, rolling_window=3)
plot_stock_price(df)
plot_sentiment_score(df, rolling_window=3)