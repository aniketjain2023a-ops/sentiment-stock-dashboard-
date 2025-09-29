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

# ====== API KEYS ======

api_key = '08cbfd9ce468468688c55507dfc9b27b'

# ====== Data Collection ======

def collect_news(api_key):
    url = f'https://newsapi.org/v2/everything?q=stock market&from=2024-11-01&sortBy=popularity&pageSize=100&apiKey={api_key}'
    response = requests.get(url)
    return response.json().get('articles', [])

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

def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_data = []
    for article in articles:
        cleaned_title = preprocess_text(article['title'])
        sentiment_score = analyzer.polarity_scores(cleaned_title)['compound']
        sentiment_data.append({
            'title': article['title'],
            'sentiment': sentiment_score,
            'date': article['publishedAt'],
            'source': article['source']['name'],
            'closing_price': np.random.uniform(100, 200) 
        })
    return pd.DataFrame(sentiment_data)

# ====== Feature Engineering ======

def extract_features(df):
    df['keyword_buy'] = df['title'].str.contains('buy', case=False).astype(int)
    df['keyword_sell'] = df['title'].str.contains('sell', case=False).astype(int)
    df['keyword_crash'] = df['title'].str.contains('crash', case=False).astype(int)
    df['lagged_sentiment'] = df['sentiment'].shift(1)

    df['stock_movement'] = (df['closing_price'].shift(-1) > df['closing_price']).astype(int)
    
    df.dropna(inplace=True)  
    return df

# ====== Visualization Functions ======

def plot_stock_and_sentiment(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    df['rolling_sentiment'] = df['sentiment'].rolling(window=7).mean()
    df['rolling_closing_price'] = df['closing_price'].rolling(window=7).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_closing_price'], mode='lines', 
        name='Stock Price', line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_sentiment'], mode='lines',
        name='Sentiment Score', line=dict(color='red', width=2), yaxis='y2'
    ))

    fig.update_layout(
        title='Stock Price and Sentiment Over Time (Smoothed)',
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
    df['rolling_closing_price'] = df['closing_price'].rolling(window=7).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_closing_price'], mode='lines',
        name='Stock Price', line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Smoothed Stock Price Over Time',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        template="plotly_white"
    )

    fig.show()


def plot_sentiment_score(df):

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df['rolling_sentiment'] = df['sentiment'].rolling(window=7).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['rolling_sentiment'], mode='lines',
        name='Sentiment Score', line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title='Smoothed Sentiment Score Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        template="plotly_white"
    )

    fig.show()


# ====== Main Execution Flow ======

# Step 1: Collect articles
articles = collect_news(api_key)
print(f"Number of articles collected: {len(articles)}")

# Step 2: Analyze sentiment if articles were collected
if len(articles) > 0:
    df = analyze_sentiment(articles)
    print(f"Data after sentiment analysis:\n{df.head()}")
    print(f"Number of rows after sentiment analysis: {df.shape[0]}")

    # Step 3: Save to SQLite
    conn = sqlite3.connect('sentiment_analysis.db')
    df.to_sql('articles', conn, if_exists='replace', index=False)
    conn.close()

    # Step 4: Feature extraction
    df = extract_features(df)
    print(f"Data after feature extraction:\n{df.head()}")
    print(f"Number of rows after feature extraction: {df.shape[0]}")

    # Step 5: Plotting
    if df.shape[0] > 0:
        # plot_stock_and_sentiment(df)
        plot_stock_and_sentiment(df)
        plot_stock_price(df)
        plot_sentiment_score(df)
else:
    print("No data available for analysis.")