import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import snscrape.modules.twitter as sntwitter
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Initialize NLTK
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class AppConfig:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'},
        },
        'handlers': {
            'console': {'class': 'logging.StreamHandler', 'formatter': 'standard'},
            'file': {'class': 'logging.FileHandler', 'filename': BASE_DIR / 'app.log', 'formatter': 'standard'}
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'INFO'
        }
    }

class TwitterCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def collect(self, start_date, end_date, max_tweets=500):
        keywords = [
            "Rwanda transport", "Kigali bus", "distance fare",
            "RURA transport", "public transport Rwanda"
        ]
        
        tweets = []
        
        for keyword in keywords:
            query = f"{keyword} since:{start_date.date()} until:{end_date.date()}"
            
            try:
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
                    if i >= max_tweets:
                        break
                    
                    tweets.append({
                        'source': 'twitter',
                        'id': str(tweet.id),
                        'text': tweet.content,
                        'date': tweet.date,
                        'url': tweet.url,
                        'username': tweet.user.username,
                        'user_location': tweet.user.location,
                        'language': self._detect_language(tweet.content)
                    })
                
            except Exception as e:
                self.logger.error(f"Error collecting tweets for '{keyword}': {str(e)}")
        
        return tweets
    
    def _detect_language(self, text):
        text = text.lower()
        if any(word in text for word in ['ubus', 'ikintu', 'kuri']):
            return 'rw'
        elif any(word in text for word in ['le', 'la', 'les']):
            return 'fr'
        return 'en'

class NewsScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.logger = logging.getLogger(__name__)
    
    def collect(self, max_articles=20):
        articles = []
        urls = [
            "https://www.newtimes.co.rw/search/node/transport",
            "https://www.ktpress.rw/search/?q=transport"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for article in soup.find_all('article')[:max_articles]:
                    title = article.find('h2').text if article.find('h2') else "No title"
                    link = article.find('a')['href'] if article.find('a') else url
                    
                    article_page = requests.get(link, headers=self.headers)
                    article_soup = BeautifulSoup(article_page.text, 'html.parser')
                    paragraphs = article_soup.find_all('p')
                    text = ' '.join([p.text for p in paragraphs])
                    
                    articles.append({
                        'source': 'news',
                        'text': f"{title}. {text}",
                        'date': datetime.now(),
                        'url': link,
                        'title': title,
                        'language': 'en'
                    })
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {str(e)}")
        
        return articles

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, text):
        try:
            blob = TextBlob(text)
            lang = self.detect_language(text)
            
            if lang in ['en', 'fr']:
                polarity = blob.sentiment.polarity
                return self._interpret_polarity(polarity)
            
            scores = self.sia.polarity_scores(text)
            return self._interpret_vader(scores['compound'])
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            return 'neutral'
    
    def detect_language(self, text):
        try:
            text = text.lower()
            if any(word in text for word in ['ubus', 'ikintu', 'kuri']):
                return 'rw'
            elif any(word in text for word in ['le', 'la', 'les']):
                return 'fr'
            return 'en'
        except:
            return 'en'
    
    def _interpret_polarity(self, polarity):
        if polarity > 0.2:
            return 'positive'
        elif polarity < -0.2:
            return 'negative'
        return 'neutral'
    
    def _interpret_vader(self, compound):
        if compound > 0.05:
            return 'positive'
        elif compound < -0.05:
            return 'negative'
        return 'neutral'

class Dashboard:
    def __init__(self, data):
        self.app = Dash(__name__)
        self.data = data
        self._prepare_data()
        self._setup_layout()
        self._register_callbacks()
    
    def _prepare_data(self):
        self.df = pd.DataFrame(self.data)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['date_day'] = self.df['date'].dt.date
        self.time_series = self.df.groupby(['date_day', 'sentiment']).size().unstack(fill_value=0)
        self.source_dist = self.df['source'].value_counts().reset_index()
        self.source_dist.columns = ['source', 'count']
    
    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Rwanda Transport Fare Sentiment Analysis", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            dcc.Tabs([
                dcc.Tab(label='Overview', children=[
                    html.Div([
                        dcc.Graph(
                            figure=px.line(
                                self.time_series, 
                                x=self.time_series.index, 
                                y=self.time_series.columns,
                                title='Sentiment Trend Over Time'
                            ),
                            style={'width': '100%', 'height': '400px'}
                        ),
                        
                        html.Div([
                            dcc.Graph(
                                figure=px.pie(
                                    self.source_dist,
                                    names='source',
                                    values='count',
                                    title='Data Sources Distribution'
                                ),
                                style={'width': '50%', 'display': 'inline-block'}
                            ),
                            
                            dcc.Graph(
                                figure=px.histogram(
                                    self.df,
                                    x='source',
                                    color='sentiment',
                                    title='Sentiment by Source'
                                ),
                                style={'width': '50%', 'display': 'inline-block'}
                            )
                        ])
                    ])
                ]),
                
                dcc.Tab(label='Raw Data', children=[
                    html.Div(
                        style={'height': '800px', 'overflowY': 'scroll'},
                        children=[html.Pre(self.df.to_string())]
                    )
                ])
            ])
        ])
    
    def run(self):
        self.app.run_server(host='0.0.0.0', port=8050)

def main():
    logging.basicConfig(**AppConfig.LOGGING)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting application")
        
        collector = DataCollector()
        sentiment_analyzer = SentimentAnalyzer()
        
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        logger.info("Collecting data...")
        twitter_data = TwitterCollector().collect(start_date, end_date)
        news_data = NewsScraper().collect()
        all_data = twitter_data + news_data
        
        logger.info("Analyzing data...")
        processed_data = []
        for item in all_data:
            item['sentiment'] = sentiment_analyzer.analyze(item['text'])
            processed_data.append(item)
        
        df = pd.DataFrame(processed_data)
        output_file = AppConfig.PROCESSED_DATA_DIR / 'sentiment_results.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        logger.info("Launching dashboard...")
        Dashboard(processed_data).run()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
