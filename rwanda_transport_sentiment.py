import logging
import logging.config
import random
import feedparser                        # pip instapipll feedparser
import praw                              # pip install praw
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, quote_plus

import nltk
import pandas as pd
from dash import Dash, dash_table, dcc, html
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── NLTK one-time downloads ──────────────────────────────────────────────────
for _res in ['vader_lexicon', 'punkt', 'averaged_perceptron_tagger']:
    nltk.download(_res, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
#  App Configuration
# ─────────────────────────────────────────────────────────────────────────────
class AppConfig:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'

    for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {'class': 'logging.StreamHandler', 'formatter': 'standard'},
            'file':    {
                'class': 'logging.FileHandler',
                'filename': str(BASE_DIR / 'app.log'),
                'formatter': 'standard'
            }
        },
        'root': {'handlers': ['console', 'file'], 'level': 'INFO'}
    }

    # ── Reddit API credentials ────────────────────────────────────────────
    # Create a FREE script-type app at https://www.reddit.com/prefs/apps
    # Then replace the placeholders below OR export as environment variables.
    REDDIT_CLIENT_ID     = 'YOUR_CLIENT_ID'       # <── replace
    REDDIT_CLIENT_SECRET = 'YOUR_CLIENT_SECRET'   # <── replace
    REDDIT_USER_AGENT    = 'RwandaTransportBot/1.0'


# ─────────────────────────────────────────────────────────────────────────────
#  Language Detection
# ─────────────────────────────────────────────────────────────────────────────
class LanguageDetector:
    RW_WORDS = {'ubus', 'ikintu', 'kuri', 'ndishimye', 'cyane', 'mwiza'}
    FR_WORDS = {'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont'}

    @classmethod
    def detect(cls, text: str) -> str:
        words = set(text.lower().split())
        if words & cls.RW_WORDS:
            return 'rw'
        if len(words & cls.FR_WORDS) >= 2:
            return 'fr'
        return 'en'


# ─────────────────────────────────────────────────────────────────────────────
#  Reddit Collector  (replaces broken snscrape / Twitter)
# ─────────────────────────────────────────────────────────────────────────────
class RedditCollector:
    """
    Collects posts and comments from Reddit using the official praw library.
    Fully compatible with Python 3.12.
    Free API — sign up at https://www.reddit.com/prefs/apps
    """

    KEYWORDS = [
        'Rwanda transport', 'Kigali bus', 'RURA transport',
        'distance fare Rwanda', 'Rwanda public transport',
        'Kigali fare', 'Rwanda bus fare'
    ]

    def __init__(self):
        self.logger   = logging.getLogger(__name__)
        self._reddit  = None

    def _get_client(self):
        if self._reddit:
            return self._reddit

        import os
        cid     = os.getenv('REDDIT_CLIENT_ID',     AppConfig.REDDIT_CLIENT_ID)
        csecret = os.getenv('REDDIT_CLIENT_SECRET', AppConfig.REDDIT_CLIENT_SECRET)

        if 'YOUR_' in cid:
            raise ValueError(
                "Reddit credentials missing.\n"
                "  Option A: Edit AppConfig.REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET\n"
                "  Option B: Set env vars REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET\n"
                "  Free sign-up: https://www.reddit.com/prefs/apps"
            )

        self._reddit = praw.Reddit(
            client_id=cid,
            client_secret=csecret,
            user_agent=AppConfig.REDDIT_USER_AGENT,
        )
        return self._reddit

    def collect(self, start_date: datetime, end_date: datetime,
                max_posts: int = 200) -> list:
        posts = []
        try:
            reddit   = self._get_client()
            start_ts = start_date.timestamp()
            end_ts   = end_date.timestamp()

            for keyword in self.KEYWORDS:
                try:
                    for sub in reddit.subreddit('all').search(
                        keyword, limit=max_posts, sort='new'
                    ):
                        if not (start_ts <= sub.created_utc <= end_ts):
                            continue

                        text = f"{sub.title}. {sub.selftext or ''}".strip()
                        posts.append(self._record(
                            'reddit', sub.id, text,
                            datetime.fromtimestamp(sub.created_utc),
                            f"https://reddit.com{sub.permalink}",
                            str(sub.author)
                        ))

                        sub.comments.replace_more(limit=0)
                        for c in sub.comments.list()[:10]:
                            if not c.body or c.body == '[deleted]':
                                continue
                            posts.append(self._record(
                                'reddit_comment', c.id, c.body,
                                datetime.fromtimestamp(c.created_utc),
                                f"https://reddit.com{sub.permalink}",
                                str(c.author)
                            ))
                except Exception as e:
                    self.logger.error(f"Reddit search error '{keyword}': {e}")

        except ValueError as e:
            self.logger.warning(f"Skipping Reddit: {e}")

        self.logger.info(f"Reddit: {len(posts)} items collected.")
        return posts

    @staticmethod
    def _record(source, rid, text, date, url, username):
        return {
            'source': source, 'id': str(rid), 'text': text,
            'date': date, 'url': url, 'title': text[:80],
            'username': username, 'user_location': 'N/A',
            'language': LanguageDetector.detect(text),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  RSS Feed Collector
# ─────────────────────────────────────────────────────────────────────────────
class RSSCollector:
    """Google News RSS + local outlet feeds. No API key required."""

    KEYWORDS = [
        'Rwanda transport fare', 'Kigali bus fare',
        'RURA Rwanda transport', 'Rwanda public transport',
    ]
    DIRECT_FEEDS = [
        'https://www.newtimes.co.rw/rss.xml',
        'https://www.ktpress.rw/feed/',
    ]

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def collect(self, start_date: datetime, end_date: datetime,
                max_items: int = 50) -> list:
        articles = []

        for kw in self.KEYWORDS:
            url = (
                "https://news.google.com/rss/search"
                f"?q={quote_plus(kw)}&hl=en-RW&gl=RW&ceid=RW:en"
            )
            articles += self._parse(url, max_items, start_date, end_date)

        for url in self.DIRECT_FEEDS:
            articles += self._parse(url, max_items, start_date, end_date)

        self.logger.info(f"RSS: {len(articles)} items collected.")
        return articles

    def _parse(self, url, max_items, start_date, end_date) -> list:
        out = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                pub  = self._date(entry)
                if pub and not (start_date <= pub <= end_date):
                    continue
                title   = entry.get('title',   'No title')
                summary = entry.get('summary', '')
                link    = entry.get('link',    url)
                text    = f"{title}. {summary}".strip()
                out.append({
                    'source': 'rss', 'id': link, 'text': text,
                    'date': pub or datetime.now(), 'url': link,
                    'title': title, 'username': entry.get('author', 'N/A'),
                    'user_location': 'N/A',
                    'language': LanguageDetector.detect(text),
                })
        except Exception as e:
            self.logger.warning(f"RSS parse error ({url}): {e}")
        return out

    @staticmethod
    def _date(entry) -> datetime | None:
        try:
            from email.utils import parsedate_to_datetime
            raw = entry.get('published') or entry.get('updated')
            if raw:
                return parsedate_to_datetime(raw).replace(tzinfo=None)
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  News Scraper  (BeautifulSoup)
# ─────────────────────────────────────────────────────────────────────────────
class NewsScraper:
    URLS = [
        'https://www.newtimes.co.rw/search/node/transport',
        'https://www.ktpress.rw/search/?q=transport',
    ]
    HEADERS = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def collect(self, max_articles: int = 20) -> list:
        articles = []
        for base_url in self.URLS:
            try:
                r    = requests.get(base_url, headers=self.HEADERS, timeout=10)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')

                for art in soup.find_all('article')[:max_articles]:
                    h2    = art.find('h2')
                    title = h2.text.strip() if h2 else 'No title'
                    a     = art.find('a')
                    link  = a['href'] if a else base_url

                    if link.startswith('/'):
                        p    = urlparse(base_url)
                        link = f"{p.scheme}://{p.netloc}{link}"

                    body = self._body(link)
                    text = f"{title}. {body}".strip()

                    articles.append({
                        'source': 'news', 'id': link, 'text': text,
                        'date': datetime.now(), 'url': link, 'title': title,
                        'username': 'N/A', 'user_location': 'N/A',
                        'language': LanguageDetector.detect(text),
                    })
            except Exception as e:
                self.logger.error(f"NewsScraper error ({base_url}): {e}")

        self.logger.info(f"News scraper: {len(articles)} articles.")
        return articles

    def _body(self, url: str) -> str:
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            return ' '.join(p.text for p in soup.find_all('p'))
        except Exception as e:
            self.logger.warning(f"Body fetch failed ({url}): {e}")
            return ''


# ─────────────────────────────────────────────────────────────────────────────
#  Data Collector — orchestrates all sources
# ─────────────────────────────────────────────────────────────────────────────
class DataCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reddit = RedditCollector()
        self.rss    = RSSCollector()
        self.news   = NewsScraper()

    def collect_all(self, start_date: datetime, end_date: datetime,
                    max_reddit: int = 200,
                    max_rss:    int = 50,
                    max_news:   int = 20) -> list:
        self.logger.info("Collecting Reddit data…")
        reddit_data = self.reddit.collect(start_date, end_date, max_reddit)

        self.logger.info("Collecting RSS feed data…")
        rss_data = self.rss.collect(start_date, end_date, max_rss)

        self.logger.info("Collecting news scrape data…")
        news_data = self.news.collect(max_news)

        combined = reddit_data + rss_data + news_data
        self.logger.info(f"Total collected: {len(combined)} items.")
        return combined


# ─────────────────────────────────────────────────────────────────────────────
#  Sentiment Analyser
# ─────────────────────────────────────────────────────────────────────────────
class SentimentAnalyzer:
    def __init__(self):
        self.sia    = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)

    def analyze(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return 'neutral'
        try:
            lang = LanguageDetector.detect(text)
            if lang in ('en', 'fr'):
                p = TextBlob(text).sentiment.polarity
                return 'positive' if p > 0.2 else ('negative' if p < -0.2 else 'neutral')
            c = self.sia.polarity_scores(text)['compound']
            return 'positive' if c > 0.05 else ('negative' if c < -0.05 else 'neutral')
        except Exception as e:
            self.logger.error(f"Sentiment error: {e}")
            return 'neutral'

    def batch_analyze(self, items: list) -> list:
        for item in items:
            item['sentiment'] = self.analyze(item.get('text', ''))
        return items


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard
# ─────────────────────────────────────────────────────────────────────────────
class Dashboard:
    COLORS = {'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}

    def __init__(self, data: list):
        self.app  = Dash(__name__)
        self.data = data
        self._prep()
        self._layout()

    def _prep(self):
        import plotly.express as px
        self._px = px
        self.df  = pd.DataFrame(self.data)

        for col in ['date', 'sentiment', 'source', 'text', 'url', 'language']:
            if col not in self.df.columns:
                self.df[col] = 'N/A'

        self.df['date']     = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['date_day'] = self.df['date'].dt.date

        self.ts = (
            self.df.groupby(['date_day', 'sentiment'])
            .size().unstack(fill_value=0).reset_index()
            if not self.df.empty
            else pd.DataFrame(columns=['date_day', 'positive', 'neutral', 'negative'])
        )

        def vc(col):
            s = self.df[col].value_counts().reset_index()
            s.columns = [col, 'count']
            return s

        self.src_dist  = vc('source')
        self.sent_dist = vc('sentiment')

    def _layout(self):
        px      = self._px
        ts_cols = [c for c in self.ts.columns if c != 'date_day']

        self.app.layout = html.Div(
            style={'fontFamily': 'Arial, sans-serif', 'padding': '20px',
                   'background': '#f5f6fa'},
            children=[
                html.H1("🚌 Rwanda Transport Fare — Sentiment Dashboard",
                        style={'textAlign': 'center', 'color': '#2c3e50',
                               'marginBottom': '30px'}),

                html.Div(style={'display': 'flex', 'gap': '16px',
                                'marginBottom': '30px'},
                         children=self._cards()),

                dcc.Tabs(children=[

                    dcc.Tab(label='📊 Overview', children=[
                        dcc.Graph(style={'height': '400px'}, figure=px.line(
                            self.ts, x='date_day', y=ts_cols,
                            title='Sentiment Trend Over Time',
                            color_discrete_map=self.COLORS,
                            labels={'value': 'Count', 'date_day': 'Date',
                                    'variable': 'Sentiment'}
                        )),
                        html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                            dcc.Graph(style={'flex': '1'}, figure=px.pie(
                                self.src_dist, names='source', values='count',
                                title='Data Sources')),
                            dcc.Graph(style={'flex': '1'}, figure=px.bar(
                                self.sent_dist, x='sentiment', y='count',
                                color='sentiment', color_discrete_map=self.COLORS,
                                title='Sentiment Distribution')),
                        ]),
                        dcc.Graph(style={'height': '400px'}, figure=px.histogram(
                            self.df, x='source', color='sentiment', barmode='group',
                            color_discrete_map=self.COLORS,
                            title='Sentiment by Source')),
                    ]),

                    dcc.Tab(label='🌍 Language', children=[
                        dcc.Graph(style={'height': '550px'}, figure=px.sunburst(
                            self.df, path=['language', 'sentiment'],
                            title='Sentiment by Language'))
                    ]),

                    dcc.Tab(label='📡 Sources', children=[
                        dcc.Graph(style={'height': '550px'}, figure=px.treemap(
                            self.df, path=['source', 'sentiment'],
                            title='Source & Sentiment Breakdown'))
                    ]),

                    dcc.Tab(label='🗂 Raw Data', children=[
                        html.Div(style={'marginTop': '20px'}, children=[
                            dash_table.DataTable(
                                data=self.df[[
                                    'date', 'source', 'sentiment',
                                    'language', 'text', 'url'
                                ]].to_dict('records'),
                                columns=[{'name': c, 'id': c} for c in
                                         ['date', 'source', 'sentiment',
                                          'language', 'text', 'url']],
                                page_size=20,
                                filter_action='native',
                                sort_action='native',
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left', 'maxWidth': '320px',
                                    'overflow': 'hidden',
                                    'textOverflow': 'ellipsis',
                                    'whiteSpace': 'nowrap',
                                },
                                style_header={
                                    'backgroundColor': '#2c3e50',
                                    'color': 'white', 'fontWeight': 'bold'
                                },
                                style_data_conditional=[
                                    {'if': {'filter_query': '{sentiment} = positive'},
                                     'backgroundColor': '#d5f5e3'},
                                    {'if': {'filter_query': '{sentiment} = negative'},
                                     'backgroundColor': '#fadbd8'},
                                    {'if': {'filter_query': '{sentiment} = neutral'},
                                     'backgroundColor': '#fef9e7'},
                                ],
                                tooltip_data=[
                                    {'text': {'value': str(r.get('text', '')),
                                              'type': 'markdown'}}
                                    for r in self.df.to_dict('records')
                                ],
                                tooltip_duration=None,
                            )
                        ])
                    ]),
                ]),
            ]
        )

    def _cards(self):
        total  = len(self.df)
        counts = self.df['sentiment'].value_counts().to_dict()
        style  = {
            'flex': '1', 'padding': '20px', 'borderRadius': '10px',
            'textAlign': 'center', 'color': 'white', 'fontWeight': 'bold',
            'boxShadow': '0 2px 6px rgba(0,0,0,.15)'
        }
        return [
            html.Div([html.H2(total),                          html.P("Total Items")],
                     style={**style, 'background': '#2c3e50'}),
            html.Div([html.H2(counts.get('positive', 0)),      html.P("✅ Positive")],
                     style={**style, 'background': '#2ecc71'}),
            html.Div([html.H2(counts.get('neutral',  0)),      html.P("➖ Neutral")],
                     style={**style, 'background': '#f39c12'}),
            html.Div([html.H2(counts.get('negative', 0)),      html.P("❌ Negative")],
                     style={**style, 'background': '#e74c3c'}),
        ]

    def run(self, debug: bool = False):
        print("\n  Dashboard ready →  http://localhost:8050\n")
        self.app.run(host='0.0.0.0', port=8050, debug=debug)


# ─────────────────────────────────────────────────────────────────────────────
#  Built-in sample data  (always loads if scrapers return nothing)
# ─────────────────────────────────────────────────────────────────────────────
def _sample_data() -> list:
    rows = [
        ("Rwanda transport fares are too high for daily commuters.",  'news'),
        ("RURA has done a great job regulating bus fares.",           'rss'),
        ("Kigali bus service improved significantly this year.",      'reddit'),
        ("Public transport in Rwanda needs more investment.",         'news'),
        ("Ubus uri mwiza cyane kuri iki gihe. Ndishimye.",           'reddit'),
        ("Les tarifs des transports au Rwanda semblent équitables.", 'rss'),
        ("Distance-based fare system is fair for all passengers.",   'news'),
        ("Long routes are very expensive for low-income people.",    'reddit'),
        ("New buses on Kigali–Musanze route are comfortable.",       'rss'),
        ("Transport operators are complaining about fuel costs.",    'news'),
        ("RURA should enforce distance-based fares strictly.",       'reddit'),
        ("Tap-to-pay on Kigali buses is a great innovation.",        'rss'),
    ]
    analyzer = SentimentAnalyzer()
    now      = datetime.now()
    return [
        {
            'source': src, 'id': str(i), 'text': text,
            'date': now - timedelta(days=random.randint(0, 6)),
            'url': 'https://example.com', 'title': text[:60],
            'username': 'demo_user', 'user_location': 'Kigali, Rwanda',
            'language': LanguageDetector.detect(text),
            'sentiment': analyzer.analyze(text),
        }
        for i, (text, src) in enumerate(rows)
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.config.dictConfig(AppConfig.LOGGING)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting application")

        start_date = datetime.now() - timedelta(days=7)
        end_date   = datetime.now()

        collector      = DataCollector()
        all_data       = collector.collect_all(start_date, end_date)
        analyzer       = SentimentAnalyzer()
        processed_data = analyzer.batch_analyze(all_data)

        # Save to CSV
        out = AppConfig.PROCESSED_DATA_DIR / 'sentiment_results.csv'
        pd.DataFrame(processed_data).to_csv(out, index=False)
        logger.info(f"Results saved → {out}")

        # Fall back to sample data so dashboard is never blank
        if not processed_data:
            logger.warning("No live data — using built-in sample data.")
            processed_data = _sample_data()

        logger.info("Launching dashboard…")
        Dashboard(processed_data).run(debug=False)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
