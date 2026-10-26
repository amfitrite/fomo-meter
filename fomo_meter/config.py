"""
FOMO Meter — central configuration.

All paths are resolved relative to the project root so the application can be
started from a desktop shortcut, a terminal, or a different working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

APP_NAME = "FOMO Meter"
APP_VERSION = "2.0.0"
APP_TAGLINE = "Crowd Psychology Radar"
APP_DESCRIPTION = (
    "Radar sentimen publik multi-sumber dengan gate anti-halusinasi, "
    "klasifikasi slang trading Indonesia/Inggris, dan model saraf lokal Sentiment SLM."
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
WEB_DIR = ROOT / "fomo_meter" / "web"
LOG_DIR = ROOT / "logs"
SCREENSHOT_DIR = ROOT / "screenshots"
MODELS_DIR = ROOT / "models"
HISTORY_FILE = DATA_DIR / "history.json"

HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8085

# Network behaviour ---------------------------------------------------------
REQUEST_TIMEOUT = 8
SCAN_CACHE_TTL = 60          # seconds a completed scan is reused
SOURCE_CACHE_TTL = 90        # default per-source cache
NEWS_CACHE_TTL = 300
MARKET_CACHE_TTL = 60
STALE_AFTER = 180            # feed older than this is considered stale
PARTIAL_AFTER = 60           # feed older than this is marked partial in UI
MAX_TEXT_LENGTH = 4000
MAX_HISTORY = 250

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 FOMOMeter/2.0"
)

# Ticker universe -----------------------------------------------------------
# StockTwits uses the public symbol endpoint. News uses Google News RSS.
# Market context is optional and is never allowed to influence sentiment.
TICKERS: Dict[str, Dict[str, Any]] = {
    "BTCUSD": {
        "label": "Bitcoin",
        "name": "Bitcoin / USD",
        "short": "BTC",
        "stocktwits": "BTC.X",
        "news_query": "Bitcoin",
        "news_terms": ["bitcoin", "btc", "crypto"],
        "subreddit": "CryptoCurrency",
        "market": {"kind": "coingecko", "id": "bitcoin", "yahoo": "BTC-USD"},
        "fear_greed": True,
        "decimals": 0,
    },
    "ETHUSD": {
        "label": "Ethereum",
        "name": "Ethereum / USD",
        "short": "ETH",
        "stocktwits": "ETH.X",
        "news_query": "Ethereum",
        "news_terms": ["ethereum", "eth", "ether", "crypto"],
        "subreddit": "ethtrader",
        "market": {"kind": "coingecko", "id": "ethereum", "yahoo": "ETH-USD"},
        "fear_greed": True,
        "decimals": 2,
    },
    "XAUUSD": {
        "label": "Gold",
        "name": "Gold Spot / USD",
        "short": "XAU",
        "stocktwits": "XAUUSD",
        "news_query": "gold price",
        "news_terms": ["gold", "xau", "bullion", "precious metal"],
        "subreddit": "Gold",
        "market": {"kind": "yahoo", "yahoo": "GC=F"},
        "fear_greed": False,
        "decimals": 2,
    },
    "SPY": {
        "label": "S&P 500",
        "name": "SPDR S&P 500 ETF",
        "short": "SPY",
        "stocktwits": "SPY",
        "news_query": "S&P 500 stock market",
        "news_terms": ["s&p", "sp500", "s&p 500", "spy", "stock market", "wall street", "stocks"],
        "subreddit": "wallstreetbets",
        "market": {"kind": "yahoo", "yahoo": "SPY"},
        "fear_greed": False,
        "decimals": 2,
    },
    "EURUSD": {
        "label": "EUR / USD",
        "name": "Euro / US Dollar",
        "short": "EUR",
        "stocktwits": "EURUSD",
        "news_query": "EUR USD forex",
        "news_terms": ["eurusd", "eur/usd", "euro", "dollar", "forex", "ecb"],
        "subreddit": "Forex",
        "market": {"kind": "yahoo", "yahoo": "EURUSD=X"},
        "fear_greed": False,
        "decimals": 4,
    },
}

DEFAULT_TICKER = "BTCUSD"

SOURCE_META: List[Dict[str, Any]] = [
    {
        "id": "stocktwits",
        "name": "StockTwits",
        "icon": "ST",
        "kind": "Crowd",
        "description": "Pesan publik trader dari API StockTwits.",
        "requires_auth": False,
    },
    {
        "id": "news",
        "name": "News Wire",
        "icon": "NW",
        "kind": "News",
        "description": "Headline finansial dari Google News RSS.",
        "requires_auth": False,
    },
    {
        "id": "yahoo",
        "name": "Yahoo Wire",
        "icon": "YH",
        "kind": "News",
        "description": "Headline per simbol dari Yahoo Finance RSS.",
        "requires_auth": False,
    },
    {
        "id": "reddit",
        "name": "Reddit",
        "icon": "RD",
        "kind": "Forum",
        "description": "Thread publik subreddit. Beberapa jaringan memblokir akses (403).",
        "requires_auth": False,
    },
    {
        "id": "x",
        "name": "X / Twitter",
        "icon": "X",
        "kind": "Social",
        "description": "Memerlukan X API Bearer Token melalui environment variable.",
        "requires_auth": True,
    },
]

NEWS_LOCALE = {"hl": "en-US", "gl": "US", "ceid": "US:en"}

# In-house neural model (Sentiment SLM) ---------------------------------------------
# This is not a wrapper around Ollama, OpenAI, or any hosted model. The small
# hashed-feature neural language model is trained locally from the curated
# dataset below and loaded from local weights.
SLM_WEIGHTS = MODELS_DIR / "sentiment_slm.npz"
SLM_META = MODELS_DIR / "sentiment_slm_meta.json"
SLM_DATASET = DATA_DIR / "slm_training.jsonl"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"
GOLDEN_FILE = DATA_DIR / "golden_set.json"
SLM_FEATURE_DIM = 2048
SLM_MIN_CONFIDENCE = 0.50
SLM_VERSION = "Sentiment SLM"
SLM_ARCHITECTURE = "Hashed n-gram features → 128 → 64 → 3 (softmax)"
RETRAIN_MIN_GOLDEN_ACC = 0.90

DISCLAIMER = (
    "FOMO Meter adalah alat riset. Data berasal dari sumber publik dan dapat "
    "salah, tertunda, atau tidak lengkap. Bukan saran investasi."
)


def ensure_dirs() -> None:
    """Create the folders the app writes to."""
    for folder in (DATA_DIR, ASSETS_DIR, WEB_DIR, LOG_DIR, SCREENSHOT_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def ticker_payload() -> List[Dict[str, str]]:
    """Small payload used by the GUI ticker selector."""
    return [
        {"id": key, "label": value["label"], "short": value["short"]}
        for key, value in TICKERS.items()
    ]
