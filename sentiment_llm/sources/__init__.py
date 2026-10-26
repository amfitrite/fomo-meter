"""Public source adapters used by the sentiment engine."""

from __future__ import annotations

from typing import List

from sentiment_llm.classifier import FastCrowdClassifier
from sentiment_llm.relevance_gate import FinancialRelevanceGate
from sentiment_llm.sources.base import BaseSource
from sentiment_llm.sources.market import MarketSource
from sentiment_llm.sources.news import NewsSource
from sentiment_llm.sources.reddit import RedditSource
from sentiment_llm.sources.stocktwits import StockTwitsSource
from sentiment_llm.sources.x_twitter import XSource
from sentiment_llm.sources.yahoo_wire import YahooWireSource

__all__ = [
    "BaseSource",
    "MarketSource",
    "NewsSource",
    "RedditSource",
    "StockTwitsSource",
    "XSource",
    "YahooWireSource",
    "build_default_sources",
]


def build_default_sources(
    classifier: FastCrowdClassifier,
    relevance_gate: FinancialRelevanceGate,
) -> List[BaseSource]:
    sources: List[BaseSource] = [
        StockTwitsSource(),
        NewsSource(),
        YahooWireSource(),
        RedditSource(),
        XSource(),
    ]
    for source in sources:
        source.attach_nlp(classifier, relevance_gate)
    return sources
