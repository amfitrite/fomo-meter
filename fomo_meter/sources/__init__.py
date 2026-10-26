"""Public source adapters used by the sentiment engine."""

from __future__ import annotations

from typing import List

from fomo_meter.classifier import FastCrowdClassifier
from fomo_meter.relevance_gate import FinancialRelevanceGate
from fomo_meter.sources.base import BaseSource
from fomo_meter.sources.market import MarketSource
from fomo_meter.sources.news import NewsSource
from fomo_meter.sources.reddit import RedditSource
from fomo_meter.sources.stocktwits import StockTwitsSource
from fomo_meter.sources.x_twitter import XSource
from fomo_meter.sources.yahoo_wire import YahooWireSource

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
