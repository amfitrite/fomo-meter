"""
Build the curated training set for SLM-1.

The dataset is deliberately small, bilingual, and finance-specific. Sentences
are generated from hand-written templates plus a set of difficult curated
examples (negation, slang, sarcasm, macro headlines, Indonesian trading slang).
No live data is copied into the training set, and the golden benchmark is
excluded so the reported evaluation stays honest.

Run:
    python tools/build_slm_dataset.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GOLDEN_FILE = DATA_DIR / "golden_set.json"
OUTPUT_FILE = DATA_DIR / "slm_training.jsonl"
META_FILE = DATA_DIR / "slm_training_meta.json"

RNG = random.Random(20261026)

PROFILES = [
    {
        "name": "Bitcoin", "sym": "BTC", "price": "$80,000",
        "support": "$78,000", "target": "$92,000",
    },
    {
        "name": "Ethereum", "sym": "ETH", "price": "$3,200",
        "support": "$3,000", "target": "$3,800",
    },
    {
        "name": "Gold", "sym": "XAU", "price": "$2,750",
        "support": "$2,600", "target": "$3,000",
    },
    {
        "name": "the S&P 500", "sym": "SPY", "price": "5,500",
        "support": "5,300", "target": "5,900",
    },
    {
        "name": "EUR/USD", "sym": "EURUSD", "price": "1.0850",
        "support": "1.0800", "target": "1.1200",
    },
]

BULL_OPENERS = [
    "{name} breakout above resistance",
    "{name} reclaims the 200 EMA",
    "Whales are accumulating {name}",
    "{name} holds support at {support}",
    "ETF inflows accelerate into {name}",
    "{name} prints a bullish engulfing candle",
    "Rate cut hopes lift {name}",
    "Institutional demand for {name} rises",
    "A short squeeze is brewing in {name}",
    "Futures are rippin on {name}",
    "{name} bounces hard from the dip",
    "{name} pushes to a record high",
    "Analysts upgrade {name} to buy",
    "Strong earnings beat for {name}",
    "Risk-on mood boosts {name}",
    "{name} liquidity grab to the upside",
    "Bids are stacked on {name}",
    "Shorts are about to get smoked on {name}",
    "The {name} trend remains bullish",
    "Spot volume confirms the {name} breakout",
    "Smart money is accumulating {name}",
    "{name} leaves the accumulation range",
]

BULL_CLAUSES = [
    "buy the dip",
    "to the moon",
    "diamond hands",
    "serok kawan",
    "akumulasi mumpung diskon",
    "gaskeun",
    "cuan menanti",
    "target {target}",
    "momentum is strong",
    "not bearish at all",
    "no breakdown in sight",
    "this dip is a gift",
    "green candles everywhere",
    "load the boat",
]

BEAR_OPENERS = [
    "{name} breakdown below support",
    "Whales are dumping {name}",
    "{name} liquidation cascade",
    "Selloff deepens in {name}",
    "Bearish divergence on {name}",
    "Death cross confirms the {name} downtrend",
    "ETF outflows hit {name}",
    "Regulatory crackdown weighs on {name}",
    "A hack drains funds from {name}",
    "{name} loses {support}",
    "Weak earnings and guidance cut at {name}",
    "Rate hike fears pressure {name}",
    "Risk-off selloff hits {name}",
    "Profit taking crushes {name}",
    "Overbought {name} is due for a correction",
    "Paper hands are exiting {name}",
    "FUD is spreading across {name}",
    "There is no bid for {name}",
    "Offers are heavy on {name}",
    "Red candles everywhere on {name}",
    "{name} rejects at resistance",
    "{name} confirms a bull trap",
]

BEAR_CLAUSES = [
    "cut loss",
    "boncos parah",
    "nyangkut di pucuk",
    "jangan beli dulu",
    "protect capital",
    "lower low incoming",
    "this is a bull trap",
    "bloodbath",
    "not bullish at all",
    "dead cat bounce",
    "pump and dump",
    "no support left",
    "sell and stay safe",
    "the trend has flipped down",
]

WAIT_OPENERS = [
    "{name} is sideways",
    "{name} is ranging",
    "A choppy session for {name}",
    "{name} consolidation continues",
    "Mixed signals on {name}",
    "Traders are flat on {name}",
    "Volatility compresses in {name}",
    "No clear direction for {name}",
    "{name} is steady ahead of CPI",
    "{name} waits for the FOMC",
    "Range-bound trading in {name}",
    "{name} fakeout risk is high",
    "Market indecision on {name}",
    "A quiet session for {name}",
    "{name} is coiling before the data",
    "{name} holds the range",
]

WAIT_CLAUSES = [
    "wait and see",
    "wait for confirmation",
    "hold and observe",
    "tunggu breakout",
    "lihat dulu",
    "stay neutral",
    "no trade zone",
    "range play",
    "sabar dulu",
    "no clear signal yet",
    "the market is undecided",
    "flat is a position",
]

HEADLINE_BULL = [
    "{name} surges to a record high on ETF inflows",
    "{name} rallies after a softer inflation print",
    "{name} rebounds from support as buyers return",
    "{name} upgraded to buy at a major bank",
    "{name} climbs as risk appetite improves",
    "{name} gains after strong earnings",
    "Gold hits a record high on safe-haven demand",
    "Euro strengthens as the dollar weakens",
    "Stablecoin inflows signal renewed risk appetite",
    "Exchange reserves fall, supply shock brewing",
    "Regulatory approval clears the path for an ETF",
    "Bitcoin whale accumulation hits a six-month high",
]

HEADLINE_BEAR = [
    "{name} falls below {support} as macro pressure builds",
    "{name} slips as traders de-risk ahead of the Fed",
    "{name} selloff deepens and outflows accelerate",
    "{name} downgraded to sell",
    "{name} drops after weak guidance",
    "{name} faces a regulatory crackdown",
    "Crypto fund outflows hit $443 million",
    "Bitcoin whale deposits hit a six-month high",
    "Miners capitulate as hashprice drops",
    "Hot inflation print triggers a risk-off move",
    "Euro slips as the dollar strengthens",
    "Gold falls as yields rise",
]

HEADLINE_WAIT = [
    "{name} is flat as the market digests earnings",
    "{name} stuck in a range as investors await CPI",
    "S&P 500 futures are flat ahead of payrolls",
    "{name} steady before the FOMC decision",
    "Options open interest points to a volatile move",
    "The SEC delays its ETF decision",
    "{name} traders wait for fresh catalysts",
    "Gold trades sideways as the dollar stabilizes",
    "Euro holds a narrow range before the ECB",
    "Crypto volatility compresses ahead of the weekend",
]

HEADLINE_BULL_EXTRA = [
    "{name} surges as buyers return",
    "{name} rallies after softer inflation data",
    "{name} climbs to a record high",
    "{name} jumps on ETF approval",
    "{name} gains as risk appetite returns",
    "{name} rebounds from support",
    "{name} holds above {support}",
    "{name} breaks out above resistance",
    "{name} outperforms as the dollar weakens",
    "{name} attracts record inflows",
    "{name} upgraded after strong earnings",
    "{name} defies the selloff",
    "{name} recovers its losses",
    "{name} jumps after a short squeeze",
    "{name} rises on safe-haven demand",
    "{name} hits an all-time high",
    "{name} extends its rally",
    "{name} is the best-performing asset this week",
    "{name} beats expectations",
    "{name} outlook turns bullish",
    "{name} sees strong institutional demand",
    "{name} adds {pct}% as buyers step in",
]

HEADLINE_BEAR_EXTRA = [
    "{name} falls below {support}",
    "{name} slips under {support}",
    "{name} drops as the selloff deepens",
    "{name} plunges after a regulatory crackdown",
    "{name} tumbles on weak flows",
    "{name} sinks as risk appetite fades",
    "{name} slumps to a record low",
    "{name} slides on hot inflation data",
    "{name} retreats as yields rise",
    "{name} declines after an ETF rejection",
    "{name} bleeds as outflows accelerate",
    "{name} crashes below support",
    "{name} enters a bear market",
    "{name} is under pressure as the dollar strengthens",
    "{name} loses ground for a third day",
    "{name} faces heavy selling",
    "{name} is the worst-performing asset this week",
    "{name} misses earnings expectations",
    "{name} outlook turns bearish",
    "{name} sees record outflows",
    "{name} wipes out {amount} in market value",
    "{name} drops {pct}% as sellers take control",
]

HEADLINE_WAIT_EXTRA = [
    "{name} trades sideways as investors await the Fed",
    "{name} holds a narrow range",
    "{name} is little changed",
    "{name} consolidates after a volatile week",
    "{name} waits for a catalyst",
    "{name} is stuck in a range",
    "{name} pauses after its recent move",
    "{name} volatility compresses",
    "{name} traders are undecided",
    "{name} is unchanged in early trading",
    "{name} eyes the next data release",
    "{name} options market points to a big move",
    "{name} is in a holding pattern",
    "{name} faces a quiet session",
    "{name} tests resistance without a breakout",
    "{name} tests support without a breakdown",
    "{name} is range-bound",
    "{name} avoids a clear direction",
    "{name} awaits confirmation",
    "How much {name} do you need to retire?",
    "Is {name} a buy right now?",
    "What is next for {name}?",
    "{name} price prediction: where next?",
    "Should you buy {name} before the next halving?",
    "{name} versus gold: which is the better hedge?",
    "Can {name} double from here?",
    "Why is {name} moving today?",
    "What history says about {name} after a drop",
    "{name} outlook: analysts remain divided",
]

EXTRA_VALUES = [
    {"pct": "2", "amount": "$2 billion"},
    {"pct": "3", "amount": "$8 billion"},
    {"pct": "5", "amount": "$12 billion"},
]


# Curated hard cases. These are intentionally short, noisy, and sometimes
# sarcastic, matching the language found on public finance feeds.
CURATED = [
    ("short stops about to get smoked", "BUY"),
    ("futes rippin holy shit lol", "BUY"),
    ("splatterblasting bids here, hope you kids have a raincoat", "BUY"),
    ("bull pendant forming on the chart", "BUY"),
    ("offers heavy, bids pulled", "SELL"),
    ("no bid, liquidity gone", "SELL"),
    ("green across the board today", "BUY"),
    ("red sea today, everything is bleeding", "SELL"),
    ("whales hold back while retail deposits hit a six-month high", "SELL"),
    ("a wallet bug lets rogue servers steal funds", "SELL"),
    ("not bullish, this is a textbook bull trap", "SELL"),
    ("I would not sell here, support is holding", "BUY"),
    ("can't ignore the dump, cut loss", "SELL"),
    ("pump then dump, classic scam", "SELL"),
    ("dead cat bounce, do not buy", "SELL"),
    ("buy the rumor, sell the news", "WAIT"),
    ("record high but profit taking is hitting", "WAIT"),
    ("better than expected, but guidance was cautious", "WAIT"),
    ("serok tipis-tipis, jangan all in", "BUY"),
    ("great, another dump. love it.", "SELL"),
    ("to the moon, said every bagholder", "SELL"),
    ("this is the bottom, trust me bro", "BUY"),
    ("RIP shorts", "BUY"),
    ("bids stacked, asks thin", "BUY"),
    ("no breakout yet, still range-bound", "WAIT"),
    ("whale buying spotted", "BUY"),
    ("whale dumping spotted", "SELL"),
    ("rate cut odds rising", "BUY"),
    ("rate hike odds rising", "SELL"),
    ("inflation cools, risk-on", "BUY"),
    ("hot inflation print, risk-off", "SELL"),
    ("ETF approval confirmed", "BUY"),
    ("ETF application rejected", "SELL"),
    ("SEC delays the decision", "WAIT"),
    ("halving supply shock narrative builds", "BUY"),
    ("miners capitulate", "SELL"),
    ("stablecoin inflows are rising", "BUY"),
    ("exchange outflows suggest accumulation", "BUY"),
    ("exchange inflows suggest sell pressure", "SELL"),
    ("funding rates negative, shorts are crowded", "BUY"),
    ("funding rates overheated, longs are crowded", "SELL"),
    ("open interest flushed, reset incoming", "WAIT"),
    ("liquidity is thin, volatility ahead", "WAIT"),
    ("support is holding for now", "BUY"),
    ("resistance rejected again", "SELL"),
    ("breakout confirmed with volume", "BUY"),
    ("breakdown confirmed with volume", "SELL"),
    ("bull trap confirmed", "SELL"),
    ("bear trap confirmed", "BUY"),
    ("sideways grind, no edge", "WAIT"),
    ("no clear signal, wait", "WAIT"),
    ("wait for CPI before adding risk", "WAIT"),
    ("FOMC minutes are due tomorrow", "WAIT"),
    ("serok kawan, mumpung diskon", "BUY"),
    ("akumulasi tipis-tipis dulu", "BUY"),
    ("boncos parah, cut loss", "SELL"),
    ("nyangkut di pucuk, jangan Average down", "SELL"),
    ("longsor, jangan masuk dulu", "SELL"),
    ("sideways, tunggu breakout", "WAIT"),
    ("konsolidasi, lihat dulu", "WAIT"),
    ("cuan gede di depan mata", "BUY"),
    ("rugi terus, mending cut loss", "SELL"),
    ("hijau semua, gaskeun", "BUY"),
    ("merah semua, hati-hati", "SELL"),
    ("gaskeun tembus resistance", "BUY"),
    ("jangan beli, masih turun", "SELL"),
    ("jangan panik, support masih kuat", "BUY"),
    ("belum breakout, sabar dulu", "WAIT"),
    ("pucuk, ambil profit", "SELL"),
    ("whale beli, kita ikut", "BUY"),
    ("whale jual, hati-hati", "SELL"),
    ("berita bagus tapi harga turun", "SELL"),
    ("berita buruk sudah priced in, siap rebound", "BUY"),
    ("tunggu retest, jangan FOMO", "WAIT"),
    ("pump nya palsu, waspada dump", "SELL"),
    ("volume tipis, rawan fakeout", "WAIT"),
    ("diamond hands, jangan panik", "BUY"),
    ("cut loss dulu, capital is king", "SELL"),
    ("entry aman di support", "BUY"),
    ("sell the rip, trend is down", "SELL"),
    ("buy the dip, trend is up", "BUY"),
    ("chop city, no trade", "WAIT"),
    ("gap up but fading, wait", "WAIT"),
    ("gap down but bid is strong", "BUY"),
    ("short squeeze fuel is building", "BUY"),
    ("longs are trapped above", "SELL"),
    ("liquidity sweep below support then reclaim", "BUY"),
    ("liquidity sweep above resistance then reject", "SELL"),
    ("strong close, buyers in control", "BUY"),
    ("weak close, sellers in control", "SELL"),
    ("doji at resistance, indecision", "WAIT"),
    ("volume dry-up before a move", "WAIT"),
    ("positive funding and rising open interest", "BUY"),
    ("negative funding and falling open interest", "WAIT"),
    ("not a breakout, just a wick", "WAIT"),
    ("not a breakdown, just a wick", "WAIT"),
    ("the dip is being bought aggressively", "BUY"),
    ("the rip is being sold aggressively", "SELL"),
    ("whale wallet moved coins to an exchange", "SELL"),
    ("whale wallet moved coins off an exchange", "BUY"),
    ("stablecoin supply is shrinking", "SELL"),
    ("stablecoin supply is expanding", "BUY"),
    ("halving is priced in, no reaction", "WAIT"),
    ("regulation is clear, institutions can enter", "BUY"),
    ("regulatory uncertainty remains", "WAIT"),
    ("banking stress boosts safe-haven demand", "BUY"),
    ("banking stress eases, risk-on returns", "BUY"),
    ("safe-haven bid is fading", "SELL"),
    ("dollar breaks higher, commodities under pressure", "SELL"),
    ("dollar breaks lower, commodities rally", "BUY"),
    ("yields spike, growth stocks under pressure", "SELL"),
    ("yields fall, growth stocks bid", "BUY"),
    ("earnings beat and raise", "BUY"),
    ("earnings miss and guidance cut", "SELL"),
    ("earnings in line, stock unchanged", "WAIT"),
    ("guidance withdrawn, uncertainty high", "WAIT"),
    ("buyback announced", "BUY"),
    ("dividend cut", "SELL"),
    ("merger arbitrage spread is tight", "WAIT"),
    ("acquisition premium announced", "BUY"),
    ("IPO priced above range", "BUY"),
    ("IPO postponed", "SELL"),
    ("short interest is elevated", "BUY"),
    ("short interest is collapsing", "WAIT"),
    ("insider buying", "BUY"),
    ("insider selling", "SELL"),
    ("volume is drying up", "WAIT"),
    ("volume expansion confirms the move", "BUY"),
    ("volume expansion confirms the breakdown", "SELL"),
    ("price is above the 200-day moving average", "BUY"),
    ("price is below the 200-day moving average", "SELL"),
    ("golden cross on the daily", "BUY"),
    ("death cross on the daily", "SELL"),
    ("higher high, higher low", "BUY"),
    ("lower high, lower low", "SELL"),
    ("double top at resistance", "SELL"),
    ("double bottom at support", "BUY"),
    ("head and shoulders breakdown", "SELL"),
    ("inverse head and shoulders breakout", "BUY"),
    ("ascending triangle breakout", "BUY"),
    ("descending triangle breakdown", "SELL"),
    ("bull flag continuation", "BUY"),
    ("bear flag continuation", "SELL"),
    ("cup and handle breakout", "BUY"),
    ("rising wedge breakdown", "SELL"),
    ("falling wedge breakout", "BUY"),
    ("overbought RSI", "SELL"),
    ("oversold RSI", "BUY"),
    ("MACD bullish crossover", "BUY"),
    ("MACD bearish crossover", "SELL"),
    ("price consolidates below resistance", "WAIT"),
    ("price consolidates above support", "WAIT"),
    ("breakout retest holding", "BUY"),
    ("breakout retest failing", "SELL"),
    ("fakeout above resistance", "SELL"),
    ("fakeout below support", "BUY"),
    ("breaking resistance with massive volume", "BUY"),
    ("breaking support with massive volume", "SELL"),
    ("time to buy the dip and long", "BUY"),
    ("I am long here, breakout confirmed", "BUY"),
    ("long the breakout with volume", "BUY"),
    ("longs are in control", "BUY"),
    ("shorts are trapped below", "BUY"),
    ("massive volume breakout, buy the dip", "BUY"),
    ("buy the dip and add to longs", "BUY"),
    ("breakout above resistance, long", "BUY"),
    ("price is breaking out, not breaking down", "BUY"),
    ("resistance is breaking, buyers in control", "BUY"),
    ("support is breaking, sellers in control", "SELL"),
    ("not a breakdown, a liquidity sweep", "BUY"),
    ("not a breakout, a liquidity sweep", "SELL"),
    ("I am short here, breakdown confirmed", "SELL"),
    ("shorts are in control", "SELL"),
    ("long squeeze, flush the longs", "SELL"),
    ("short squeeze, flush the shorts", "BUY"),
]


def load_golden_texts() -> set[str]:
    if not GOLDEN_FILE.exists():
        return set()
    with GOLDEN_FILE.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {re.sub(r"\s+", " ", str(row.get("text", "")).strip().lower()) for row in rows}


def main() -> int:
    golden = load_golden_texts()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(text: str, label: str, source: str, lang: str = "en") -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        key = text.lower()
        if key in seen or key in golden:
            return
        seen.add(key)
        rows.append({"text": text, "label": label, "source": source, "lang": lang})

    # Template-generated directional examples.
    for profile in PROFILES:
        for opener in RNG.sample(BULL_OPENERS, 12):
            for clause in RNG.sample(BULL_CLAUSES, 3):
                add(
                    f"{opener.format(**profile)}, {clause.format(**profile)}.",
                    "BUY",
                    "template_bull",
                    "mixed",
                )
        for opener in RNG.sample(BEAR_OPENERS, 12):
            for clause in RNG.sample(BEAR_CLAUSES, 3):
                add(
                    f"{opener.format(**profile)}, {clause.format(**profile)}.",
                    "SELL",
                    "template_bear",
                    "mixed",
                )
        for opener in RNG.sample(WAIT_OPENERS, 10):
            for clause in RNG.sample(WAIT_CLAUSES, 2):
                add(
                    f"{opener.format(**profile)}, {clause}.",
                    "WAIT",
                    "template_wait",
                    "mixed",
                )
        for template in RNG.sample(HEADLINE_BULL, 4):
            add(template.format(**profile), "BUY", "headline_bull", "en")
        for template in RNG.sample(HEADLINE_BEAR, 4):
            add(template.format(**profile), "SELL", "headline_bear", "en")
        for template in RNG.sample(HEADLINE_WAIT, 3):
            add(template.format(**profile), "WAIT", "headline_wait", "en")
        for template in HEADLINE_BULL_EXTRA:
            values = {**profile, **RNG.choice(EXTRA_VALUES)}
            add(template.format(**values), "BUY", "headline_bull", "en")
        for template in HEADLINE_BEAR_EXTRA:
            values = {**profile, **RNG.choice(EXTRA_VALUES)}
            add(template.format(**values), "SELL", "headline_bear", "en")
        for template in HEADLINE_WAIT_EXTRA:
            add(template.format(**profile), "WAIT", "headline_wait", "en")

    for text, label in CURATED:
        lang = "id" if any(word in text.lower() for word in ("yang", "jangan", "kawan", "dulu", "masih", "tipis", "parah", "cuan", "rugi", "hijau", "merah", "tunggu", "sabar", "waspada")) else "en"
        add(text, label, "curated", lang)

    RNG.shuffle(rows)
    counts = Counter(row["label"] for row in rows)
    lang_counts = Counter(row["lang"] for row in rows)
    if min(counts.values()) < 50:
        print("Dataset terlalu kecil atau tidak seimbang:", dict(counts), file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "version": "SLM-1",
        "generated_at": "2026-10-26",
        "total": len(rows),
        "counts": dict(counts),
        "languages": dict(lang_counts),
        "sources": dict(Counter(row["source"] for row in rows)),
        "golden_excluded": True,
        "note": "Curated/templated finance data. No live feed data is stored here.",
    }
    with META_FILE.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(rows)} examples to {OUTPUT_FILE}")
    print("Labels:", dict(counts))
    print("Languages:", dict(lang_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
