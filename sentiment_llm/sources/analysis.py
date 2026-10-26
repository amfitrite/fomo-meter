"""Aggregate raw public posts into a gate-compliant SourceResult."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sentiment_llm.classifier import Classification, FastCrowdClassifier
from sentiment_llm.relevance_gate import FinancialRelevanceGate
from sentiment_llm.schema import Post, SourceResult, SourceStatus, Stance


@dataclass
class AnalyzedPost:
    post: Post
    classification: Classification
    effective_stance: Stance
    weight: float
    sanitized_text: str
    user_label_agrees: bool = False
    user_label_conflict: bool = False


def _excerpt(text: str, limit: int = 240) -> str:
    """Return an exact substring (no added ellipsis) suitable for evidence."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    if " " in clipped:
        clipped = clipped[: clipped.rfind(" ")]
    return clipped.rstrip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _target_position(text: str, terms: Optional[Iterable[str]]) -> Optional[int]:
    """Position of the first target mention, or None when it is absent."""
    if not terms:
        return None
    lower = (text or "").lower()
    positions = [lower.find(str(term).lower()) for term in terms if str(term).strip()]
    positions = [position for position in positions if position >= 0]
    return min(positions) if positions else None


def aggregate_posts(
    posts: Iterable[Post],
    classifier: FastCrowdClassifier,
    relevance_gate: FinancialRelevanceGate,
    *,
    source_id: str,
    name: str,
    icon: str,
    kind: str,
    provenance: Optional[str] = None,
    symbol_verified: bool = True,
    min_analyzed: int = 1,
    max_posts: int = 24,
    target_terms: Optional[Iterable[str]] = None,
) -> SourceResult:
    """Filter, classify, and aggregate posts without inventing any values."""
    post_list = [post for post in posts if post.text and post.text.strip()]
    analyzed: List[AnalyzedPost] = []
    rejected = 0
    injection_count = 0
    buy_weight = 0.0
    sell_weight = 0.0
    wait_weight = 0.0
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for post in post_list:
        decision = relevance_gate.evaluate(post.text)
        if not decision.is_relevant:
            rejected += 1
            continue
        if decision.injection_detected:
            injection_count += 1

        clean = decision.sanitized_text or post.text
        classification = classifier.classify(clean)
        user_label = post.user_sentiment if post.user_sentiment in (Stance.BUY, Stance.SELL) else None

        effective = classification.stance
        weight = 1.0
        label_agrees = False
        label_conflict = False
        if user_label:
            if classification.stance == Stance.WAIT:
                effective = user_label
                label_agrees = True
            elif classification.stance == user_label:
                effective = user_label
                weight = 1.25
                label_agrees = True
            else:
                effective = Stance.WAIT
                weight = 0.6
                label_conflict = True
                neutral_count += 1

        if classification.is_ambiguous and effective in (Stance.BUY, Stance.SELL):
            weight *= 0.85

        # A post with no directional lexicon signal is still counted as
        # analysed, but it must not outvote posts that actually contain a
        # signal. This is especially important for neutral news headlines.
        has_signal = classification.hit_count > 0 or user_label is not None
        if not has_signal:
            weight = 0.0
            effective = Stance.WAIT

        if effective == Stance.BUY:
            buy_weight += weight
            if not label_conflict:
                bullish_count += 1
        elif effective == Stance.SELL:
            sell_weight += weight
            if not label_conflict:
                bearish_count += 1
        else:
            wait_weight += weight
            if not label_conflict:
                neutral_count += 1

        analyzed.append(AnalyzedPost(
            post=post,
            classification=classification,
            effective_stance=effective,
            weight=weight,
            sanitized_text=clean,
            user_label_agrees=label_agrees,
            user_label_conflict=label_conflict,
        ))

    if not analyzed:
        reason = (
            f"{len(post_list)} pesan diterima, tetapi tidak ada yang lolos gate relevansi finansial."
            if post_list else "Sumber tidak mengembalikan pesan."
        )
        result = SourceResult(
            id=source_id,
            name=name,
            icon=icon,
            kind=kind,
            status=SourceStatus.NO_DATA,
            reason=reason,
            stance=Stance.WAIT,
            intensity=0,
            confidence=0.0,
            sample_count=len(post_list),
            analyzed_count=0,
            keywords=[],
            provenance=provenance,
            posts=post_list[:max_posts],
            symbol_verified=symbol_verified,
        )
        return result

    total = buy_weight + sell_weight + wait_weight
    if total <= 0:
        return SourceResult(
            id=source_id,
            name=name,
            icon=icon,
            kind=kind,
            status=SourceStatus.LIVE,
            reason=(
                "Data live tersedia, tetapi tidak ada sinyal arah yang terdeteksi. "
                "Tidak ada stance yang dipaksakan."
            ),
            stance=Stance.WAIT,
            intensity=50,
            confidence=0.0,
            sample_count=len(post_list),
            analyzed_count=len(analyzed),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            keywords=[],
            evidence=None,
            evidence_url=None,
            latest_post_at=max((post.created_at for post in post_list if post.created_at), default=None),
            provenance=provenance,
            posts=post_list[:max_posts],
            symbol_verified=symbol_verified,
            low_sample=len(analyzed) < 3,
            mixed_signals=False,
        )

    ranked = sorted(
        ((Stance.BUY, buy_weight), (Stance.SELL, sell_weight), (Stance.WAIT, wait_weight)),
        key=lambda item: item[1],
        reverse=True,
    )
    top_stance, top_weight = ranked[0]
    second_weight = ranked[1][1]

    mixed = (
        buy_weight > 0 and sell_weight > 0
        and min(buy_weight, sell_weight) / max(buy_weight, sell_weight) >= 0.65
    )
    tie = abs(buy_weight - sell_weight) < 0.01 and buy_weight > 0 and sell_weight > 0
    if tie:
        stance = Stance.WAIT
    else:
        stance = top_stance

    if stance == Stance.WAIT:
        intensity = 50
    else:
        margin = (top_weight - second_weight) / total if total else 0.0
        intensity = max(51, min(99, int(round(50 + margin * 50))))

    confidence = 0.35 + (0.40 * (top_weight / total if total else 0.0))
    confidence += 0.25 * min(1.0, len(analyzed) / 6.0)
    if len(analyzed) < 3:
        confidence -= 0.10
    if mixed:
        confidence -= 0.08
    confidence = round(max(0.0, min(0.95, confidence)), 2)

    # Evidence must prefer a post that supports the final stance, explicitly
    # mentions the requested instrument, and carries a meaningful textual
    # signal. A wall of cashtags is penalised: it is a weaker citation than a
    # post that actually talks about the target.
    candidates = [
        item for item in analyzed
        if item.effective_stance == stance and item.classification.matched_terms
    ]
    if not candidates:
        candidates = [item for item in analyzed if item.classification.matched_terms] or analyzed

    def evidence_key(item: AnalyzedPost):
        position = _target_position(item.post.text, target_terms)
        return (
            position is not None,
            position is not None and position < 80,
            item.effective_stance == stance,
            item.user_label_agrees,
            -min(8, item.post.text.count("$")),
            item.classification.hit_count,
            item.post.created_at,
        )

    candidates.sort(key=evidence_key, reverse=True)
    evidence_item = candidates[0]
    evidence_text = _excerpt(evidence_item.post.text, 240)
    evidence_url = evidence_item.post.url

    keyword_counter: Counter[str] = Counter()
    for item in analyzed:
        for term in item.classification.matched_terms:
            if len(term) > 1:
                keyword_counter[term] += 1
    keywords = [term for term, _count in keyword_counter.most_common(6)]

    latest_post_at = max((post.created_at for post in post_list if post.created_at), default=None)
    status = SourceStatus.LIVE if len(analyzed) >= min_analyzed else SourceStatus.NO_DATA
    reason = None
    if status == SourceStatus.NO_DATA:
        reason = (
            f"Hanya {len(analyzed)} post relevan; minimum {min_analyzed} untuk sumber live. "
            "Tidak ada angka yang dipaksakan."
        )

    return SourceResult(
        id=source_id,
        name=name,
        icon=icon,
        kind=kind,
        status=status,
        reason=reason,
        stance=stance,
        intensity=intensity,
        confidence=confidence,
        sample_count=len(post_list),
        analyzed_count=len(analyzed),
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        keywords=keywords,
        evidence=evidence_text,
        evidence_url=evidence_url,
        latest_post_at=latest_post_at,
        provenance=provenance,
        posts=post_list[:max_posts],
        symbol_verified=symbol_verified,
        low_sample=len(analyzed) < 3,
        mixed_signals=mixed,
    )
