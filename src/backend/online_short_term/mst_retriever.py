from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from online_short_term.mst_store import MSTStore


RECENT_KWS = ["刚才", "刚刚", "最近", "现在", "当前", "后来", "之后", "recent", "just now", "now", "later", "after"]
SUMMARY_KWS = [
    "总结",
    "概括",
    "到目前为止",
    "主要发生",
    "整个",
    "全程",
    "summarize",
    "summary",
    "recap",
    "overview",
    "overall",
    "so far",
    "everything so far",
    "what happened so far",
]
SUMMARY_TOP_K = 16
COUNT_KWS = ["一共", "总共", "一共有", "总共有", "多少", "几个", "几次", "几幅", "几张", "count", "how many", "total"]
LOCATION_TRACKING_KWS = [
    "在哪",
    "哪里",
    "哪儿",
    "放哪",
    "放在",
    "位置",
    "where",
    "place",
    "placed",
    "put",
    "left",
    "leave",
]
BM25_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "around",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "context",
    "could",
    "did",
    "do",
    "does",
    "access",
    "for",
    "forget",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "once",
    "on",
    "or",
    "place",
    "placed",
    "previous",
    "put",
    "query",
    "range",
    "remember",
    "tell",
    "that",
    "the",
    "there",
    "this",
    "time",
    "to",
    "used",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "you",
}
TOKEN_ALIASES = {
    "phone": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "smartphone": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "mobile": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "cellphone": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "iphone": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "手机": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "电话": {"phone", "smartphone", "mobile", "cellphone", "iphone", "手机", "电话"},
    "bag": {"bag", "backpack", "pack", "包", "背包"},
    "backpack": {"bag", "backpack", "pack", "包", "背包"},
    "包": {"bag", "backpack", "pack", "包", "背包"},
    "背包": {"bag", "backpack", "pack", "包", "背包"},
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _tokens(text: str) -> set[str]:
    text = (text or "").lower()
    words = set(re.findall(r"[a-z0-9_]+", text))
    chinese = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    words.update(chinese)
    words.update("".join(chinese[i : i + 2]) for i in range(max(0, len(chinese) - 1)))
    return {token for token in words if token.strip()}


def _token_sequence(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    chinese = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
    tokens.extend(chinese)
    tokens.extend("".join(chinese[i : i + 2]) for i in range(max(0, len(chinese) - 1)))
    return [token for token in tokens if token.strip()]


def _bm25_terms(text: str, *, expand_aliases: bool = False) -> list[str]:
    terms: list[str] = []
    seen_aliases: set[str] = set()
    for token in _token_sequence(text):
        if token in BM25_STOPWORDS or token.isdigit() or len(token) <= 1:
            continue
        if expand_aliases and token in TOKEN_ALIASES:
            for alias in sorted(TOKEN_ALIASES[token]):
                if alias not in seen_aliases:
                    terms.append(alias)
                    seen_aliases.add(alias)
            continue
        terms.append(token)
    return terms


def _retrieval_query_text(question: str) -> str:
    text = str(question or "")
    if "answer:" in text.lower():
        text = re.split(r"answer:\s*", text, maxsplit=1, flags=re.IGNORECASE)[-1]
    text = re.sub(r"\b\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+(?:\.\d+)?s\b", " ", text, flags=re.IGNORECASE)
    return text.strip()


def _overlap_score(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def _is_summary_query(question: str) -> bool:
    text = (question or "").lower()
    return any(keyword.lower() in text for keyword in SUMMARY_KWS)


def _is_location_tracking_query(question: str) -> bool:
    text = (question or "").lower()
    return any(keyword.lower() in text for keyword in LOCATION_TRACKING_KWS)


def _event_text(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(key, "") or "")
        for key in (
            "retrieval_text",
            "event_caption_refined",
            "event_caption_fast",
            "event_caption_placeholder",
            "transcript",
        )
    )


def _summary_event_quality(event: dict[str, Any]) -> float:
    text = _event_text(event)
    refined = 1.0 if event.get("event_caption_refined") else 0.0
    diff = min(max(_safe_float(event.get("diff_score")), 0.0), 1.0)
    duration = max(0.0, _safe_float(event.get("end_time")) - _safe_float(event.get("start_time")))
    text_bonus = min(len(text) / 1000.0, 1.0)
    return 0.45 * refined + 0.25 * text_bonus + 0.20 * diff + 0.10 * min(duration / 10.0, 1.0)


def _choose_evenly(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or not indices:
        return []
    if len(indices) <= limit:
        return list(indices)
    if limit == 1:
        return [indices[0]]
    chosen: list[int] = []
    last_pos = len(indices) - 1
    for i in range(limit):
        pos = round(i * last_pos / (limit - 1))
        value = indices[int(pos)]
        if value not in chosen:
            chosen.append(value)
    for value in indices:
        if len(chosen) >= limit:
            break
        if value not in chosen:
            chosen.append(value)
    return sorted(chosen)


class _BM25Corpus:
    def __init__(self, documents: list[str]) -> None:
        self.doc_terms = [_bm25_terms(document) for document in documents]
        self.term_frequencies = [Counter(terms) for terms in self.doc_terms]
        self.lengths = [len(terms) for terms in self.doc_terms]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for terms in self.doc_terms:
            self.document_frequency.update(set(terms))
        self.total_docs = len(documents)

    def scores(self, query: str) -> list[float]:
        query_terms = list(dict.fromkeys(_bm25_terms(_retrieval_query_text(query), expand_aliases=True)))
        if not query_terms or not self.doc_terms or self.avg_len <= 0:
            return [0.0 for _ in self.doc_terms]

        k1 = 1.5
        b = 0.75
        scores: list[float] = []
        for term_frequency, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                if not df:
                    continue
                idf = math.log(1.0 + (self.total_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1.0 - b + b * length / self.avg_len)
                score += idf * (frequency * (k1 + 1.0)) / denominator
            scores.append(score)

        max_score = max(scores) if scores else 0.0
        if max_score <= 0.0:
            return [0.0 for _ in self.doc_terms]
        return [round(score / max_score, 6) for score in scores]


def _continuity_scores(events: list[dict[str, Any]], bm25_scores: list[float], question: str) -> list[float]:
    if not _is_location_tracking_query(question) or not events:
        return [0.0 for _ in events]
    scores = [0.0 for _ in events]
    for index, bm25_score in enumerate(bm25_scores):
        if bm25_score < 0.45:
            continue
        end = _safe_float(events[index].get("end_time"))
        if index + 1 < len(events):
            next_start = _safe_float(events[index + 1].get("start_time"))
            if 0.0 <= next_start - end <= 20.0:
                scores[index + 1] = max(scores[index + 1], bm25_score * 0.75)
        start = _safe_float(events[index].get("start_time"))
        if index > 0:
            prev_end = _safe_float(events[index - 1].get("end_time"))
            if 0.0 <= start - prev_end <= 10.0:
                scores[index - 1] = max(scores[index - 1], bm25_score * 0.35)
    return [round(score, 6) for score in scores]


class MSTRetriever:
    def __init__(self, store: MSTStore) -> None:
        self.store = store
        self.mst_version = -1
        self.events: list[dict[str, Any]] = []
        self._bm25_cache_key: tuple[str, int, int] | None = None
        self._bm25_corpus: _BM25Corpus | None = None

    def refresh_if_needed(self) -> None:
        state = self.store.get_state()
        version = int(state.get("mst_version", 0) or 0)
        if version != self.mst_version:
            self.events = self.store.load_events()
            self.mst_version = version

    def search(
        self,
        question: str,
        top_k: int = 5,
        cache_context: dict[str, Any] | None = None,
        include_archive: bool = False,
    ) -> list[dict[str, Any]]:
        if include_archive:
            state = self.store.get_state()
            version = int(state.get("archive_version", state.get("mst_version", 0)) or 0)
            events = self.store.load_archive_events()
            bm25_cache_key = ("archive", version, len(events))
        else:
            self.refresh_if_needed()
            events = self.events
            bm25_cache_key = ("active", self.mst_version, len(events))
        if not events:
            return []
        cache_context = cache_context or {}
        latest_time = max(_safe_float(event.get("end_time")) for event in events)
        recent_query = any(keyword.lower() in (question or "").lower() for keyword in RECENT_KWS)
        count_query = any(keyword.lower() in (question or "").lower() for keyword in COUNT_KWS)
        summary_query = _is_summary_query(question)
        if summary_query and not count_query:
            return self._timeline_summary(events, top_k=max(int(top_k or 0), SUMMARY_TOP_K))
        event_texts = [
            " ".join(
                [
                    str(event.get("retrieval_text", "")),
                    str(event.get("event_caption_refined", "")),
                    str(event.get("transcript", "")),
                    str(event.get("event_caption_fast", "")),
                    str(event.get("event_caption_placeholder", "")),
                    str(event.get("boundary_reason", "")),
                ]
            )
            for event in events
        ]
        if self._bm25_cache_key != bm25_cache_key or self._bm25_corpus is None:
            self._bm25_corpus = _BM25Corpus(event_texts)
            self._bm25_cache_key = bm25_cache_key
        bm25_values = self._bm25_corpus.scores(question)
        continuity_values = _continuity_scores(events, bm25_values, question)
        scored = []
        for index, event in enumerate(events):
            text = event_texts[index]
            lexical = _overlap_score(_retrieval_query_text(question), text)
            bm25 = bm25_values[index] if index < len(bm25_values) else 0.0
            continuity = continuity_values[index] if index < len(continuity_values) else 0.0
            age = max(0.0, latest_time - _safe_float(event.get("end_time")))
            recency = 1.0 / (1.0 + age / 60.0)
            cache_score = self._cache_context_score(event, cache_context)
            diff = min(max(_safe_float(event.get("diff_score")), 0.0), 1.0)
            if recent_query:
                score = 0.36 * bm25 + 0.14 * lexical + 0.24 * recency + 0.14 * cache_score + 0.08 * continuity + 0.04 * diff
            else:
                score = 0.42 * bm25 + 0.18 * lexical + 0.16 * recency + 0.12 * cache_score + 0.08 * continuity + 0.04 * diff
            item = dict(event)
            item["score"] = round(float(score), 4)
            item["bm25_score"] = round(float(bm25), 4)
            item["lexical_score"] = round(float(lexical), 4)
            item["recency_score"] = round(float(recency), 4)
            item["cache_score"] = round(float(cache_score), 4)
            item["continuity_score"] = round(float(continuity), 4)
            item["diff_score_component"] = round(float(diff), 4)
            item["source"] = "M_st"
            item["caption_source"] = item.get("caption_source") or ("refined" if item.get("event_caption_refined") else "placeholder")
            scored.append(item)
        scored.sort(key=lambda item: (-float(item.get("score", 0.0)), -float(item.get("end_time", 0.0))))
        if count_query and (recent_query or summary_query):
            limit = max(int(top_k), min(len(scored), 16))
            recent_scored = sorted(scored, key=lambda item: _safe_float(item.get("end_time", 0.0)), reverse=True)[:limit]
            recent_scored.sort(key=lambda item: _safe_float(item.get("start_time", 0.0)))
            return recent_scored
        if summary_query:
            return self._timeline_summary(events, top_k=max(int(top_k or 0), SUMMARY_TOP_K))
        return scored[: max(1, int(top_k))]

    def _timeline_summary(self, events: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        by_window: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            start = _safe_float(event.get("start_time"))
            window = int(start // 30.0)
            by_window.setdefault(window, []).append(event)

        representatives: list[dict[str, Any]] = []
        for window in sorted(by_window):
            candidates = sorted(
                by_window[window],
                key=lambda item: (-_summary_event_quality(item), _safe_float(item.get("start_time"))),
            )
            representatives.append(candidates[0])

        if not representatives:
            return []

        limit = max(1, int(top_k or SUMMARY_TOP_K))
        indices = list(range(len(representatives)))
        keep = {idx for idx in indices[: min(4, len(indices))]}
        if indices:
            keep.add(indices[-1])
        remaining = limit - len(keep)
        if remaining > 0:
            middle = [idx for idx in indices if idx not in keep]
            keep.update(_choose_evenly(middle, remaining))

        selected = [representatives[idx] for idx in sorted(keep)[:limit]]
        out: list[dict[str, Any]] = []
        for rank, event in enumerate(selected):
            item = dict(event)
            item["score"] = round(max(0.0, 1.0 - rank * 0.01), 4)
            item["source"] = "M_st"
            item["summary_timeline_rank"] = rank
            item["summary_timeline_fallback"] = True
            item["caption_source"] = item.get("caption_source") or ("refined" if item.get("event_caption_refined") else "placeholder")
            out.append(item)
        return out

    def _cache_context_score(self, event: dict[str, Any], cache_context: dict[str, Any]) -> float:
        if not cache_context:
            return 0.0
        score = 0.0
        start = _safe_float(event.get("start_time"))
        end = _safe_float(event.get("end_time"), start)
        for window in cache_context.get("referenced_time_ranges", []) or []:
            w_start = _safe_float(window.get("start"))
            w_end = _safe_float(window.get("end"), w_start)
            if max(start, w_start) <= min(end, w_end):
                score = max(score, 0.8)
                break
            distance = min(abs(start - w_end), abs(end - w_start))
            if distance <= 10:
                score = max(score, 0.45)
        entity_terms = []
        for entity in cache_context.get("referenced_entities", []) or []:
            if not isinstance(entity, dict):
                continue
            for key in ("canonical_name", "name", "entity_key"):
                if entity.get(key):
                    entity_terms.append(str(entity[key]).lower())
        if entity_terms:
            haystack = str(event.get("retrieval_text", "")).lower()
            if any(term in haystack for term in entity_terms):
                score = max(score, 0.7)
        return score
