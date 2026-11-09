"""Query-aware, bounded evidence assembly shared by text and image packing.

Stored memories stay lossless. Only the model-facing view is budgeted here;
whole observations/actions are selected instead of cutting a caption prefix.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from typing import Any

from online_time import event_time_seconds
from online_query_constraints import actor_fit, overlaps, constraint_window


CONCEPT_GROUPS = (
    ("手机", "电话", "phone", "smartphone", "cellphone", "mobile", "iphone"),
    ("包", "背包", "书包", "bag", "backpack", "rucksack"),
    ("墨镜", "太阳镜", "sunglasses", "shades"),
    ("眼镜", "glasses", "eyeglasses", "spectacles"),
    ("帽子", "遮阳帽", "hat", "cap", "sunhat"),
    ("口罩", "mask", "masked", "facemask"),
    ("外套", "夹克", "jacket", "coat"),
    ("安全带", "seatbelt", "seat belt"),
    ("笔记本", "电脑", "laptop", "computer"),
    ("瓶子", "水瓶", "bottle"),
    ("杯子", "cup", "mug"),
    ("方向盘", "steering wheel"),
    ("贴到耳边", "贴耳", "耳边", "ear", "calling", "call"),
    ("戴上", "穿上", "put on", "puts on", "putting on", "wears", "wearing"),
    ("摘下", "脱下", "remove", "removes", "removing", "take off"),
    ("拿起", "pick up", "picks up", "picking up", "lift", "lifts"),
    ("黑色", "黑衣", "black", "dark"),
    ("白色", "白衣", "white"),
    ("灰色", "gray", "grey"),
    ("蓝色", "blue"),
    ("红色", "red"),
    ("穿戴", "服饰", "衣着", "clothing", "outfit", "appearance", "clothes", "garment", "apparel"),
    ("车辆", "汽车", "vehicle", "vehicles", "car", "cars"),
    ("车流", "交通", "traffic"),
    ("排队", "拥堵", "密集", "queue", "queued", "congested", "dense", "closely spaced"),
    ("空旷", "稀疏", "开阔", "sparse", "clear road", "open road", "far ahead"),
)
TOKEN_ALIASES: dict[str, set[str]] = {}
for _group in CONCEPT_GROUPS:
    for _word in _group:
        TOKEN_ALIASES.setdefault(_word, set()).update(_group)
_STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "when", "where", "which", "did", "does", "do", "he", "she", "it", "and", "or", "of", "to", "in", "on", "at", "with", "for", "please", "give", "evidence", "video", "frame", "time"}


def as_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def query_terms(text: str, *, expand: bool = True) -> set[str]:
    # Return a fresh set: callers may extend it while cached values stay immutable.
    return set(_query_terms_cached(str(text or "").lower(), bool(expand)))


@lru_cache(maxsize=2048)
def _query_terms_cached(text: str, expand: bool) -> frozenset[str]:
    text = str(text or "").lower()
    terms = set(re.findall(r"[a-z][a-z0-9_]*", text)) - _STOP
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        terms.update(run[i:i+n] for n in (2, 3) for i in range(len(run)-n+1))
    if expand:
        for word, aliases in TOKEN_ALIASES.items():
            matched = (re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text)
                       if word.isascii() else word in text)
            if matched:
                terms.update(aliases)
    return frozenset(terms)


def relevance(question: str, value: Any) -> float:
    q = query_terms(question)
    if not q:
        return 0.0
    lexical = len(q & query_terms(as_text(value))) / len(q)
    focus = focus_match(question, value)
    return lexical if focus is None else 0.65 * focus + 0.35 * lexical


_FOCUS_GROUPS = {
    "phone": CONCEPT_GROUPS[0], "bag": CONCEPT_GROUPS[1],
    "sunglasses": CONCEPT_GROUPS[2], "glasses": CONCEPT_GROUPS[3],
    "hat": (*CONCEPT_GROUPS[4], "visor", "遮阳帽"), "mask": CONCEPT_GROUPS[5],
    "coat": CONCEPT_GROUPS[6], "seatbelt": CONCEPT_GROUPS[7],
    "computer": CONCEPT_GROUPS[8], "bottle": CONCEPT_GROUPS[9],
    "cup": CONCEPT_GROUPS[10], "steering": CONCEPT_GROUPS[11],
}


def _has_words(text: str, words) -> bool:
    return _has_words_cached(text.lower(), tuple(words))


@lru_cache(maxsize=4096)
def _has_words_cached(text: str, words: tuple[str, ...]) -> bool:
    # These compounds do not mean a bag.
    text = re.sub(r"包括|包含|包边|包装", "", text)
    return any(bool(re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text))
               if word.isascii() else word in text for word in words)


@lru_cache(maxsize=512)
def _target_question(question: str) -> str:
    # Remove identifying relative clauses only, not requested comparisons such
    # as “有没有戴眼镜”. Keep the stored question/observations unchanged.
    text = re.sub(r"(?:戴着?|佩戴)(?:普通)?(?:眼镜|墨镜|太阳镜)(?:的)?(?:人|男子|男人|女性|女人)", "人", question)
    return re.sub(r"(?:man|woman|person) (?:wearing|with) (?:sun)?glasses(?= who| holding| holds)", "person", text, flags=re.I)


def query_focus(question: str) -> set[str]:
    """Separate the queried object/property from shared actor descriptors."""
    return set(_query_focus_cached(question))


@lru_cache(maxsize=512)
def _query_focus_cached(question: str) -> frozenset[str]:
    question = _target_question(question)
    focus = {name for name, words in _FOCUS_GROUPS.items() if _has_words(question, words)}
    if _has_words(question, ("穿戴", "服饰", "衣着", "穿什么", "穿的", "穿着什么", "clothing", "outfit", "appearance", "clothes")):
        focus.add("clothing")
    if _has_words(question, ("车流", "车辆", "交通", "排队", "前方车", "traffic", "vehicles", "queue", "congestion")):
        focus.add("traffic")
    return frozenset(focus)


def focus_match(question: str, value: Any) -> float | None:
    focus = query_focus(question)
    if not focus:
        return None
    signals = target_signals(question, value)
    return sum({"confirmed": 1.0, "candidate": 0.35}.get(signals.get(name), 0.0) for name in focus) / len(focus)


def target_signals(question: str, value: Any) -> dict[str, str]:
    """Text-only candidate signals, never a claim that an image proves a fact.

    Negation and uncertainty stay local to their target clause. Multiple actors
    are not combined to satisfy a conjunction.
    """
    text = as_text(value.get("description") or value).lower() if isinstance(value, dict) else str(value).lower()
    # Normalize only this scoring view; returned/stored records retain their
    # original wording, including uncertainty and contractions.
    text = re.sub(r"\b(does|did|do|is|are|was|were|has|have|had|can|could|would|should)n[’']t\b", r"\1 not", text)
    clauses = [c for c in re.split(r"[.;。；]|,\s*(?=(?:the |a |another )?(?:man|woman|person)\b)|\bwhile\s+(?=(?:the |a |another )?(?:front|rear|left|right|another|man|woman|person))|\bwhereas\b|\band\s+(?=(?:the |a |another )?(?:front|rear|left|right|another|man|woman|person))", text) if c.strip()]
    # Preserve clause boundaries while preferring explicit requested-actor
    # clauses. An unbound console object must not borrow a seated person's
    # identity merely because both are mentioned in the same observation.
    scored_clauses = [(clause, actor_fit(question, clause)) for clause in clauses]
    strongest_actor = max((fit for _, fit in scored_clauses), default=0)
    clauses = [clause for clause, fit in scored_clauses if fit > 0 and
               (fit == strongest_actor or re.match(r"\s*(?:he|she|his|her)\b|\s*[他她]", clause))]
    result = {}
    for name in query_focus(question):
        words = _FOCUS_GROUPS.get(name, ())
        if name == "clothing":
            words = ("衣", "帽", "眼镜", "墨镜", "口罩", "shirt", "jacket", "coat", "hood", "hat", "visor", "sunglasses", "glasses", "mask", "clothing")
        elif name == "traffic":
            words = ("车", "排队", "道路", "traffic", "car", "cars", "vehicle", "vehicles", "road", "suv", "mpv", "van", "truck", "sedan", "bus")
        states = []
        for clause in clauses:
            if actor_fit(question, clause) == 0:
                continue
            if not _has_words(clause, words):
                if name == "phone" and re.search(r"(?:dark|black|small|handheld) (?:rectangular )?device|rectangular (?:device|object)|矩形设备|长方形设备", clause):
                    states.append("candidate")
                continue
            # Match the target adjacent to the negator, rather than allowing
            # “no glasses” to negate a phone elsewhere in the observation.
            target = "(?:" + "|".join(re.escape(w) for w in words) + ")"
            negative = re.search(r"(?:\bno\b|\bwithout\b|not (?:(?:visibly|currently|actually|clearly) ){0,2}(?:hold(?:s|ing)?|wear(?:s|ing)?|carry(?:ing)?)|未见|没有|未戴|没拿)(?:(?!\b(?:and|but|while|holds?|wears?|holding|wearing)\b|[,;，；]).){0,24}?" + target, clause)
            negative = negative or re.search(target + r"\s*(?:is |was |are |remains )?(?:not (?:(?:clearly|fully|directly) )?visible|not seen|absent|不可见|未见)", clause)
            uncertain = re.search(r"possibly|possible|perhaps|may be|might be|unclear|cannot confirm|could be|not (?:clearly|fully|directly) visible|可能|疑似|无法确认|看不清", clause)
            visibility_uncertain = re.search(target + r"\s*(?:is |was |are |remains )?not (?:clearly|fully|directly) visible", clause)
            states.append("confirmed" if name in {"clothing", "traffic"} else "candidate" if visibility_uncertain else "negative" if negative else "candidate" if uncertain else "confirmed")
        result[name] = next((state for state in ("confirmed", "candidate", "negative") if state in states), "unknown")
    # A multi-person sentence cannot establish that different target objects
    # belong to the same person. Require a single actor clause with all targets.
    if len(result) > 1 and all(state in {"confirmed", "candidate"} for state in result.values()):
        # A single timed record may contain several people or sequential
        # sentences. Only a complete clause establishes a text-level same-
        # observation candidate; pronoun resolution across sentences remains
        # uncertain and must be checked in the image.
        complete = any(all(_has_words(c, _FOCUS_GROUPS.get(name, ())) for name in result) for c in clauses)
        if not complete:
            result = {name: "candidate" if state == "confirmed" else state for name, state in result.items()}
    return result


def actor_binding_fit(question: str, observation: Any, event: dict[str, Any] | None = None) -> float:
    """Use only explicit co-descriptions from the same event to resolve aliases."""
    # Environmental directions describe the scene, not a competing person.
    # Explicit actor/entity fields remain authoritative and are never scrubbed.
    actor_view = observation
    if (isinstance(observation, dict) and not observation.get("actor") and not observation.get("triple")
            and any(key in observation for key in ("before", "after", "from_state", "to_state"))):
        # A state change's entity identifies its object, not the person acting.
        actor_view = observation.get("description") or {key: value for key, value in observation.items() if key != "entity"}
    if not isinstance(observation, dict) or not (observation.get("actor") or observation.get("entity") or observation.get("triple")):
        text = as_text(observation.get("description") or observation) if isinstance(observation, dict) else str(observation)
        text = re.sub(r"\b(?:front|rear|left|right)(?:[- ](?:side|hand))?[- ]+(?:doorway|door|window|console|hand|arm|mirror)\b", "scene fixture", text, flags=re.I)
        text = re.sub(r"(?:左侧|右侧|左边|右边)(?:车门|门口|车窗|窗口|扶手)", "场景设施", text)
        actor_view = text
    fit = actor_fit(question, actor_view)
    if fit in (0.0, 1.0) or not event:
        return fit
    text = as_text(observation).lower()
    for obj in content_fields(event).get("visual_objects", []) or []:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or "").lower()
        if name and _has_words(text, (name,)) and actor_fit(question, as_text(obj)) == 1.0:
            return 1.0
    return fit


# Request grammar and actor words do not identify an object or an action.
_INSPECTION_STOP = {"show", "find", "look", "see", "give", "get", "image", "images", "picture", "pictures", "clip", "clips", "person", "people", "man", "woman", "someone", "happen", "happened", "happening", "doing", "happens", "second", "seconds", "please", "scene", "event", "events", "moment", "moments", "compare", "earlier", "later", "then", "now", "before", "after", "recent", "recently", "these", "those", "this", "that", "front", "rear", "left", "right"}


def inspection_terms(question: str) -> set[str]:
    text = _target_question(question).lower()
    text = re.sub(r"什么时候|什么时间", " ", text)
    if not query_focus(question) and re.search(r"\bwhat\b|什么|怎么|如何", text):
        return set()
    text = re.sub(r"有没有|是否|请给|给我|画面|图片|视频|发生了什么|发生什么|做什么|这个人|那个人", " ", text)
    return query_terms(text, expand=False) - _INSPECTION_STOP


def _word_roots(value: Any) -> set[str]:
    return set(_word_roots_cached(as_text(value)))


@lru_cache(maxsize=2048)
def _word_roots_cached(text: str) -> frozenset[str]:
    terms = query_terms(text)
    roots = set(terms)
    for term in terms:
        if not term.isascii():
            continue
        if len(term) > 4 and term.endswith(("ies", "ied")):
            roots.add(term[:-3] + "y")
        for suffix in ("ing", "ed", "es", "s"):
            if term.endswith(suffix) and len(term) > len(suffix) + 2:
                stem = term[:-len(suffix)]
                roots.update((stem, stem + "e"))
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    roots.add(stem[:-1])
    return frozenset(roots)


def _field_matches(question: str, value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_field_matches(question, item) for item in value)
    if isinstance(value, dict):
        return any(_field_matches(question, value[key]) for key in ("name", "label", "entity", "object") if value.get(key))
    text = str(value or "").strip().lower()
    if not text:
        return False
    # Single-character CJK actions are meaningful only as complete native
    # fields; query tokenization otherwise deliberately uses larger ngrams.
    if re.fullmatch(r"[\u4e00-\u9fff]+", text) and text in question.lower():
        return True
    return bool(_word_roots(question) & _word_roots(text))


# Auxiliary/modal grammar constrains scope without a catalog of action verbs.
_POLARITY_GRAMMAR = {"am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "to", "not", "no", "never", "without"}


def _local_predicate_polarity(question: str, text: str, predicate: Any = None) -> tuple[bool, bool]:
    """Scope lexical polarity to the queried predicate in its own clause."""
    text = re.sub(r"\b(does|did|do|is|are|was|were|has|have|had|can|could|would|should)n[’']t\b", r"\1 not", text.lower())
    requested = _word_roots(question)
    targets = _word_roots(predicate) - _POLARITY_GRAMMAR if predicate is not None else inspection_terms(question)
    matched_targets = targets & requested
    targets = matched_targets or targets
    if not targets:
        return False, False
    clauses = re.split(r"[.;!?。；！？,，]|\b(?:while|whereas|but|and)\b|但是|同时|并且|但|而|并", text)
    states = []
    for clause in clauses:
        spans = [match.span() for match in re.finditer(r"[a-z][a-z0-9_]*", clause)
                 if _word_roots(match.group()) & targets]
        for target in targets:
            if not target.isascii():
                spans.extend(match.span() for match in re.finditer(re.escape(target), clause))
        if not spans:
            continue
        prefix = clause[:min(start for start, _ in spans)]
        def governs(pattern):
            for marker in re.finditer(pattern, prefix):
                gap = prefix[marker.end():]
                if re.search(r"[\u4e00-\u9fff]", marker.group()):
                    if not gap.strip():
                        return True
                else:
                    words = re.findall(r"[a-z]+", gap)
                    if all(word in _POLARITY_GRAMMAR or word.endswith("ly") for word in words) and not re.search(r"[\u4e00-\u9fff]", gap):
                        return True
            return False
        negative = governs(r"\b(?:not|no|never|without)\b|没有|并未|未曾|不")
        uncertain = governs(r"\b(?:possibly|possible|perhaps|may|might|could|unclear)\b|cannot confirm|可能|疑似|无法确认|看不清")
        suffix = clause[min(end for _, end in spans):]
        uncertain = uncertain or bool(re.search(r"\b(?:possibly|possible|perhaps|unclear)\b|cannot confirm|可能|疑似|无法确认|看不清", suffix))
        states.append("negative" if negative else "uncertain" if uncertain else "positive")
    negative = bool(states) and all(state == "negative" for state in states)
    uncertain = "uncertain" in states or ("negative" in states and "positive" in states)
    return negative, uncertain


def _observation_polarity(question: str, observation: Any, text: str, predicate: Any = None) -> tuple[bool, bool]:
    negative, uncertain = _local_predicate_polarity(question, text, predicate)
    if not isinstance(observation, dict):
        return negative, uncertain
    polarity = str(observation.get("polarity") or "").lower()
    certainty = str(observation.get("certainty") or observation.get("confidence_label") or "").lower()
    if polarity in {"negative", "negated", "false", "absent"} or observation.get("negated") is True:
        negative = True
    elif polarity in {"positive", "affirmed", "true", "present"} or observation.get("negated") is False:
        negative = False
    if certainty in {"uncertain", "possible", "unknown", "low"} or polarity == "unknown":
        uncertain = True
    elif certainty in {"certain", "confirmed", "high"}:
        uncertain = False
    return negative, uncertain


@lru_cache(maxsize=512)
def query_requests_action(question: str) -> bool:
    """A manner question requests an action without naming its predicate."""
    auxiliaries = "|".join(sorted(_POLARITY_GRAMMAR - {"not", "no", "never", "without", "be", "been", "being"}))
    return bool(re.search(r"\bhow\s+(?:" + auxiliaries + r")\b|怎么|如何|怎样|以何种方式", question, re.I))


def _native_predicate(action: Any, objects: Any, actor: Any = None) -> str:
    """Remove native object/actor labels before looking for predicate proof."""
    text = as_text(action).lower()
    excluded = _word_roots(objects) | (_word_roots(actor) if actor else set()) | _INSPECTION_STOP | _POLARITY_GRAMMAR
    text = re.sub(r"[a-z][a-z0-9_]*", lambda match: " " if _word_roots(match.group()) & excluded else match.group(), text)
    for term in sorted((word for word in excluded if not word.isascii()), key=len, reverse=True):
        text = text.replace(term, " ")
    return text.strip()


def native_action_level(question: str, observation: Any) -> int:
    """Inspection category, separate from lexical topic scores or proof."""
    binding = action_support(question, observation)
    if not binding or not binding["object_match"] or binding["status"] == "negative":
        return 0
    if binding["action_match"]:
        return 2 if binding["status"] == "supported" else 1
    return 1 if query_requests_action(question) else 0


def action_support(question: str, observation: Any) -> dict[str, Any] | None:
    """Rank native actor/action/object bindings separately from scene text.

    This is textual inspection support, not visual proof. Missing fields leave
    the legacy text path available; explicit bindings cannot borrow an object
    mentioned only in a description of the surrounding scene.
    """
    if not isinstance(observation, dict):
        return None
    triple = observation.get("triple")
    triple = triple if isinstance(triple, (list, tuple)) and len(triple) >= 3 else []
    action = observation.get("action") or observation.get("predicate") or (triple[1] if triple else None)
    objects = observation.get("objects") or observation.get("object") or observation.get("target") or (triple[2] if triple else None)
    state = any(key in observation for key in ("before", "after", "from_state", "to_state"))
    if not objects and state:
        objects = observation.get("entity")
    if not action and state:
        action = observation.get("after") or observation.get("to_state") or observation.get("attribute")
    if not action or not objects:
        return None
    object_match = _field_matches(question, objects)
    predicate = _native_predicate(action, objects, observation.get("actor") or (triple[0] if triple else None))
    action_match = bool(predicate) and _field_matches(question, predicate)
    score = (0.7 if object_match else 0.0) + (0.3 if action_match else 0.0)
    text = str(observation.get("description") or "").lower()
    negative, uncertain = _observation_polarity(question, observation, text, predicate)
    status = ("negative" if negative and object_match else
              "candidate" if object_match and (uncertain or not action_match) else
              "supported" if object_match else "background")
    return {"score": 0.0 if negative else score * (0.35 if uncertain else 1.0),
            "status": status, "object_match": object_match, "action_match": action_match,
            "action": action, "objects": objects, "uncertain": uncertain}


def inspection_signals(question: str, observation: Any) -> dict[str, str]:
    """Shared bounded candidate status for catalogued and arbitrary objects."""
    signals = target_signals(question, observation)
    binding = action_support(question, observation)
    if signals:
        if binding and binding["status"] == "negative":
            return {name: "negative" for name in signals}
        if binding and not binding["object_match"]:
            return {name: "background" for name in signals}
        if binding and binding["status"] == "candidate":
            return {name: "candidate" if value == "confirmed" else value for name, value in signals.items()}
        return signals
    if not inspection_terms(question):
        return {}
    if binding:
        status = binding["status"]
        return {"binding": "confirmed" if status == "supported" else "candidate" if status == "candidate" else "negative" if status == "negative" else "background"}
    text = query_actor_text(question, observation)
    matches = inspection_terms(question) & _word_roots(text)
    # Unstructured text can nominate a candidate, never establish the binding.
    if not matches:
        return {"text": "unknown"}
    negative, _ = _observation_polarity(question, observation, text)
    return {"text": "negative" if negative else "candidate"}


def observation_match(question: str, observation: Any, event: dict[str, Any] | None = None) -> float:
    text = query_actor_text(question, observation)
    signals = inspection_signals(question, observation)
    fit = actor_binding_fit(question, observation, event)
    if not fit:
        return 0.0
    binding = action_support(question, observation)
    if binding:
        quality = binding["score"]
        if (query_requests_action(question) and binding["object_match"]
                and not binding["uncertain"] and binding["status"] != "negative"):
            # A "how" question does not supply a predicate to match. A native
            # action is as inspectable as a clear prose observation; it is not
            # a higher class that preempts all other representations.
            quality = 1.0
        return fit * (0.95 * quality + 0.05 * relevance(question, text))
    if signals:
        values = [{"confirmed": 1.0, "candidate": 0.35}.get(v, 0.0) for v in signals.values()]
        # Conjunctions favour all conditions on the same observation, rather
        # than letting one highly repeated condition dominate the score.
        quality = min(values) if len(values) > 1 else values[0]
        return fit * (0.95 * quality + 0.05 * relevance(question, text))
    return fit * relevance(question, text)


def query_actor_text(question: str, value: Any) -> str:
    """Score clauses about the requested actor, retaining original records."""
    if not isinstance(value, dict) or value.get("actor") or value.get("entity"):
        return as_text(value)
    text = value.get("description") or value.get("action")
    if not isinstance(text, str):
        return as_text(value)
    clauses = re.split(r"(?<=[.!?。！？])\s+|[;；]|\bwhile\b|\bwhereas\b", text, flags=re.I)
    scored = [(clause, actor_fit(question, clause)) for clause in clauses if clause.strip()]
    strongest = max((score for _,score in scored), default=0)
    return " ".join(clause for clause,score in scored if score > 0 and
                    (score == strongest or re.match(r"\s*(?:he|she|his|her|他|她)\b", clause, re.I)))


def token_estimate(value: Any) -> int:
    """Conservative mixed CJK/Latin estimate, independent of a model tokenizer."""
    text = as_text(value)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk + math.ceil((len(text) - cjk) / 3)


def content_fields(item: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    metadata = item.get("metadata") or {}
    sources = [metadata.get("text") or {}, metadata, item]
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("visual_summary", "main_actions", "state_changes", "visual_observations", "visual_objects", "supporting_facts", "parent_context", "keyframe_captions", "keyframes", "source_micro_events"):
            if source.get(key):
                fields[key] = source[key]
    return fields


def action_window(action: dict[str, Any]) -> tuple[float | None, float | None]:
    start = action.get("start_time", action.get("start", action.get("timestamp")))
    end = action.get("end_time", action.get("end", start))
    window = action.get("time_span") or action.get("time_range") or action.get("time")
    if isinstance(window, (list, tuple)) and len(window) >= 2:
        start, end = window[:2]
    elif isinstance(window, dict):
        start = window.get("start_time", window.get("start"))
        end = window.get("end_time", window.get("end", start))
    elif isinstance(window, str):
        numbers = re.findall(r"\d+(?:\.\d+)?", window)
        if numbers:
            start = float(numbers[0])
            end = float(numbers[1]) if len(numbers) > 1 else start
    return event_time_seconds(start), event_time_seconds(end)


def action_records(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Native localized action/state records; no event-time substitution."""
    fields = content_fields(item)
    records = []
    sources = [fields] + [micro for micro in fields.get("source_micro_events", []) or [] if isinstance(micro, dict)]
    for source in sources:
        for key in ("main_actions", "state_changes", "visual_observations"):
            value = source.get(key)
            records.extend(row for row in (value if isinstance(value, list) else [value]) if isinstance(row, dict))
    return records


def visual_observation_records(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Read stored observations from the parent and its explicitly linked children."""
    fields = content_fields(item)
    sources = [fields] + [child for child in fields.get("source_micro_events", []) or [] if isinstance(child, dict)]
    return [observation for source in sources for observation in source.get("visual_observations", []) or []
            if isinstance(observation, dict)]


def image_observation_records(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Read LT image labels for selection without expanding the event prose."""
    from online_time import seconds_value, keyframe_path_seconds
    records = list(action_records(item))
    fields = content_fields(item)
    seen = {(row.get("path"), seconds_value(row.get("timestamp")), as_text(row.get("description")))
            for row in records if isinstance(row.get("path"), str)}
    for raw in fields.get("keyframe_captions", []) or []:
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        description = raw.get("caption") or raw.get("description") or raw.get("keyframe_caption")
        capture = seconds_value(raw.get("timestamp"))
        if capture is None:
            capture = keyframe_path_seconds(raw["path"])
        if capture is None or not isinstance(description, str) or not description.strip():
            continue
        identity = (raw["path"], capture, description.strip())
        if identity in seen:
            continue
        records.append({**raw, "timestamp": capture, "description": description.strip(),
                        "description_source": raw.get("description_source") or "stored_keyframe_caption"})
        seen.add(identity)
    return records


def visual_anchors(question: str, item: dict[str, Any], *, records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Only return time-grounded observations; never assign event time to a frame."""
    actions = action_records(item) if records is None else records
    anchors = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        start, end = action_window(action)
        if start is None:
            continue
        score = observation_match(question, action, item)
        if score > 0:
            anchors.append({"start": start, "end": end if end is not None else start, "score": score, "paths": action.get("keyframe_paths", []) or ([action["path"]] if action.get("path") else []), "observation": action})
    # Specific, strongly matched observations should outrank generic actions.
    return sorted(anchors, key=lambda row: -row["score"])


def _units(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+|\n+", value) if s.strip()]
    return [value] if value else []


def _compact_parent(question: str, parent: dict[str, Any], max_tokens: int = 200) -> dict[str, Any]:
    """Deduplicate a parent narrative without selecting away its chronology.

    Like child synopses, this background has a soft budget. Per-sentence query
    matching used to delete a transition already lost in child selection.
    """
    sentences = _units(parent.get("caption")) + _units(parent.get("visual_summary"))
    chosen = []
    seen = set()
    for index, sentence in enumerate(sentences):
        signature = re.sub(r"\W+", " ", sentence.lower()).strip()
        if signature in seen:
            continue
        seen.add(signature)
        chosen.append((index, sentence))
    return {key: value for key, value in {
        "doc_id": parent.get("doc_id"), "start_time": parent.get("start_time"), "end_time": parent.get("end_time"),
        "scope": parent.get("scope") or "coarse background; not localized to this child event",
        "caption": " ".join(sentence for _, sentence in sorted(chosen)),
    }.items() if value is not None}


def has_multiple_object_targets(question: str) -> bool:
    return len(query_focus(question) - {"clothing", "traffic"}) > 1


def supplemental_observation_matches(question: str, events: list[dict[str, Any]], constraints: dict[str, Any]) -> list[tuple[str, float]]:
    """One targeted text pass over timed observations, never a new dense index.

    First-occurrence and multi-object queries may lose their detail in coarse
    summaries. Returned hits are inspection candidates, never verified facts.
    """
    requirements = constraints.get("requirements") or {}
    multiple_targets = has_multiple_object_targets(question)
    if not (requirements.get("earliest") or requirements.get("simultaneous") or multiple_targets):
        return []
    requested = query_focus(question)
    hits = []
    for event in events:
        best = 0.0
        earliest = None
        parent_start, parent_end = event_time_seconds(event.get("start_time")), event_time_seconds(event.get("end_time"))
        parent_window = [parent_start, parent_end] if parent_start is not None and parent_end is not None else None
        for observation in visual_observation_records(event):
            start, end = action_window(observation)
            if start is None or not overlaps(start, end, constraint_window(constraints)):
                continue
            if not overlaps(start, end, parent_window):
                continue
            if actor_binding_fit(question, observation, event) == 0:
                continue
            signals = target_signals(question, observation)
            if requirements.get("simultaneous") and requested and not all(signals.get(name) in {"confirmed", "candidate"} for name in requested):
                continue
            if multiple_targets and not all(signals.get(name) in {"confirmed", "candidate"}
                                            for name in requested - {"clothing", "traffic"}):
                continue
            score = observation_match(question, observation, event)
            if score >= 0.08:
                best = max(best, score)
                earliest = start if earliest is None else min(start, earliest)
        doc_id = event.get("doc_id") or event.get("evidence_doc_id") or event.get("evidence_id")
        if doc_id and earliest is not None:
            hits.append((str(doc_id), best, earliest))
    hits.sort(key=(lambda row: (row[2], -row[1])) if requirements.get("earliest") else (lambda row: (-row[1], row[2])))
    return [(doc_id, score) for doc_id, score, _ in hits]


def supplemental_requirement_support(question: str, event: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """Describe a localized match without treating it as task completion.

    Text co-mentions only establish an inspection candidate. They do not prove
    that an actor performs the requested action on every named object at once.
    Preserve the original observation for the answer model to judge.
    """
    simultaneous = bool((constraints.get("requirements") or {}).get("simultaneous"))
    comparison = bool((constraints.get("requirements") or {}).get("comparison"))
    multiple_targets = has_multiple_object_targets(question)
    requested = sorted(query_focus(question) - ({"clothing", "traffic"} if multiple_targets else set()))
    parent_start, parent_end = event_time_seconds(event.get("start_time")), event_time_seconds(event.get("end_time"))
    parent_window = [parent_start, parent_end] if parent_start is not None and parent_end is not None else None
    anchors = _comparison_anchors(question, event, constraints) if comparison else visual_anchors(question, event)
    for anchor in anchors:
        if (anchor["score"] < 0.08 or
                not overlaps(anchor["start"], anchor["end"], constraint_window(constraints)) or
                not overlaps(anchor["start"], anchor["end"], parent_window)):
            continue
        observation = anchor["observation"]
        signals = target_signals(question, observation)
        co_mentioned = bool(requested) and all(signals.get(name) in {"confirmed", "candidate"} for name in requested)
        if not comparison and (simultaneous or multiple_targets) and not co_mentioned:
            continue
        return {"confirmed": False, "simultaneous": simultaneous,
                "support_level": "comparison_phase_candidate" if comparison and not co_mentioned else "text_candidate",
                "comparison_phase": comparison, "conditions_co_mentioned": co_mentioned,
                "observation_matched": True, "requested_conditions": requested,
                "signals": signals, "start": anchor["start"], "end": anchor["end"],
                "observation": observation, "match_score": anchor["score"]}
    return {"confirmed": False, "simultaneous": simultaneous,
            "observation_matched": False, "requested_conditions": requested, "signals": {}}

def _comparison_anchors(question: str, event: dict[str, Any], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    """A changed/absent starting condition is still evidence for the later side."""
    records = image_observation_records(event)
    anchors = visual_anchors(question, event, records=records)
    for observation in records:
        start, end = action_window(observation)
        if start is None or actor_binding_fit(question, observation, event) == 0:
            continue
        signals = inspection_signals(question, observation)
        if any(value in {"confirmed", "candidate", "negative"} for value in signals.values()):
            anchors.append({"start": start, "end": end if end is not None else start,
                            "score": max(0.08, relevance(question, observation)), "observation": observation})
    start, end = event_time_seconds(event.get("start_time")), event_time_seconds(event.get("end_time"))
    parent_window = [start, end] if start is not None and end is not None else None
    return [anchor for anchor in anchors if anchor["score"] >= 0.08
            and overlaps(anchor["start"], anchor["end"], constraint_window(constraints))
            and overlaps(anchor["start"], anchor["end"], parent_window)]


def comparison_observation_matches(question: str, events: list[dict[str, Any]], constraints: dict[str, Any]) -> list[tuple[str, float]]:
    """Recover timed action/state candidates on both sides of a comparison.

    Return event IDs only; exact observations keep their original wording and
    time. The caller must retain temporal diversity instead of assuming that
    the highest-scored single occurrence answers a before/after question.
    """
    if not (constraints.get("requirements") or {}).get("comparison"):
        return []
    rows = []
    for event in events:
        doc_id = event.get("doc_id") or event.get("evidence_doc_id") or event.get("evidence_id")
        if not doc_id:
            continue
        anchors = _comparison_anchors(question, event, constraints)
        if anchors:
            rows.append((str(doc_id), max(a["score"] for a in anchors), min(a["start"] for a in anchors)))
    rows.sort(key=lambda row: row[2])
    return [(doc_id, score) for doc_id, score, _ in rows]

def prompt_evidence(question: str, items: list[dict[str, Any]], max_tokens: int = 6000, *, image_paths: list[str] | None = None, constraints: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep selected event synopses intact; budget the supplemental details."""
    from online_evidence_coverage import coverage_features
    from online_query_constraints import resolve_query_constraints
    constraints = resolve_query_constraints(question, query_constraints=constraints)
    output = []
    candidates = []
    excluded = 0
    list_fields = {"main_actions", "state_changes", "visual_observations", "visual_objects", "supporting_facts", "parent_context"}
    summary = bool(constraints.get("summary"))
    seen_parents: set[str] = set()
    seen_observation_states: set[str] = set()
    sent = set(image_paths or [])
    window = constraint_window(constraints)
    synopsis_tokens = 0
    for i, item in enumerate(items):
        entry = {"evidence_id": item.get("evidence_id") or item.get("memory_id"), "source": item.get("source_memory") or "M_lt", "time": [item.get("start_time"), item.get("end_time")], "status": item.get("status") or "final"}
        output.append(entry)
        fields = {"caption": item.get("caption") or "", "transcript": item.get("transcript") or "", **content_fields(item)}
        # A selected event needs its own narrative, especially for action order.
        # Frame observations supplement that narrative instead of competing with
        # individual caption sentences and silently removing the event context.
        for key in ("caption", "visual_summary"):
            if fields.get(key):
                entry[key] = fields[key]
                synopsis_tokens += token_estimate({key: fields[key]})
        for key in ("main_actions", "state_changes", "visual_observations", "visual_objects", "transcript", "supporting_facts", "parent_context"):
            units = _units(fields.get(key))
            for j, unit in enumerate(units):
                if not unit:
                    continue
                if isinstance(unit, dict) and key in {"main_actions", "state_changes", "visual_observations"}:
                    start, end = action_window(unit)
                    if window and start is not None and not overlaps(start, end, window):
                        continue
                different_actor = key in {"visual_observations", "main_actions", "supporting_facts"} and actor_fit(question, unit) == 0
                if key == "parent_context":
                    if window:
                        continue
                    identity = str(unit.get("doc_id")) if isinstance(unit, dict) and unit.get("doc_id") else as_text(unit)
                    if identity in seen_parents:
                        continue
                    seen_parents.add(identity)
                    if summary and isinstance(unit, dict):
                        unit = _compact_parent(question, unit)
                        if not unit.get("caption"):
                            continue
                        entry.setdefault("parent_context", []).append(unit)
                        continue
                match = observation_match(question, unit, item) if key in {"main_actions", "state_changes", "visual_observations"} else relevance(question, unit)
                # Include complementary field types, not only repeated captions.
                weight = 1.0 + 8.0 * match + (0.8 if key in {"main_actions", "visual_observations", "state_changes"} else 0.0)
                if key == "visual_observations":
                    weight += 1.0
                    if isinstance(unit, dict) and unit.get("path") in sent:
                        weight += 8.0
                    binding = action_support(question, unit)
                    if binding and binding["score"] > 0:
                        weight += 0.8 * binding["score"]
                    if summary:
                        states = {feature for feature in coverage_features(question, {"visual_observations": [unit], "caption": fields.get("caption")})
                                  if not feature.startswith("object:")}
                        if states - seen_observation_states:
                            weight += 3.0
                        elif states and not (isinstance(unit, dict) and unit.get("path") in sent):
                            weight -= 1.5
                        seen_observation_states.update(states)
                if summary and key == "parent_context":
                    weight += 3.0
                if summary and j in {0, len(units)-1}:
                    weight += 0.6
                if different_actor:
                    # Keep scene context eligible without promoting another
                    # actor's observation as support for the requested person.
                    weight *= 0.25
                candidates.append((weight, i, key, j, unit))
    base_tokens = token_estimate(output)
    remaining = max(0, max_tokens - base_tokens)
    per_event = max(1, remaining // max(1, len(output)))
    used = [0] * len(output)
    chosen: dict[tuple[int, str], list[tuple[int, Any]]] = {}
    # After preserving event synopses, reserve one complete time/path observation
    # per event before repeated action lists consume the remaining budget.
    protected = []
    for i in range(len(output)):
        observations = [row for row in candidates if row[1] == i and row[2] == "visual_observations"]
        if observations:
            protected.append(max(observations, key=lambda row: row[0]))
    for row in sorted(protected, key=lambda row: -row[0]):
        weight, i, key, j, unit = row
        cost = token_estimate({key: [unit]}) + 2
        if cost <= remaining:
            chosen.setdefault((i, key), []).append((j, unit))
            used[i] += cost
            remaining -= cost
            candidates.remove(row)
    # First a fair share per event, then use the unspent global capacity.
    pending = sorted(candidates, key=lambda row: (-row[0], row[1], row[3]))
    for fair_pass in (True, False):
        deferred = []
        for weight, i, key, j, unit in pending:
            cost = token_estimate({key: unit}) + 2
            if cost > remaining or (fair_pass and used[i] + cost > per_event):
                deferred.append((weight, i, key, j, unit))
                continue
            chosen.setdefault((i, key), []).append((j, unit))
            used[i] += cost
            remaining -= cost
        pending = deferred
    for (i, key), units in chosen.items():
        units.sort(key=lambda row: row[0])
        output[i][key] = [unit for _, unit in units] if key in list_fields else " ".join(as_text(unit) for _, unit in units)
    excluded = len(pending)
    estimated_tokens = token_estimate(output)
    return output, {"token_budget": max_tokens, "estimated_tokens": estimated_tokens,
                    "base_tokens": base_tokens, "budget_exceeded": estimated_tokens > max_tokens,
                    "budget_shortfall_tokens": max(0, estimated_tokens - max_tokens),
                    "budget_scope": "evidence_text_estimate", "budget_policy": "soft_target_preserve_event_synopses",
                    "synopsis_tokens": synopsis_tokens,
                    "retained_event_captions": sum(bool(entry.get("caption")) for entry in output),
                    "retained_visual_summaries": sum(bool(entry.get("visual_summary")) for entry in output),
                    "selected_event_count": len(output), "omitted_units": excluded,
                    "retained_parent_summaries": sum(len(entry.get("parent_context", [])) for entry in output),
                    "retained_visual_observations": sum(len(entry.get("visual_observations", [])) for entry in output),
                    "policy": "event_synopses_then_whole_detail_units"}
