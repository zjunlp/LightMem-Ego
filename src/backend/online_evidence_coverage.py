"""Budgeted selection of observed changes, shared by retrieval and packing.

These helpers inspect saved text/metadata only. They neither open images nor
infer continuous states between sampled observations.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from online_evidence_content import (as_text, content_fields, focus_match, query_focus, query_terms, relevance,
                                   target_signals, actor_binding_fit, action_window, _observation_polarity)
from online_query_constraints import actor_fit
from online_time import event_time_seconds


TOPICS = {
    "glasses": r"sunglasses|(?<!sun)glasses|墨镜|眼镜|太阳镜",
    "hat": r"\bhat\b|\bvisor\b|\bhood\b|帽",
    "coat": r"\bjacket\b|\bcoat\b|\bpuffer\b|外套|羽绒服|夹克",
    "clothing": r"\bshirt\b|\bblouse\b|\btop\b|上衣|衬衫|内搭",
    "phone": r"\bphone\b|smartphone|手机|电话",
    "bag": r"\bbag\b|backpack|rucksack|背包|灰包|包[，。\s]",
    "traffic": r"\btraffic\b|\bcars?\b|\bvehicles?\b|\bqueue\b|\broad\b|\bsuv\b|\bmpv\b|\bhatchback\b|\bsedan\b|\bvan\b|\btruck\b|\bbus(?:es)?\b|车流|车辆|轿车|厢式车|货车|排队|拥堵|道路",
    "mask": r"\bmask\b|口罩",
    "bottle": r"\bbottle\b|水瓶|瓶子",
}
_ACTIONS = {
    "put_on": r"put(?:s|ting)? on|wearing|戴上|穿上",
    "remove": r"remov|takes? off|摘下|脱下",
    "raise": r"rais|lift|拿起|举起|提起",
    "lower": r"lower|puts? .{0,20}down|放下|放低",
    "transfer": r"hand(?:s|ed|ing)? .{0,25}to|transfer|递给|递向|交给",
    "open": r"\bopen|unzip|打开|拉开",
    "close": r"\bclos|\bzip|关上|拉上",
}


def evidence_time(item: dict[str, Any]) -> float:
    for key in ("timestamp", "start_time", "start"):
        raw = item.get(key)
        day = re.fullmatch(r"DAY(\d+)\s+(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)", str(raw or ""), re.I)
        value = event_time_seconds(day[2] if day else raw)
        if value is not None:
            return value + ((int(day[1])-1)*86400 if day else 0)
    return 0.0


def _states(topic: str, text: str) -> set[str]:
    states = set()
    if topic in {"glasses", "hat", "mask", "coat"}:
        if topic == "glasses" and re.search(r"top of (?:her |his |the )?head|on (?:her |his |the )?head|头顶", text):
            states.add("head")
        target = {"glasses": r"(?:sunglasses|glasses|墨镜|眼镜|太阳镜)", "hat": r"(?:hat|visor|hood|帽)",
                  "mask": r"(?:mask|口罩)", "coat": r"(?:coat|jacket|puffer|外套|羽绒服|夹克)"}[topic]
        worn = bool(re.search(r"(?:wear\w*|put\w* on|戴着|戴上|穿上|穿着).{0,35}" + target, text))
        if topic == "glasses":
            worn = bool(re.search(r"over (?:her |his |the )?eyes|cover.{0,12}eyes|眼前|脸上", text)) or (worn and "head" not in states)
            if re.search(r"no longer|not (?:on|over|wear)|without|未戴|没有戴", text):
                worn = False
        if worn:
            states.add("worn")
        negative = re.search(r"(?:without|not wearing|no longer wearing|未戴|没有戴).{0,18}" + target, text)
        removed = re.search(r"(?:remov\w*|takes? off|摘下|脱下).{0,20}" + target, text)
        removed = removed or re.search(target + r".{0,20}(?:removed|no longer (?:on|over)|not (?:on|over)|不在.*(?:头|脸))", text)
        if negative or removed:
            states.discard("worn")
            states.add("removed")
        if topic == "coat" and re.search(r"(?:jacket|coat).{0,10}(?:is on|over her shoulders|over his shoulders)|外套.{0,6}穿好", text):
            states.add("worn")
    if topic == "phone":
        if re.search(r"\bear\b|贴耳|耳边", text):
            states.add("ear")
        if re.search(r"lower|down|放下|放低", text):
            states.add("lowered")
    if topic == "traffic":
        if re.search(r"queue|congest|closely|dense|排队|拥堵|密集|近距离", text):
            states.add("dense")
        if re.search(r"close ahead|close (?:on|in front)|nearby vehicles|靠近|紧邻", text):
            states.add("near")
        if re.search(r"stopped|slow.moving|unchanged positions|排队|停着|缓慢", text):
            states.add("stopped_or_slow")
        if re.search(r"sparse|far(?:ther)? (?:ahead|away|in front)|distant|in the distance|open road|road (?:is |remains |looks )?(?:relatively )?(?:open|clear)\b|空旷|稀疏|开阔|远处", text):
            states.add("sparse")
    return states


def coverage_features(question: str, item: dict[str, Any]) -> set[str]:
    fields = content_fields(item)
    focus = query_focus(question)
    units = []
    for key in ("state_changes", "main_actions", "visual_observations"):
        units.extend(fields.get(key) or [])
    if not units:
        units = [fields.get("visual_summary") or item.get("caption") or item.get("text") or ""]
    features = set()
    context = as_text(fields.get("visual_summary") or item.get("caption") or item.get("text") or "").lower()
    context_coat = bool(re.search(r"wear.{0,80}(?:jacket|coat)|in a .{0,30}(?:jacket|coat)|穿.{0,10}(?:外套|羽绒服)", context))
    for unit in units:
        if actor_fit(question, unit) == 0:
            continue
        fit = focus_match(question, unit)
        if fit == 0:
            continue
        text = (str(unit.get("description") or unit.get("action") or unit.get("change") or as_text(unit))
                if isinstance(unit, dict) else str(unit)).lower()
        entity = str(unit.get("entity") or "").lower() if isinstance(unit, dict) else ""
        # Native before/after records remain useful outside the concept
        # vocabulary. This is a bounded diversity hint, not inferred vision.
        if isinstance(unit, dict) and entity.strip() and not any(re.search(pattern, entity) for pattern in TOPICS.values()):
            before, after = unit.get("before"), unit.get("after")
            if before is not None and after is not None:
                native_text = as_text([unit.get("description") or "", before, after]).lower()
                # A structured field can itself say "possibly open". Be
                # conservative about adding diversity credit; the original
                # uncertain evidence is still preserved for the answer model.
                uncertain_text = bool(re.search(
                    r"\b(?:may|might|could|possibly|possible|perhaps|unclear|uncertain|unknown|unconfirmed)\b|不能确认|无法确认|不确定|可能|疑似|看不清", native_text))
                before = re.sub(r"\W+", " ", as_text(before).lower()).strip()
                after = re.sub(r"\W+", " ", as_text(after).lower()).strip()
                start, _ = action_window(unit)
                negative, uncertain = _observation_polarity(entity + " " + after, unit,
                    str(unit.get("description") or ""), after)
                if (before and after and before != after and start is not None
                        and not negative and not uncertain and not uncertain_text
                        and actor_binding_fit(question, unit, item) > 0):
                    actor = re.sub(r"\s+", " ", as_text(unit.get("actor") or "").lower()).strip()
                    if not actor:
                        # Missing identity is not shared identity. Use a bounded
                        # observation reference, without inventing an actor.
                        reference = as_text([start, unit.get("path"), unit.get("description")])
                        actor = "unbound-" + hashlib.sha256(reference.encode()).hexdigest()[:16]
                    features.add(f"transition:generic:{entity.strip()[:100]}:actor={actor[:80]}:{before[:80]}->{after[:80]}")
        # A before/after record's current state is its after value. Do not
        # misread a no-longer-visible bottle as a removed hat in the same row.
        state_text = (entity + " " + str(unit.get("after"))).lower() if isinstance(unit, dict) and unit.get("after") else text
        for topic, pattern in TOPICS.items():
            if not re.search(pattern, text + " " + entity):
                continue
            if focus and topic not in focus and not ("clothing" in focus and topic in {"clothing", "glasses", "hat", "coat", "mask"}):
                continue
            signal = target_signals(question, unit).get(topic)
            if signal == "candidate":
                features.add("candidate:" + topic)
                continue
            if signal == "negative" and topic in {"phone", "bag", "bottle"}:
                features.add("state:" + topic + ":not_visible")
                continue
            features.add("object:" + topic)
            clauses = re.split(r"[.;!?。！？；]|,\s*(?:and|then|but)|\band\b", state_text)
            relevant = " ".join(clause for clause in clauses if re.search(pattern, clause))
            states = _states(topic, relevant)
            for state in states:
                features.add(f"state:{topic}:{state}")
                if "clothing" in focus and topic == "glasses":
                    features.add(f"appearance:glasses:{state}:coat={context_coat}")
            for action, regex in (() if topic == "traffic" else _ACTIONS.items()):
                if re.search(regex, relevant):
                    features.add(f"action:{topic}:{action}")
            if isinstance(unit, dict) and unit.get("before") and unit.get("after") and re.search(pattern, entity):
                before = re.sub(r"\W+", " ", str(unit["before"]).lower()).strip()
                after = re.sub(r"\W+", " ", str(unit["after"]).lower()).strip()
                if before != after:
                    # Structured transitions retain their supplied values;
                    # object/state features merge synonymous prose elsewhere.
                    features.add(f"transition:{topic}:{before[:80]}->{after[:80]}")
    return features


# These are generic narration words.  They are deliberately independent of
# the object catalog above: the event may contain an object that has never
# appeared in a hand-written topic list.
_DETAIL_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "with", "for", "from",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "does", "did",
    "do", "can", "could", "will", "would", "should", "may", "might", "then", "still", "also",
    "person", "people", "man", "woman", "someone", "right", "left", "front", "rear", "side",
    "inside", "outside", "scene", "view", "visible", "appears", "looks", "looking", "shown",
    "there", "here", "near", "around", "toward", "towards", "camera", "vehicle", "car", "cars",
    "second", "seconds", "time", "moment", "frame", "event", "image", "picture", "video",
    "他", "她", "其", "人", "男子", "女子", "前方", "后方", "左侧", "右侧", "画面", "场景",
    "可以", "可能", "似乎", "正在", "然后", "之后", "仍然", "依然", "一个", "一位", "东西",
}


def _detail_text(observation: Any) -> str:
    if not isinstance(observation, dict):
        return as_text(observation)
    parts: list[str] = []
    # Keep the supplied fields separate from the JSON key names.  The latter
    # are schema labels, not observed details.
    for key in ("description", "action", "change", "before", "after", "from_state", "to_state", "attribute", "entity"):
        value = observation.get(key)
        if value is not None and as_text(value).strip():
            parts.append(as_text(value))
    return " ".join(parts)


def _query_relative_detail_text(question: str, text: str) -> str:
    """Keep the local clauses around query terms in free-form captions."""
    terms = {term for term in query_terms(question, expand=True)
             if term not in _DETAIL_STOP and (not term.isascii() or len(term) > 1)}
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for term in terms:
        if term.isascii():
            spans.extend(match.span() for match in re.finditer(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", lowered))
        else:
            start = 0
            while True:
                start = lowered.find(term, start)
                if start < 0:
                    break
                spans.append((start, start + len(term)))
                start += max(1, len(term) // 2)
    if not spans:
        return text
    chunks = []
    for start, end in spans:
        left, right = max(0, start - 36), min(len(text), end + 36)
        while left > 0 and text[left - 1].isalnum():
            left -= 1
        while right < len(text) and text[right].isalnum():
            right += 1
        chunks.append(text[left:right])
    return " ".join(dict.fromkeys(chunks))


def observation_detail_features(question: str, observation: Any) -> set[str]:
    """Return generic, query-relative details from one stored observation.

    This is a diversity signal for evidence packing, not a visual classifier.
    Query terms are removed so that the target object itself is shared by all
    candidate frames; remaining state/action words can distinguish, for
    example, two hands from one hand or worn from removed.
    """
    text = _detail_text(observation)
    if not text.strip():
        return set()
    query_tokens = query_terms(question, expand=True)
    detail_text = _query_relative_detail_text(question, text)
    tokens = query_terms(detail_text, expand=False)
    features: set[str] = set()
    for token in tokens - query_tokens:
        normalized = str(token).strip().lower()
        if not normalized or normalized in _DETAIL_STOP:
            continue
        # Keep short CJK n-grams because they are often the only signal for a
        # supplied state such as 单手/双手 or 头顶.  Latin one-character
        # fragments are never useful here.
        if normalized.isascii() and len(normalized) < 2:
            continue
        features.add("detail:" + normalized)
    lowered = detail_text.lower()
    if re.search(r"\b(?:without|no longer|not wearing|removed|absent)\b|没有|未戴|摘下|脱下|取下|不在", lowered):
        features.add("detail:polarity=negative")
    if re.search(r"\b(?:possibly|possible|unclear|uncertain|unknown|may|might)\b|可能|疑似|不确定|看不清", lowered):
        features.add("detail:polarity=uncertain")
    if re.search(r"\b(?:now|begin\w*|start\w*|rais\w*|lower\w*|turn\w*|chang\w*|becom\w*|mov\w*|switch\w*|shift\w*)\b|现在|开始|抬起|放下|转向|改变|变成|改为|转为|随后", lowered):
        features.add("detail:change_marker")
    if re.search(r"\b(?:remains?|continues?|still|keeps?|same|unchanged|similar)\b|仍然|继续|依旧|同样|未变|保持", lowered):
        features.add("detail:continuation_marker")
    if isinstance(observation, dict):
        before = as_text(observation.get("before") or observation.get("from_state") or "").strip().lower()
        after = as_text(observation.get("after") or observation.get("to_state") or "").strip().lower()
        if before and after and before != after:
            before = re.sub(r"\s+", " ", before)[:100]
            after = re.sub(r"\s+", " ", after)[:100]
            features.add(f"detail:transition={before}->{after}")
    return features


def _detail_feature_weight(feature: str) -> float:
    if feature.startswith("detail:transition="):
        return 3.0
    if feature.startswith("detail:polarity="):
        return 2.0
    if feature == "detail:change_marker":
        return 2.5
    if feature == "detail:continuation_marker":
        return 0.1
    return 1.0


def order_frame_candidates_by_complementarity(question: str, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order an already-ranked event bundle by complementary observations.

    The first frame keeps the caller's existing priority.  Each following
    frame is chosen by novel stored details, with the previous local score and
    time spread as small tie-breakers.  The function never drops frames or
    changes their event/source/time metadata.
    """
    if len(frames) <= 1:
        for index, frame in enumerate(frames):
            frame["_complementarity_rank"] = index
            frame["_complementarity_enabled"] = False
        return list(frames)
    # Native action records already have a separate repeated-occurrence
    # boundary policy in the packer.  Do not let the free-text diversity pass
    # reorder those records and hide that established boundary.  State
    # transitions (before/after) remain eligible because their supplied
    # transition itself is the complementary detail.
    observations = [observation for frame in frames
                    for observation in (frame.get("_coverage_observations") or frame.get("_direct_observations") or [])
                    if isinstance(observation, dict)]
    if any(observation.get("action") or observation.get("predicate") or observation.get("triple")
           for observation in observations):
        for index, frame in enumerate(frames):
            frame["_complementarity_rank"] = index
            frame["_complementarity_enabled"] = False
        return list(frames)
    feature_sets: list[set[str]] = []
    for frame in frames:
        observations = frame.get("_coverage_observations") or frame.get("_direct_observations") or []
        if not isinstance(observations, list):
            observations = [observations]
        features = set()
        for observation in observations:
            features.update(observation_detail_features(question, observation))
        feature_sets.append(features)
    # There is no basis for a content-aware reorder when records are absent;
    # retaining the existing order is safer than inventing diversity.
    if not any(feature_sets):
        for index, frame in enumerate(frames):
            frame["_complementarity_rank"] = index
            frame["_complementarity_enabled"] = False
        return list(frames)
    selected_indices = [0]
    covered = set(feature_sets[0])
    while len(selected_indices) < len(frames):
        def utility(index: int) -> tuple[float, float, float, int]:
            novel = feature_sets[index] - covered
            novelty_score = sum(_detail_feature_weight(feature) for feature in novel)
            local = float(frames[index].get("_local_relevance") or 0.0)
            action = float(frames[index].get("_action_support_level") or 0.0)
            timestamp = frames[index].get("timestamp")
            if timestamp is None:
                spread = 0.0
            else:
                spread = min((abs(float(timestamp) - float(frames[picked].get("timestamp")))
                              for picked in selected_indices if frames[picked].get("timestamp") is not None), default=0.0)
            # Novel detail outranks the old order.  Once detail novelty is
            # equal, retain the caller's existing chronology/rank before using
            # time spread as a final tie-breaker; repeated action records then
            # keep their established occurrence boundary.
            return (novelty_score, local, action, -index, 0.01 * spread)
        remaining = [index for index in range(len(frames)) if index not in selected_indices]
        best = max(remaining, key=utility)
        selected_indices.append(best)
        covered.update(feature_sets[best])
    ordered = [frames[index] for index in selected_indices]
    for rank, frame in enumerate(ordered):
        frame["_complementarity_rank"] = rank
        frame["_complementarity_enabled"] = True
    return ordered


def _feature_weight(feature: str) -> float:
    if feature.startswith("first:state:"):
        return 5.0
    if feature.startswith("first:appearance:"):
        return 4.0
    if feature.startswith(("state:", "appearance:")):
        return 5.0
    if feature.startswith("transition:"):
        return 0.5
    if feature.startswith("action:"):
        return 0.5
    return 0.2


def select_coverage(question: str, items: list[dict[str, Any]], limit: int, *, preserve_start: bool = True,
                    preserve_end: bool = True, preserve_middle: bool = False) -> list[dict[str, Any]]:
    """Prefer complementary evidence, using temporal spread as a tie-breaker."""
    if limit <= 0 or not items:
        return []
    ordered = sorted(items, key=evidence_time)
    if len(ordered) <= limit:
        return ordered
    features = [coverage_features(question, item) for item in ordered]
    # A later rich event must not substitute for the first observed instance
    # of a state. This also prevents an apparent first occurrence drifting as
    # the history grows. These are first observations in this candidate set.
    previously_seen = set()
    for feature_set in features:
        first = {feature for feature in feature_set - previously_seen if feature.startswith(("state:", "appearance:"))}
        previously_seen.update(feature_set)
        feature_set.update("first:" + feature for feature in first)
    selected = [0] if preserve_start else []
    if preserve_end and limit > len(selected) and len(ordered)-1 not in selected:
        selected.append(len(ordered)-1)
    if preserve_middle and limit >= 4 and len(selected) < limit:
        middle = (evidence_time(ordered[0]) + evidence_time(ordered[-1])) / 2
        candidates = [i for i in range(len(ordered)) if i not in selected]
        if candidates:
            selected.append(min(candidates, key=lambda i: abs(evidence_time(ordered[i])-middle)))
    if preserve_start and len(selected) < limit:
        first_change = next((i for i in range(1, len(ordered)) if any(feature.startswith("first:state:") for feature in features[i])), None)
        if first_change is not None and first_change not in selected:
            selected.append(first_change)
    covered = set().union(*(features[i] for i in selected)) if selected else set()
    span = max(1.0, evidence_time(ordered[-1]) - evidence_time(ordered[0]))
    while len(selected) < min(limit, len(ordered)):
        def utility(i):
            novel = sum(_feature_weight(feature) for feature in features[i] - covered)
            fit = focus_match(question, as_text(content_fields(ordered[i])))
            gap = min((abs(evidence_time(ordered[i])-evidence_time(ordered[j])) for j in selected), default=span)/span
            rank = float(ordered[i].get("final_score") or ordered[i].get("score") or ordered[i].get("episodic_score") or 0)
            return novel + 0.4 * (fit or 0) + 0.2 * relevance(question, content_fields(ordered[i])) + 0.15 * gap + 0.05 * rank
        best = max((i for i in range(len(ordered)) if i not in selected), key=utility)
        selected.append(best)
        covered.update(features[best])
    return [ordered[i] for i in sorted(selected)]


def observation_priority(question: str, observation: dict[str, Any], covered: set[str]) -> tuple[float, set[str]]:
    features = coverage_features(question, {"visual_observations": [observation]})
    novelty = sum(_feature_weight(feature) for feature in features - covered)
    return novelty + relevance(question, observation), features


GROUNDING_INSTRUCTION = (
    "Describe visible observations separately from inferred intent: a phone held at an ear does not prove "
    "a connected call, and an unclear or occluded object does not prove removal or absence. "
    "Attach a specific time/evidence reference to each major claim; never attach an observation to a different image time. "
    "When an action has multiple observed occurrences, report distinct periods and their different states. "
    "For before/after comparisons check whether occurrences exist on BOTH sides; do not turn one example into a universal conclusion. "
    "For summaries preserve distinct supported state changes in chronological order, including repeated changes; "
    "do not replace the sequence with only its first and last state. Sparse samples do not establish continuous duration."
)
