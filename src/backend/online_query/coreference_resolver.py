from __future__ import annotations

import os
import re
from typing import Any

from online_query.cache_schema import CacheContext, has_valid_binding
from online_query.interaction_cache import InteractionCache
from online_query_constraints import resolve_query_constraints
from online_time import seconds_value


_EVENT_TIME = r"的时候|之时|那一刻|那次|(?<!当)(?<!那)(?<!此)时(?=[，,。？?]|有|他|她|它)"


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    for keyword in keywords:
        needle = str(keyword or "").strip().lower()
        if not needle:
            continue
        if needle.isascii():
            prefix = r"(?<![a-z0-9])" if needle[0].isalnum() else ""
            suffix = r"(?![a-z0-9])" if needle[-1].isalnum() else ""
            if re.search(prefix + re.escape(needle) + suffix, lower):
                return True
        elif needle in lower:
            return True
    return False


class CoreferenceResolver:
    """Lightweight rule-based follow-up/coreference resolver."""

    PRONOUN_KWS = [
        "他", "她", "它", "这个", "那个", "这些", "那些", "这个东西", "那个人", "那个物体", "刚才那个",
        "he", "she", "it", "they", "this", "that", "those", "the object", "the person", "the thing",
        "that object", "this object", "that person", "this person", "the previous one",
    ]
    FOLLOWUP_KWS = [
        "刚才", "刚刚", "上一段", "那个时候", "当时", "那时", "后来呢", "之后呢", "以后呢", "然后呢", "接下来呢", "后来", "之后", "以后",
        "then", "later", "after that", "what about it", "what about that", "where did it go",
        "just now", "recently", "a moment ago", "a few seconds ago", "moments ago",
        "previous moment", "previously", "last segment", "last scene", "what just happened",
    ]
    LATER_KWS = ["后来", "之后", "以后", "然后", "接下来", "later", "after that", "then", "next", "then what"]
    SAME_TIME_KWS = [
        "那个时候", "当时", "那时", "刚才那段", "刚才的那段", "刚刚那段",
        "刚才的画面", "刚刚的画面", "上一段画面", "上一段视频",
        "at that time", "at the time", "during that", "previous segment", "previous scene",
    ]

    def __init__(self, followup_window_seconds: int | None = None) -> None:
        self.followup_window_seconds = followup_window_seconds or int(os.getenv("EM2MEM_CACHE_FOLLOWUP_WINDOW_SECONDS", "60"))

    def resolve(self, question: str, interaction_cache: InteractionCache, latest_seconds: float | None = None,
                *, query_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        q = (question or "").strip()
        context = interaction_cache.latest_context()
        latest = context.get("latest_interaction") or {}
        hot_entities = list(context.get("hot_entities", []) or [])
        hot_ranges = list(context.get("hot_time_ranges", []) or [])
        hot_memories = list(context.get("hot_memories", []) or [])
        constraints = resolve_query_constraints(q, query_constraints=query_constraints, latest=latest_seconds)
        if constraints.get("time_explicit") or (constraints.get("scope") == "current" and not constraints.get("requires_history")):
            return CacheContext(reason="explicit question time takes precedence over cached context").to_dict()

        if not latest and not hot_entities and not hot_ranges:
            return CacheContext(reason="interaction cache is empty").to_dict()

        # A pronoun can name the same actor without referring to the last
        # answer's time. Self-contained historical scope must stay searchable.
        historical_scope = bool(re.search(r"刚开始|最开始|一开始|最初|起初|开头|早先|较早|最早|首次|第一次|前面|\bearliest\b|\binitially\b|\bat the beginning\b|\bfirst time\b", q, re.I))
        dialogue_reference = bool(re.search(r"你(?:刚才|之前|前面)?(?:说|提)|(?:前面|刚才|之前|上次)(?:说|聊|谈|讨论|提)|上[一次]?(?:答|问)|上一段|刚才那段|那个时候|那时候|\byou (?:said|mentioned)\b|\bprevious answer\b", q, re.I))
        has_pronoun = _contains_any(q, self.PRONOUN_KWS)
        if historical_scope and not dialogue_reference and not constraints.get("summary"):
            return self._identity_context(q, latest, has_pronoun)

        has_followup = _contains_any(q, self.FOLLOWUP_KWS + self.SAME_TIME_KWS)
        image_reference = bool(re.search(r"第一.*图|最早.*图|最晚.*图|最后.*图|上一.*图|first.*(?:image|frame)|earliest.*(?:image|frame)|latest.*(?:image|frame)|last.*(?:image|frame)", q, re.I))
        if image_reference:
            frames = latest.get("displayed_evidence_frames") or latest.get("evidence_frames") or []
            frames = [frame for frame in frames if frame.get("timestamp") is not None]
            if frames:
                frame = (min(frames, key=lambda row: float(row["timestamp"])) if re.search(r"最早|earliest", q, re.I)
                         else max(frames, key=lambda row: float(row["timestamp"])) if re.search(r"最晚|latest", q, re.I)
                         else frames[-1] if re.search(r"最后|last", q, re.I) else frames[0])
                resolved = CacheContext(cache_hit=True, is_followup=True, binding_valid=True, confidence=0.95,
                    referenced_time_ranges=[{"start": frame["timestamp"], "end": frame["timestamp"], "source": "explicit_previous_image"}],
                    referenced_evidence_frames=[frame], reason="explicit reference to an actually displayed image").to_dict()
                resolved["temporal_relation"] = "same_time"
                return resolved
        if has_pronoun and not has_followup:
            video_summary_reference = _contains_any(q, ["这个视频", "这段视频", "this video", "the video"])
            specific_reference = _contains_any(q, ["这个东西", "那个东西", "那个人", "那个物体", "刚才那个", "the object", "the person", "the thing"])
            if video_summary_reference and not specific_reference:
                has_pronoun = False
        # Interrogative "When did he ...?" searches for an occurrence. In
        # "When he lifted it, what ...?", the when-clause names an event.
        # Neither a shared pronoun nor a shared object supplies a new time.
        seeks_occurrence = bool(re.search(
            r"什么时候|何时|几秒|几点|\bwhen\s+(?:did|does|do|was|were|is|are|has|have|had|will|would|can|could)\b|\bat what (?:time|point)\b",
            q, re.I)) or bool(constraints.get("requires_history") and re.search(
                r"有没有|有没|是否曾|曾经|\bever\b|\b(?:has|have) there been\b", q, re.I))
        names_previous_time = bool(_contains_any(q, self.SAME_TIME_KWS) or re.search(_EVENT_TIME, q, re.I))
        if seeks_occurrence and not names_previous_time and not image_reference:
            return self._identity_context(q, latest, has_pronoun)
        is_followup = bool(has_pronoun or has_followup)
        if has_pronoun and not has_followup and re.search(r"画面[左右]|右侧|左侧|前排|后排|整个视频|全程|from the beginning|entire video", q, re.I):
            is_followup = False
        if not is_followup:
            return CacheContext(
                cache_hit=True,
                reason="cache exists but question has no follow-up signal",
            ).to_dict()
        # A summary spans a sequence, not a single cached answer. Keep the
        # conversation's ranges as soft diagnostics rather than rewriting the
        # user's endpoint ("to now") into one old point.
        summary_to_now = bool(re.search(r"到现在|至今|到目前|to now|until now|up to now", q, re.I))
        if constraints.get("summary") and (summary_to_now or (constraints.get("requirements") or {}).get("comparison") or not (
                _contains_any(q, self.SAME_TIME_KWS) or re.search(r"\bwhen\b|" + _EVENT_TIME, q, re.I))):
            resolved = CacheContext(cache_hit=True, is_followup=True, resolved_question=q,
                reason="summary preserves original scope instead of binding one cached event").to_dict()
            resolved["conversation_time_hints"] = [
                {"turn_id": turn.get("turn_id"), "time_ranges": turn.get("time_ranges", [])}
                for turn in context.get("recent_interactions", []) if turn.get("time_ranges")]
            resolved["temporal_relation"] = "related"
            return resolved
        # Generic object references inherit the primary topic of the actual
        # previous answer. An unrelated noun in its background is not a focus.
        object_followup = bool(re.search(r"你说的|那个东西|这个东西|刚才那个|它|\bit\b|that (?:thing|object)|this (?:thing|object)|you (?:said|mentioned)", q, re.I))
        focus_entities = list(latest.get("focus_entities", []) or []) if object_followup else []
        if len(focus_entities) > 1:
            resolved = CacheContext(cache_hit=True, is_followup=True, confidence=0.4,
                referenced_entities=focus_entities,
                reason="previous answer has multiple primary objects; reference is ambiguous").to_dict()
            resolved.update(reference_ambiguous=True, temporal_relation="related")
            return resolved
        explicit_event_reference = bool(re.search(r"\bwhen\b|" + _EVENT_TIME, q, re.I))
        scene_reference = bool(_contains_any(q, self.SAME_TIME_KWS)) and not explicit_event_reference
        generic_after = bool(re.fullmatch(
            r"(?:那|然后|后来|之后|以后|接下来|呢|[？?。\s])+|(?:what happened )?after that[?.\s]*|later[?.\s]*",
            q, re.I))
        if (focus_entities and not explicit_event_reference) or scene_reference or generic_after:
            ambiguous = False
        else:
            # Binding an object identifies the topic; a newly described action
            # still needs an observed compatible occurrence in the dialogue.
            latest, ambiguous = self._match_turn(q, context, latest, focus_entities)
        if ambiguous:
            resolved = CacheContext(cache_hit=True, is_followup=True, confidence=0.4,
                reason="event reference does not uniquely match a compatible cached turn; no hard time anchor").to_dict()
            resolved["reference_ambiguous"] = True
            resolved["temporal_relation"] = "related"
            return resolved
        if latest.get("anchor_status") == "unresolved":
            resolved = CacheContext(cache_hit=True, is_followup=True, confidence=0.5,
                reason="previous answer did not establish an event time; do not reuse an older hot time range").to_dict()
            resolved["anchor_status"] = "unresolved"
            return resolved

        referenced_entities = (focus_entities or list(latest.get("entities", []) or [])[:5]) if latest else hot_entities[:5]
        # Each range can be a different occurrence with a different state.
        # Keep the complete bounded turn; image/text budgets are applied later.
        referenced_ranges = list(latest.get("time_ranges", []) or []) if latest else hot_ranges
        referenced_memory_ids = list(latest.get("retrieved_memory_ids", []) or [])
        referenced_segment_ids = list(latest.get("retrieved_segment_ids", []) or [])
        referenced_visual_ids = list(latest.get("visual_ids", []) or [])
        referenced_evidence_frames = list(latest.get("evidence_frames", []) or [])
        if focus_entities:
            references = latest.get("focus_evidence_refs", []) or []
            if references:
                referenced_memory_ids = [str(ref["evidence_id"]) for ref in references if ref.get("evidence_id")]
                referenced_segment_ids = [str(ref["evidence_id"]) for ref in references
                                          if ref.get("source_memory") == "M_lt" and ref.get("evidence_id")]
                referenced_visual_ids = []
                scopes = [(seconds_value(ref.get("start_time")), seconds_value(ref.get("end_time"))) for ref in references]
                scopes = [(start, end) for start, end in scopes if start is not None and end is not None]
                narrowed = []
                for window in referenced_ranges:
                    start, end = seconds_value(window.get("start")), seconds_value(window.get("end"))
                    if start is None or end is None:
                        continue
                    intersections = sorted((max(start, lo), min(end, hi)) for lo, hi in scopes
                                           if max(start, lo) <= min(end, hi))
                    merged = []
                    for lo, hi in intersections:
                        if merged and lo <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], hi)
                        else:
                            merged.append([lo, hi])
                    for lo, hi in merged:
                        item = {**window, "start": lo, "end": hi}
                        if item not in narrowed:
                            narrowed.append(item)
                # Keep the actual cited periods separate. The focus's coarse
                # event range cannot enlarge them or borrow another object.
                referenced_ranges = narrowed
                paths = {path for ref in references for path in ref.get("keyframe_paths", []) or []}
                referenced_evidence_frames = [frame for frame in referenced_evidence_frames
                    if (not paths or (frame.get("path") or frame.get("image_path")) in paths)
                    and seconds_value(frame.get("timestamp")) is not None
                    and any(window["start"] <= seconds_value(frame["timestamp"]) <= window["end"] for window in narrowed)]

        entity_names = [
            str(item.get("canonical_name") or item.get("name") or item.get("entity_key"))
            for item in referenced_entities
            if item.get("canonical_name") or item.get("name") or item.get("entity_key")
        ]
        range_text = self._format_ranges(referenced_ranges)
        resolved_question = q
        reason = "question contains follow-up/coreference signals and cache has recent context"
        confidence = 0.62

        if _contains_any(q, self.LATER_KWS) and referenced_ranges:
            expanded = self._expand_after_ranges(referenced_ranges, latest_seconds)
            referenced_ranges = expanded
            resolved_question = (
                f"After the previously retrieved event around {range_text}"
                f"{self._entity_clause(entity_names)}, answer: {q}"
            )
            confidence = 0.74
            # Previous pictures describe the anchor, not the later event.
            referenced_evidence_frames = []
        elif entity_names or referenced_ranges:
            resolved_question = (
                f"Using the previous context"
                f"{self._entity_clause(entity_names)}"
                f"{self._time_clause(range_text)}, answer: {q}"
            )
            confidence = 0.7

        resolved = CacheContext(
            cache_hit=True,
            is_followup=True,
            resolved_question=resolved_question,
            referenced_entities=referenced_entities,
            referenced_time_ranges=referenced_ranges,
            referenced_memory_ids=referenced_memory_ids,
            referenced_segment_ids=referenced_segment_ids,
            referenced_visual_ids=referenced_visual_ids,
            referenced_evidence_frames=referenced_evidence_frames,
            confidence=confidence,
            reason=reason,
        ).to_dict()
        resolved["binding_valid"] = has_valid_binding({
            **resolved, "binding_valid": True, "anchor_status": latest.get("anchor_status")})
        resolved["temporal_relation"] = (
            "after" if _contains_any(q, self.LATER_KWS)
            else "same_time" if _contains_any(q, self.SAME_TIME_KWS)
            else "related"
        )
        resolved["matched_turn_id"] = latest.get("turn_id")
        if focus_entities:
            resolved["bound_focus_entities"] = focus_entities
        if constraints.get("requires_current") or (constraints.get("requirements") or {}).get("comparison"):
            resolved["temporal_relation"] = "related"
        if len(referenced_ranges) > 1:
            resolved["occurrence_instruction"] = (
                "The previous answer refers to multiple observed periods. Answer each period separately "
                "when their states differ; do not collapse them to one color, state, or temporal relation."
            )
        return resolved

    def _identity_context(self, question: str, latest: dict[str, Any], has_pronoun: bool) -> dict[str, Any]:
        """Carry a known subject into a fresh search, without states or times."""
        from online_evidence_content import actor_binding_fit, query_focus

        people = bool(re.search(r"他|她|那个人|这人|那位|\b(?:he|she|they|that person|this person)\b", question, re.I))
        named_person = bool(re.search(r"乘客|司机|驾驶员|\b(?:driver|passenger|cyclist)\b", question, re.I))
        generic_object = _contains_any(question, ["它", "这个东西", "那个东西", "那个物体", "刚才那个",
                                                 "it", "that object", "this object", "that thing", "this thing"])
        source = latest.get("entities", []) if people else latest.get("focus_entities", [])
        if named_person and not people:
            source = []  # The question already names its actor; do not borrow an old object.
        identities = {}
        if has_pronoun:
            for entity in source or []:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("canonical_name") or entity.get("name") or "").strip()
                person_name = bool(re.search(r"\b(?:man|woman|person|driver|passenger|cyclist)(?!['’])\b|男子|女子|男人|女人|乘客|司机|驾驶员", name, re.I))
                if people and entity.get("type") != "person" and not person_name:
                    continue
                if not people and not generic_object and not query_focus(question).intersection(query_focus(name)):
                    continue
                # Parenthetical appearance descriptions and last_state are
                # observations, not durable identifiers for a historical query.
                name = re.split(r"[（(，,]|\b(?:wearing|dressed|holding|carrying|in)\b", name, maxsplit=1, flags=re.I)[0].strip()
                if people:
                    name = re.sub(r"(?:穿着?|戴着?|拿着?)[^，,]*?的", "", name)
                    name = re.split(r"穿着?|戴着?|拿着?", name, maxsplit=1)[0].strip()
                if people and (not name or re.search(r"['’]s|的(?:手|头|脸)|\b(?:hand|head|gaze|shirt|jacket)\b", name, re.I)):
                    continue
                if not name or (people and actor_binding_fit(question, {"description": name}) == 0):
                    continue
                identities.setdefault(name.casefold(), {"name": name, "type": "person" if people else entity.get("type", "object")})
        # Generic "person" adds no second identity to a named actor. Distinct
        # people/objects remain ambiguous rather than inheriting a random one.
        if len(identities) > 1:
            for generic in ("person", "人"):
                if len(identities) > 1:
                    identities.pop(generic, None)
        entities = list(identities.values())
        valid = len(entities) == 1
        resolved = CacheContext(cache_hit=True, is_followup=has_pronoun,
            entity_binding_valid=valid, confidence=0.75 if valid else 0.4,
            resolved_question=f"Regarding {entities[0]['name']}: {question}" if valid else question,
            referenced_entities=entities[:4],
            reason="fresh historical search inherits identity only, not cached event times or states").to_dict()
        resolved["temporal_relation"] = "related"
        if valid:
            resolved["entity_turn_id"] = latest.get("turn_id")
        elif len(entities) > 1:
            resolved["reference_ambiguous"] = True
        return resolved

    def _match_turn(self, question, context, latest, bound_focus=None):
        """Match event content, not the generic wording of an interrogative."""
        from online_evidence_content import query_terms, query_focus, actor_binding_fit, CONCEPT_GROUPS
        stop = set("the a an at that this time when did does do was were is are he she it they what where how then after before previous person people man woman answer about with and of to in on during front rear left right passenger driver cyclist".split())
        generic = re.compile(r"前排|后排|前座|后座|右边|左边|右侧|左侧|那位|这位|的时候|什么|怎么|如何|在做|做了|刚才|刚刚|当时|那时|现在|回到|说回|开头|乘客|司机")
        def normalized(text):
            # Small linguistic variants, not identities or inferred actions.
            text = re.sub(r"低着头|低下头|头低下", "低头", str(text or "").lower())
            return re.sub(r"羽绒服|厚外套|外套|上衣|衣着|服饰|clothing|outfit|clothes", "衣服", text)
        def terms(text):
            return query_terms(generic.sub(" ", normalized(text)), expand=False) - stop
        reference = re.split(_EVENT_TIME, question, maxsplit=1)[0]
        when = re.search(r"\bwhen\b\s+([^,?;]+)", reference, re.I)
        if when:
            reference = when[1]
        specific_event = bool(re.search(r"\bwhen\b|" + _EVENT_TIME, question, re.I))
        # "When we discussed clothing" locates a conversational topic; it
        # does not assert that the person performed an action called "discuss".
        discussion = re.search(r"(?:聊|谈|讨论|说)(?:到|起|过|的)?([^，。？?]*?)(?:的时候|那会儿|时)", reference + ("的时候" if "的时候" in question else ""))
        discussion_terms = terms(discussion[1]) if discussion else set()
        action_groups = (
            ("低头", "俯身"), ("拿起", "举起", "lift", "lifted", "lifts", "pick up", "picked up"),
            ("放下", "put down", "puts down", "lower", "lowered"),
            ("掉在", "掉落", "drop", "dropped", "drops"),
            ("打开", "open", "opened", "opens"), ("关闭", "closed", "close", "closes"),
        )
        topic = " ".join(str(entity.get("name") or entity.get("canonical_name") or "")
                         for entity in (bound_focus or []))
        topic_terms = terms(topic)
        requested = terms(reference if specific_event else question) | topic_terms
        focus = query_focus(reference + " " + topic)
        object_terms = set()
        for group in CONCEPT_GROUPS[:12]:
            if any(query_focus(alias) & focus for alias in group):
                for alias in group:
                    object_terms.update(terms(alias))
        # Added topic words (including descriptive attributes) cannot prove
        # the action introduced by the current question.
        predicates = requested - object_terms - topic_terms
        scored = []
        actor_groups = [{"driver", "司机", "驾驶员"}, {"passenger", "乘客"}, {"cyclist", "骑车人"}]
        def actors(text):
            return {index for index, group in enumerate(actor_groups) if _contains_any(str(text), list(group))}
        query_actors = actors(reference)
        turns = context.get("recent_interactions", [])
        for turn in turns:
            turn_question = str(turn.get("question") or "")
            # Earlier/later clauses can contain the actual requested event.
            # Do not erase the rest of an answer at the first "before".
            answer = str(turn.get("answer") or "")
            if topic:
                candidate_topics = turn.get("focus_entities") or []
                candidate_topic = (" ".join(str(entity.get("name") or entity.get("canonical_name") or "")
                                            for entity in candidate_topics)
                                   if candidate_topics else turn_question + " " + answer)
                if not InteractionCache._focus_name_matches(candidate_topic, topic):
                    continue
            turn_actors = actors(turn_question + " " + answer)
            if query_actors and turn_actors and not query_actors.intersection(turn_actors):
                continue
            if actor_binding_fit(reference, {"description": answer or turn_question}) == 0:
                continue
            # Only affirmative clauses establish an observed action. A word in
            # "could not confirm X" must not outrank a real observation of X.
            affirmative = re.sub(
                r"(?:\b(?:not|never|cannot|unable to|unclear|uncertain|maybe|might|may|possibly|perhaps)\b|没有|并未|未能|不能|无法|没|不确定|可能|疑似|看不清|看不出)([^,.;。；，]*)",
                " ", answer, flags=re.I)
            observed = terms(affirmative)
            named = terms(turn_question)
            candidate_focus = query_focus(turn_question + " " + answer)
            topic_match = bool(discussion_terms and discussion_terms.intersection(named))
            if not topic_match and focus and candidate_focus and not focus.intersection(candidate_focus):
                continue
            # Sharing just "phone/cup" cannot establish a specific lift/drop
            # event. Its discriminating action must also have support.
            if specific_event and not topic_match and predicates and not predicates.intersection(observed):
                continue
            evidence_match = predicates.intersection(observed)
            direct = bool(evidence_match) or topic_match
            overlap = requested.intersection(named | observed)
            action_match = any(_contains_any(normalized(reference), list(group))
                               and _contains_any(normalized(affirmative), list(group))
                               for group in action_groups)
            if specific_event and not topic_match and len(overlap) < 2 and not action_match:
                continue  # One shared unknown object (e.g. door) is not its action.
            # In a conversational topic reference, score its descriptors,
            # rather than letting the new question's object win the binding.
            # Reuse the shared concept aliases; this adds no task-specific
            # person, object combination, timestamp, or answer.
            topic_strength = int(topic_match)
            if topic_match and discussion:
                discussion_groups = [group for group in CONCEPT_GROUPS
                                     if _contains_any(discussion[1], list(group))]
                topic_strength += sum(_contains_any(normalized(turn_question + " " + affirmative),
                                                    [normalized(alias) for alias in group])
                                      for group in discussion_groups)
            score = (topic_strength, direct, len(overlap), len(requested & observed))
            if len(overlap) >= 2 or (specific_event and direct) or topic_match:
                scored.append((score, turn))
        if not scored:
            # Legacy callers may provide a bounded previous scene without a
            # turn list. Real turns with no matching event are never success.
            return (latest, False) if not turns and not specific_event else ({}, True)
        best = max(score for score, _ in scored)
        matches = [turn for score, turn in scored if score == best]
        if best[2] < 2 and not best[0] and not (specific_event and best[1]):
            return latest, True  # One shared noun/cue is not a unique event.
        return (matches[0], False) if len(matches) == 1 else (latest, True)

    def _format_ranges(self, ranges: list[dict[str, Any]]) -> str:
        parts = []
        for item in ranges:
            start = item.get("start")
            end = item.get("end")
            if start is not None and end is not None:
                parts.append(f"{float(start):.1f}-{float(end):.1f}s")
        return ", ".join(parts)

    def _expand_after_ranges(self, ranges: list[dict[str, Any]], latest_seconds: float | None = None) -> list[dict[str, Any]]:
        expanded = []
        for item in ranges:
            end = float(item.get("end") or item.get("start") or 0.0)
            if latest_seconds is not None and end > float(latest_seconds):
                continue
            expanded.append({
                "start": end,
                "end": min(end + float(self.followup_window_seconds), float(latest_seconds)) if latest_seconds is not None else end + float(self.followup_window_seconds),
                "score": item.get("score", item.get("confidence", 0.7)),
                "source": "interaction_cache_followup_after",
            })
        return expanded

    def _entity_clause(self, names: list[str]) -> str:
        names = [name for name in names if name][:4]
        return f" involving {', '.join(names)}" if names else ""

    def _time_clause(self, range_text: str) -> str:
        return f" around {range_text}" if range_text else ""
