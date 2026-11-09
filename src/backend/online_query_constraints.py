"""Explicit video-time constraints shared by routing, retrieval and packing.

Only recognized time expressions become hard constraints. Actor and predicate
checks are conservative: unknown identity is not treated as a contradiction.
"""
from __future__ import annotations

import re
import math
from copy import deepcopy
from typing import Any

from online_time import event_time_seconds


def _number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        digits = dict(zip("零一二三四五六七八九", range(10)))
        value = value.replace("两", "二")
        if "十" in value:
            a, b = value.split("十", 1)
            return float(digits.get(a, 1) * 10 + digits.get(b, 0))
        return float(digits.get(value, 0))


def latest_seconds(runtime: dict[str, Any]) -> float | None:
    ends = []
    for name in ("mcur_time_span", "mst_time_span", "mlt_time_span"):
        span = runtime.get(name) or []
        if len(span) > 1:
            end = event_time_seconds(span[1])
            if end is not None:
                ends.append(end)
    return max(ends) if ends else None


def observed_query_time(runtime: dict[str, Any]) -> float | None:
    """Anchor relative requests to this observed stream, not another LT clock."""
    current_observed = runtime.get("current_ready") is not False and runtime.get("mcur_ready") is not False
    if current_observed:
        for key in ("current_time", "window_end_time"):
            value = event_time_seconds(runtime.get(key))
            if value is not None:
                return value
    span = runtime.get("mcur_time_span") or []
    if len(span) > 1 and current_observed:
        value = event_time_seconds(span[1])
        if value is not None:
            return value
    fallback = runtime if current_observed else {key: value for key, value in runtime.items() if key != "mcur_time_span"}
    return latest_seconds(fallback)


def resolve_query_constraints(
    question: str, *, query_constraints: dict[str, Any] | None = None,
    latest: float | None = None,
) -> dict[str, Any]:
    """Copy an authoritative request scope, or parse the original question.

    An empty dictionary is intentional. Cache/search wording must not fill it
    with new time requirements. Copies keep derived flags request-local.
    """
    return deepcopy(query_constraints) if query_constraints is not None else parse_query_constraints(question, latest)


def prompt_query_constraints(constraints: dict[str, Any] | None) -> dict[str, Any]:
    """Expose time boundaries and positive hints, not parser misses as facts."""
    constraints = constraints or {}
    hints = {key: constraints[key] for key in ("summary", "requires_current", "requires_history", "temporal_preference")
             if constraints.get(key)}
    hints.update({key: True for key, value in (constraints.get("requirements") or {}).items() if value})
    return {"requested_time_ranges": constraint_windows(constraints), "retrieval_hints": hints,
            "interpretation": "Hints are incomplete, not semantic judgments. Answer the original question; a missing hint does not negate a requirement."}


def _parse_single_query_constraints(question: str, latest: float | None = None) -> dict[str, Any]:
    q = str(question or "").lower()
    summary = bool(re.search(r"总结|概括|依次|主要发生|summary|summari[sz]|overview|recap|捋一遍|捋一下|(?:请|帮我|帮忙).{0,12}整理.{0,30}(?:视频|事情|事件|变化|发生).{0,20}过程|(?:事情|事件|变化|发生).{0,20}过程.{0,8}整理|梳理|回顾|rundown|walk me through|从头到尾|从开头.*(?:结束|最后|结尾)", q))
    summary = summary or bool(re.search(r"按(?:时间|先后|顺序)(?:说|讲|整理|描述|梳理)|\bin chronological order\b", q))
    stage_mentions = [
        bool(re.search(pattern, q))
        for pattern in (
            r"开头|开始|起初|早期|最初|\bearly\b|\bbeginning\b|\bstart\b",
            r"中间|中段|中期|\bmiddle\b",
            r"最后|结尾|后段|后期|末段|结束|后来|\blate\b|\bend\b|\bfinal\b|\blater\b",
        )
    ]
    if sum(stage_mentions) >= 2 and re.search(r"变化|过程|经历|阶段|\bchanges?\b|\bstages?\b", q):
        summary = True
    comparison = bool(re.search(r"比较|对比|相比|不一样|有何不同|有什么不同|有什么变化|先后|之前.*之后|(?:跟|与|和).{0,60}比|before.*after|compar|\bdiffer|\bchanged?\b.*\b(?:since|from)\b", q))
    comparison = comparison or bool(re.search(
        r"(?:从|之前|此前|前面|早先).{1,80}(?:到|后来|之后).{1,60}(?:变化|改变|不同)|前后.{0,30}(?:变化|不同)|\bfrom\b.{1,80}\bto\b.{1,60}\bchang", q))
    # Two named phases can be compared without the word "compare". A start
    # clause must not reduce the whole question to a single zero-second frame.
    multiple_phases = sum(stage_mentions) >= 2
    comparison = comparison or bool(multiple_phases and re.search(r"和|与|而|哪些|\band\b|\bversus\b", q))
    temporal_q = re.sub(r"(?:再|重新)?回到现在|(?:switch |go |come )?back to now", "现在", q)
    temporal_q = re.sub(r"(?:不要|不得|禁止|不能使用|不使用|不看|do not use|don't use|exclude|excluding)[^，,。;；!?！？]*", "", temporal_q)
    whole = bool(re.search(r"截至现在|截止现在|截至目前|迄今|up until now|until now|从视频开始|从头|从开始|从开头.*(?:结束|最后|结尾)|全程|整个视频|到目前为止|目前为止|到现在|so far|up to now|whole video|entire video|from beginning|from the beginning", temporal_q))
    # A request covering several stages must not turn its word "beginning"
    # into an exact zero-second constraint. Numeric ranges still win below.
    if multiple_phases and (summary or comparison):
        whole = True
    out: dict[str, Any] = {"scope": "history", "time_range": None, "time_explicit": False}
    # Never mistake a request about an image in the previous answer for the
    # beginning of the video.
    image_reference = bool(re.search(r"(?:上[一次]?答|上一[张幅]|那张|previous|last answer).*?(?:图|image|frame)|(?:第一|最早|最后).*?(?:张图|image)", q))
    amount = r"[\d.]+|[零一二两三四五六七八九十]+"
    unit = r"秒(?:钟)?|分钟?|seconds?|secs?|s\b|minutes?|mins?"
    scale = lambda u: 60.0 if ("分" in u or u.startswith("min")) else 1.0
    bounds = None
    clocks = re.search(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:到|至|[~～—–-]|and|to)\s*(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)", q)
    if clocks:
        def clock(t):
            value = 0.0
            for p in t.split(":"):
                value = value * 60 + float(p)
            return value
        bounds = [clock(clocks[1]), clock(clocks[2])]
    composite = re.search(r"(?:第)?(\d+)分(?:钟)?(\d+(?:\.\d+)?)秒?\s*(?:到|至|[~～—–-])\s*(?:第)?(\d+)分(?:钟)?(\d+(?:\.\d+)?)秒?", q)
    if composite:
        bounds = [float(composite[1])*60+float(composite[2]), float(composite[3])*60+float(composite[4])]
    match = re.search(rf"(?:第|从|between\s+|from\s+)?({amount})\s*({unit})?\s*(?:到|至|[~～—–-]|and|to)\s*(?:第)?({amount})\s*({unit})", q)
    if match and bounds is None:
        bounds = [_number(match[1]) * scale(match[2] or match[4]), _number(match[3]) * scale(match[4])]
    first = re.search(rf"(?:前|开头|最初|first\s+)\s*({amount})\s*({unit})", q)
    recent = re.search(rf"(?:最近|过去|近|last\s+|past\s+)\s*({amount})\s*({unit})", q)
    if bounds is not None:
        out["scope"] = "range"
    elif first:
        bounds = [0.0, _number(first[1]) * scale(first[2])]
        out["scope"] = "range"
    elif recent and latest is not None:
        bounds = [max(0.0, latest - _number(recent[1]) * scale(recent[2])), latest]
        out["scope"] = "recent"
    else:
        point = None
        for candidate in re.finditer(rf"(?:第|在|at\s+|around\s+)?(\d+(?:\.\d+)?)\s*({unit})", q):
            prefix = q[max(0, candidate.start()-24):candidate.start()]
            suffix = q[candidate.end():]
            is_duration = bool(
                re.search(r"(?:相隔|间隔|相差|持续|时长|长达)(?:至少|最少|不小于|不少于|超过|大于|约|大约|\s)*$", prefix)
                or re.match(r"\s*(?:apart|long\b|in duration)", suffix)
                or re.search(r"(?:lasting|duration of)\s*$", prefix)
            )
            is_video_duration = bool(re.match(r"\s*(?:的)?(?:视频|短片|video|clip)", suffix))
            if not is_duration and not is_video_duration:
                point = candidate
                break
        if point:
            t = float(point[1]) * scale(point[2])
            suffix = q[point.end():point.end()+12]
            prefix = q[max(0, point.start()-10):point.start()]
            if re.match(r"\s*(?:之后|以后|后)", suffix) or re.search(r"after\s*$", prefix):
                bounds = [t, latest] if latest is not None else [t, 1e12]
                out["scope"] = "after"
            elif re.match(r"\s*(?:之前|以前|前)", suffix) or re.search(r"before\s*$", prefix):
                bounds = [0.0, t]
                out["scope"] = "before"
            else:
                # A point is a point; frame selection may only choose an exact
                # saved frame, rather than silently substituting a later image.
                bounds = [t, t]
                out["scope"] = "point"
        elif not whole and not image_reference and not (comparison and re.search(r"现在|当前|此刻|\bnow\b|\bcurrent\b", q)) and re.search(r"视频开始|视频开头|最开始|开头|起初|at the start|beginning of (?:the )?video|first frame", q):
            bounds = [0.0, 0.0]
            out["scope"] = "point"
        elif not whole and not image_reference and re.search(r"视频结尾|视频最后|最后画面|at the end|end of (?:the )?video|last frame", q) and latest is not None:
            bounds = [latest, latest]
            out["scope"] = "point"
    if bounds is not None:
        if bounds[1] is not None and bounds[0] > bounds[1]:
            # An explicitly requested future range is empty, not a reversed
            # range extending into the past.
            if out["scope"] == "after":
                bounds = [bounds[0], bounds[0]]
            else:
                bounds = sorted(bounds)
        out.update(time_range=bounds, time_explicit=True)
    current_q = re.sub(r"截至现在|截止现在|截至目前|迄今|到目前为止|目前为止|到现在|so far|up (?:un)?til now|up to now|until now", "", temporal_q)
    # "Up to now" can bound a history request; "by now can we still..."
    # also asks for a present observation. Do not erase that endpoint.
    if re.search(r"到现在(?=还|能|是否|有无|怎么样|如何|什么)|\b(?:by|until) now\s+(?:can|is|are|does|do|still)\b", temporal_q):
        current_q += " 现在"
    current_q = re.sub(r"(?:不要|不看|不使用|do not use|don't use)[^，,。;；!?！？]*", "", current_q)
    # 出现在 means "appeared at/in", not a separate request about 现在 (NOW).
    current_q = current_q.replace("出现在", "出现于")
    current = bool(re.search(r"现在|此刻|当前|眼下|目前|正在|当下|眼前|\bnow\b|\bcurrently\b|\bcurrent\b|at (?:the|this) moment|at present|what (?:do|can) you see|what(?:'s| is) on (?:the )?screen|这是什么|这是啥|看到什么|能看到什么", current_q))
    # Progressive activity questions describe the observed present even when
    # NOW is omitted. Historical clauses below still override this cue.
    current = current or bool(re.search(r"在(?:做|干|忙)(?:什么|啥)|\bwhat\s+(?:is|are)\b.{0,60}\bdoing\b", current_q))
    past = bool(re.search(r"当时|那时|那时候|\bat that time\b|\bback then\b|刚才|刚刚|之前|以前|过去|开头|开始|最初|最近|\bearlier\b|\bbefore\b|\bbeginning\b|\bprevious\b|\brecent", temporal_q))
    event_reference = bool(re.search(r"的时候|之时|那会儿|那一刻|(?<![当此那何小])时(?=[，,。？?])|\bwhen\b.+[,?]|\bwhile\b", temporal_q))
    past = past or event_reference or bool(re.search(r"早先|较早|先前|前面.{1,30}(?:那段|那阵|时)", temporal_q))
    # Generic perception language inherits an explicit past reference. Only a
    # separate present-time cue can request the current side of a comparison.
    explicit_current = bool(re.search(r"现在|此刻|当前|眼下|目前|当下|\bnow\b|\bcurrently\b|\bcurrent\b|at (?:the|this) moment|at present", current_q))
    # A past comparison clause supplies the reference side. Without another
    # historical target, "compared with earlier, has it changed?" asks about
    # the present even when NOW is omitted. Numeric/past targets still win.
    comparison_clause = re.search(r"(?:跟|与|和)([^，,。！？?]{1,60}?)(?:相比|比)(?=[，,。！？?\s]|$)", temporal_q) if comparison else None
    implicit_current = False
    if comparison_clause and not out["time_explicit"] and re.search(
        r"刚才|刚刚|之前|以前|早先|先前|前面|前边|开头|最初|的时候|那时|那会儿", comparison_clause[1]):
        target_clause = temporal_q[:comparison_clause.start()] + temporal_q[comparison_clause.end():]
        implicit_current = not bool(re.search(r"当时|那时|那会儿|早先|先前|此前|之前|以前|过去|开头|开始|最初", target_clause))
        current = current or implicit_current
    if (past or out["time_explicit"]) and not (explicit_current or implicit_current):
        current = False
    historical_comparison = comparison and bool(re.search(
        r"前面|前边|早先|先前|此前|之前|以前|刚才|刚刚|开头|开始|最初|上次|上一|那会儿|那时候|的时候|\bearlier\b|\bprevious\b|\bbefore\b", q))
    mixed = current and ((explicit_current and past) or historical_comparison
                         or (comparison and (whole or out["time_explicit"])))
    out["requires_current"] = bool(current and (not whole or mixed))
    out["requires_history"] = bool(not current or whole or mixed or out["time_explicit"])
    # A summary ending at the observed present needs both historical context
    # and the latest state; a cached historical point cannot replace its end.
    if summary and whole and not out["time_explicit"] and re.search(
        r"到现在|截至现在|截止现在|截至目前|迄今|到目前为止|目前为止|so far|up (?:un)?til now|up to now|until now", temporal_q):
        out["requires_current"] = True
    if not out["time_explicit"]:
        out["scope"] = "history" if whole or mixed else "current" if current else "recent" if re.search(r"刚才|刚刚|最近|just now|recently|\bjust (?:happened|did|do)\b", q) else "history"
    out["summary"] = summary
    out["event_reference"] = event_reference
    initial_discovery = bool(re.search(r"(?:刚|最初|开始)(?:能|可以|能够)|\bfirst\b.{0,30}\b(?:visible|identif\w*|recogniz\w*|discern\w*)\b", q))
    out["temporal_preference"] = ("earlier" if not out["time_explicit"] and not summary and not multiple_phases
        and (initial_discovery or re.search(r"刚开始|较早|早先|最初|起初|最早|首次|第一次|\bearliest\b|\binitially\b|\bearlier\b", q)) else None)
    if multiple_phases and comparison and stage_mentions[0] and not summary and not out["time_explicit"]:
        # Preserve evidence of the initial state as well as later matching
        # observations. A clearer later image cannot establish the earlier state.
        out["temporal_preference"] = "earlier_and_later"
    out["requirements"] = {
        "earliest": (initial_discovery or bool(re.search(r"最早|第一次|首次|first time|earliest", q))) and not image_reference,
        "simultaneous": bool(re.search(r"同时|一边.+一边|(?:和|与).+都.+(?:时|时候)|既.+又|又.+又|simultaneous|at the same time|\bwhile\b|\bsimultaneously\b", q)),
        "comparison": comparison,
    }
    return out


def constraint_windows(constraints: dict[str, Any] | None) -> list[list[float]]:
    """Exact requested union. Never replace disjoint windows with a hull."""
    constraints = constraints or {}
    if constraints.get("time_empty"):
        return [[1.0, 0.0]]  # Explicit empty interval, never an unrestricted query.
    windows = constraints.get("time_ranges")
    if windows is None:
        window = constraints.get("time_range")
        windows = [window] if window else []
    return [list(window) for window in windows if isinstance(window, (list, tuple)) and len(window) == 2]


def constraint_window(constraints: dict[str, Any] | None):
    windows = constraint_windows(constraints)
    return windows[0] if len(windows) == 1 else windows or None


def reserve_time_coverage(items, windows, limit, span=None):
    """Keep a ranked representative of each requested window before score fill.

    Prefer a narrow anchor over a coarse event covering several windows, so a
    broad summary cannot consume every side's representation by itself.
    """
    span = span or (lambda item: (item.get("start_time", item.get("timestamp")), item.get("end_time", item.get("timestamp"))))
    selected = []
    for window in windows:
        matches = [item for item in items if overlaps(*span(item), window)]
        if not matches:
            continue
        local = [item for item in matches if sum(overlaps(*span(item), other) for other in windows) == 1]
        best = (local or matches)[0]
        if best not in selected:
            selected.append(best)
    return (selected + [item for item in items if item not in selected])[:max(0, limit)]


def parse_query_constraints(question: str, latest: float | None = None) -> dict[str, Any]:
    """Parse every supported explicit span/point, preserving disjoint windows.

    The legacy parser still supplies intent flags and nonnumeric references.
    Unparsed numeric time expressions disable hard filtering rather than
    silently restricting a compound question to its first recognized span.
    """
    original_q = str(question or "").lower()
    # A negative instruction is not another requested observation. Parse its
    # temporal exclusion separately, masking it from positive point matching.
    exclusions = []
    q_chars = list(original_q)
    for clause in re.finditer(r"(?:不要|不得|禁止|不能使用|不使用|不看|do not use|don't use|exclude|excluding)[^，,。;；!?！？]*", original_q):
        negative = _parse_single_query_constraints(clause[0], latest)
        bound = negative.get("time_range")
        if not bound:
            continue
        a,b = bound
        if negative.get("scope") == "after":
            a = math.nextafter(a, math.inf)  # “after” excludes strictly later frames.
        elif negative.get("scope") == "before":
            b = math.nextafter(b, -math.inf)
        exclusions.append([a,b])
        q_chars[clause.start():clause.end()] = " " * len(clause[0])
    q = "".join(q_chars)
    out = _parse_single_query_constraints(q, latest)
    image_reference = bool(re.search(r"上一答|上一张|previous.*(?:image|answer)|那张", q))
    number = r"(?:\d+(?:\.\d+)?|[零一二两三四五六七八九十]+)"
    unit = r"(?:秒(?:钟)?|分钟?|seconds?|secs?|s\b|minutes?|mins?)"
    scale = lambda u: 60.0 if "分" in u or u.startswith("min") else 1.0
    found = []
    occupied = []
    ignored_duration = False
    def add(match, bounds):
        if any(match.start() < b and match.end() > a for a,b in occupied):
            return
        occupied.append(match.span()); found.append((match.start(), list(bounds)))
    def clock(t):
        value = 0.0
        for part in t.split(":"): value = value * 60 + float(part)
        return value
    for match in re.finditer(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:到|至|[~～—–-]|to)\s*(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)", q):
        add(match, sorted([clock(match[1]), clock(match[2])]))
    for match in re.finditer(r"(?:第)?(\d+)分(?:钟)?(\d+(?:\.\d+)?)秒?\s*(?:到|至|[~～—–-])\s*(?:第)?(\d+)分(?:钟)?(\d+(?:\.\d+)?)秒?", q):
        add(match, sorted([float(match[1])*60+float(match[2]), float(match[3])*60+float(match[4])]))
    for match in re.finditer(rf"(?:第|从|between\s+|from\s+)?({number})\s*({unit})?\s*(?:到|至|[~～—–-]|and|to)\s*(?:第)?({number})\s*({unit})", q):
        if re.search(r"\band\b", match[0]) and not re.match(r"between\b", match[0]):
            continue
        add(match, sorted([_number(match[1])*scale(match[2] or match[4]), _number(match[3])*scale(match[4])]))
    for match in re.finditer(rf"((?:第)?{number}(?:\s*(?:和|与|、|and|,|，)\s*(?:第)?{number})+)\s*({unit})", q):
        if any(match.start()<b and match.end()>a for a,b in occupied):
            continue
        values = [_number(value)*scale(match[2]) for value in re.findall(number, match[1])]
        add(match, [values[0]]*2)
        found.extend((match.start()+i/100, [value,value]) for i,value in enumerate(values[1:],1))
    for match in re.finditer(rf"(?:前|开头|最初|first\s+|最近|过去|近|last\s+|past\s+)\s*({number})\s*({unit})", q):
        n = _number(match[1])*scale(match[2])
        recent = bool(re.match(r"最近|过去|近|last|past", match[0]))
        if recent and latest is None:
            continue
        add(match, [max(0, latest-n), latest] if recent else [0.0, n])
    for match in re.finditer(r"(?:第)?(\d+)分(?:钟)?(\d+(?:\.\d+)?)秒", q):
        add(match, [float(match[1])*60+float(match[2])]*2)
    for match in re.finditer(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)", q):
        add(match, [clock(match[1])]*2)
    for match in re.finditer(rf"(?:第|在|at\s+|around\s+)?({number})\s*({unit})", q):
        if any(match.start()<b and match.end()>a for a,b in occupied): continue
        prefix=q[max(0,match.start()-30):match.start()]; suffix=q[match.end():]
        duration_context = bool(out.get("summary") and (
            re.search(r"(?:这|该|本)(?:一?段)?\s*$|(?:this|these|the whole|the entire|past)\s*$", prefix)
            and re.match(r"\s*(?:里|内|中|的|of|in|video|clip|footage|$)", suffix)))
        if duration_context:
            occupied.append(match.span()); ignored_duration = True; continue
        if re.search(r"(?:相隔|间隔|相差|持续|时长|长达)(?:至少|最少|不小于|不少于|超过|大于|约|大约|\s)*$|(?:lasting|duration of)\s*$",prefix) or re.match(r"\s*(?:apart|long\b|in duration|的?(?:视频|短片|video|clip))",suffix):
            occupied.append(match.span());continue
        t=_number(match[1])*scale(match[2])
        if re.search(r"(?:截至|截止|到)\s*第?$|until\s*$|up to\s*$|through\s*$", prefix):
            bounds=[0.0,t]; out["scope"]="prefix"
        elif re.match(r"\s*(?:之前和之后|之前与之后|前后)",suffix):
            add(match, [0.0,t])
            found.append((match.start()+0.1,[t,max(t,latest) if latest is not None else 1e12]))
            continue
        elif re.match(r"\s*(?:之后|以后|后)",suffix) or re.search(r"after\s*$",prefix):
            bounds=[t,max(t,latest) if latest is not None else 1e12];out["scope"]="after"
        elif re.match(r"\s*(?:之前|以前|前)",suffix) or re.search(r"before\s*$",prefix):
            bounds=[0.0,t];out["scope"]="before"
        else: bounds=[t,t]
        add(match,bounds)
    # Unsupported numeric units (hours/frames etc.) must not leave a partial
    # hard constraint assembled from only the easy part of a compound query.
    rest=list(q)
    for a,b in occupied:
        rest[a:b]=" "*(b-a)
    incomplete=bool(re.search(rf"{number}\s*(?:小时|hours?|帧|frames?|:\d{{2}})", "".join(rest)))
    if incomplete:
        out.update(time_range=None,time_ranges=[],time_explicit=False,time_parse_incomplete=True,scope="history")
        return out
    # Non-numeric comparison endpoints must also survive as separate points.
    if not out.get("summary") and not image_reference:
        compare = bool((out.get("requirements") or {}).get("comparison") or re.search(r"分别|respectively", q))
        beginning = bool(re.search(r"视频开始|视频开头|开头|beginning|first frame", q))
        ending = bool(re.search(r"视频结尾|视频最后|最后画面|结尾|last frame|end of (?:the )?video", q))
        if compare and beginning and ending:
            if latest is None:
                out.update(time_range=None,time_ranges=[],time_explicit=False,time_parse_incomplete=True,scope="history")
                return out
            found.extend([(-2,[0.0,0.0]),(len(q)+1,[latest,latest])])
    if found:
        windows=[]
        for _,bounds in sorted(found):
            if bounds not in windows:windows.append(bounds)
        out.update(time_ranges=windows,time_range=windows[0] if len(windows)==1 else None,time_explicit=True)
        if len(windows)>1:out["scope"]="multi_range"
        elif out["scope"] not in {"prefix","before","after","recent"}:out["scope"]="point" if windows[0][0]==windows[0][1] else "range"
    else:
        if ignored_duration:
            out.update(time_range=None, time_explicit=False, scope="history")
        out["time_ranges"]=[out["time_range"]] if out.get("time_range") else []
    if out.get("requires_current") and out.get("requires_history"):
        windows = constraint_windows(out)
        if not windows and re.search(r"视频开始|视频开头|开头|beginning of (?:the )?video|first frame", q):
            windows = [[0.0, 0.0]]
        if windows:
            if latest is None:
                # A historical point alone would filter out the explicitly
                # requested present side. Resolve both once NOW is known.
                out.update(time_range=None, time_ranges=[], time_explicit=False,
                           time_parse_incomplete=True, scope="history")
            else:
                if not any(lo <= latest <= hi for lo, hi in windows):
                    windows.append([latest, latest])
                out.update(time_ranges=windows, time_range=windows[0] if len(windows)==1 else None,
                           time_explicit=True, scope="multi_range" if len(windows)>1 else out["scope"])
    if exclusions:
        windows = constraint_windows(out) or [[0.0, latest if latest is not None else 1e12]]
        for a,b in exclusions:
            remaining = []
            for lo,hi in windows:
                if b < lo or a > hi:
                    remaining.append([lo,hi]); continue
                if lo < a:
                    remaining.append([lo, math.nextafter(a, -math.inf)])
                if hi > b:
                    remaining.append([math.nextafter(b, math.inf),hi])
            windows = remaining
        out.update(time_ranges=windows, time_range=windows[0] if len(windows)==1 else None,
                   time_explicit=True, time_empty=not windows, excluded_time_ranges=exclusions)
    return out


def overlaps(start: Any, end: Any, window: list[float] | None) -> bool:
    if not window:
        return True
    if isinstance(window[0], (list, tuple)):
        return any(overlaps(start, end, part) for part in window)
    if window[0] > window[1]:
        return False
    start = event_time_seconds(start)
    end = event_time_seconds(end) if end is not None else start
    if start is None or end is None:
        return False
    if window[0] == window[1]:
        return start <= window[0] <= end
    if start == end:
        return window[0] <= start <= window[1]
    return start < window[1] and end > window[0]


def time_fit(start: Any, end: Any, window: list[float] | None) -> float:
    if window and isinstance(window[0], (list, tuple)):
        return max((time_fit(start, end, part) for part in window), default=0.0)
    if not window or not overlaps(start, end, window):
        return 0.0
    a, b = event_time_seconds(start), event_time_seconds(end)
    b = a if b is None else b
    if a == b or window[0] == window[1]:
        return 1.0
    return min(1.0, max(0.0, min(b, window[1]) - max(a, window[0])) / max(1e-6, min(b-a, window[1]-window[0])))


def actor_fit(question: str, value: Any) -> float:
    """Prefer the requested visible position; never infer a driver's seat."""
    q = str(question or "").lower()
    if isinstance(value, dict):
        subject = value.get("actor") or value.get("entity")
        if not subject and isinstance(value.get("triple"), list) and value["triple"]:
            subject = value["triple"][0]
        text = str(subject or value.get("description") or value.get("action") or value).lower()
    else:
        text = str(value).lower()
    # A reference to a right hand/window is not the person's screen position.
    subject_clause = re.split(r"\b(?:holds?|looks?|sits?|raises?|touches|puts?|reaches|wears?|handles|turns?)\b", text, maxsplit=1)[0]
    if re.search(r"front|rear|left|right|前排|后排|左侧|右侧", subject_clause):
        text = subject_clause
    axes = [
        (r"右侧|右边|右前|前右|\bright\b", r"右侧|右边|右前|前右|\bright\b", r"左侧|左边|左前|前左|\bleft\b"),
        (r"左侧|左边|左前|前左|\bleft\b", r"左侧|左边|左前|前左|\bleft\b", r"右侧|右边|右前|前右|\bright\b"),
        (r"前排|前座|右前|左前|前右|前左|\bfront\b", r"前排|前座|右前|左前|前右|前左|\bfront\b", r"后排|后座|\brear\b|\bback[- ]seat"),
        (r"后排|后座|\brear\b|\bback[- ]seat", r"后排|后座|\brear\b|\bback[- ]seat", r"前排|前座|\bfront\b"),
    ]
    score = 1.0
    for requested, positive, negative in axes:
        if re.search(requested, q):
            if re.search(positive, text):
                continue
            if re.search(negative, text):
                return 0.0
            score *= 0.35
    return score
