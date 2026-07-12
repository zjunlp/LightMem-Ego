from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


USER_VISIBLE_TEXT_KEYS = {
    "answer",
    "final_answer",
    "finalAnswer",
    "message",
    "question",
    "response",
    "text",
    "transcript",
}

_COMMON_TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "這": "这",
        "個": "个",
        "麼": "么",
        "甚": "什",
        "裏": "里",
        "裡": "里",
        "說": "说",
        "語": "语",
        "聽": "听",
        "問": "问",
        "題": "题",
        "答": "答",
        "現": "现",
        "畫": "画",
        "視": "视",
        "頻": "频",
        "發": "发",
        "生": "生",
        "錄": "录",
        "識": "识",
        "別": "别",
        "後": "后",
        "端": "端",
        "沒": "没",
        "認": "认",
        "為": "为",
        "與": "与",
        "對": "对",
        "時": "时",
        "間": "间",
        "顯": "显",
        "示": "示",
        "應": "应",
        "該": "该",
        "實": "实",
        "際": "际",
        "簡": "简",
        "體": "体",
        "繁": "繁",
        "轉": "转",
        "換": "换",
        "將": "将",
        "會": "会",
        "還": "还",
        "請": "请",
        "開": "开",
        "關": "关",
        "看": "看",
        "到": "到",
        "當": "当",
        "前": "前",
        "內": "内",
        "容": "容",
        "螢": "萤",
        "幕": "幕",
        "顏": "颜",
        "色": "色",
        "張": "张",
        "嗎": "吗",
        "呢": "呢",
        "點": "点",
        "長": "长",
        "鐘": "钟",
        "聲": "声",
        "音": "音",
        "輕": "轻",
        "重": "重",
        "啟": "启",
        "動": "动",
        "狀": "状",
        "態": "态",
        "據": "据",
        "庫": "库",
        "證": "证",
        "選": "选",
        "擇": "择",
        "幀": "帧",
        "圖": "图",
        "像": "像",
        "總": "总",
        "結": "结",
        "剛": "刚",
        "才": "才",
        "進": "进",
        "行": "行",
    }
)


def _normalization_enabled() -> bool:
    value = os.getenv("EM2MEM_NORMALIZE_SIMPLIFIED_CHINESE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _opencc_converter() -> Any | None:
    try:
        from opencc import OpenCC  # type: ignore

        return OpenCC("t2s")
    except Exception:
        return None


def simplify_chinese_text(text: str) -> str:
    if not text or not _normalization_enabled():
        return text
    converter = _opencc_converter()
    if converter is not None:
        return str(converter.convert(text))
    return text.translate(_COMMON_TRADITIONAL_TO_SIMPLIFIED)


def normalize_user_visible_text_fields(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key in USER_VISIBLE_TEXT_KEYS and isinstance(item, str):
                normalized[key] = simplify_chinese_text(item)
            else:
                normalized[key] = normalize_user_visible_text_fields(item)
        return normalized
    if isinstance(value, list):
        return [normalize_user_visible_text_fields(item) for item in value]
    return value
