from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from online_preprocess.text_normalization import normalize_user_visible_text_fields, simplify_chinese_text


def test_simplify_chinese_text():
    assert simplify_chinese_text("這個問題發生了什麼") == "这个问题发生了什么"


def test_normalize_user_visible_text_fields_keeps_ids_and_paths():
    payload = {
        "status": "done",
        "session_id": "abc123",
        "task_path": "/tmp/這個.json",
        "question": "這個畫面是什麼",
        "result": {"answer": "這是一個答案。"},
    }

    normalized = normalize_user_visible_text_fields(payload)

    assert normalized["question"] == "这个画面是什么"
    assert normalized["result"]["answer"] == "这是一个答案。"
    assert normalized["session_id"] == "abc123"
    assert normalized["task_path"] == "/tmp/這個.json"
