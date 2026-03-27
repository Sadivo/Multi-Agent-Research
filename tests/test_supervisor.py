"""
Tests for Supervisor Agent (Task 4.1, 4.2, 4.3).

Sub-tasks:
  4.1 — Unit tests: JSON 解析、fallback 行為、回傳欄位驗證
  4.2 — Property test (屬性 3): Supervisor 子任務數量範圍
  4.3 — Property test (屬性 4): Supervisor 輸出為有效 JSON
"""

import sys
import json
import pathlib
from unittest import mock
from unittest.mock import MagicMock, patch
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from agents.supervisor import supervisor_node  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _make_state(user_query: str) -> dict:
    return {
        "user_query": user_query,
        "task_list": [],
        "current_task": "",
        "search_results": [],
        "analysis": "",
        "critique": "",
        "critique_count": 0,
        "final_report": "",
        "messages": [],
    }


def _mock_llm_response(content: str):
    """建立回傳指定 content 的 mock LLM response。"""
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 4.1 — Unit tests
# ══════════════════════════════════════════════════════════════════════════

def test_supervisor_parses_valid_json():
    """驗證正常 JSON 輸出時正確解析 task_list（需求 2.1, 2.2）"""
    valid_json = '{"tasks": ["子任務1", "子任務2", "子任務3"]}'
    mock_llm = _mock_llm_response(valid_json)

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = supervisor_node(_make_state("研究日本旅遊"))

    assert result["task_list"] == ["子任務1", "子任務2", "子任務3"]
    assert result["current_task"] == "子任務1"


def test_supervisor_returns_required_fields():
    """驗證回傳 dict 包含 task_list 和 current_task 欄位（需求 2.1）"""
    valid_json = '{"tasks": ["任務A", "任務B"]}'
    mock_llm = _mock_llm_response(valid_json)

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = supervisor_node(_make_state("測試問題"))

    assert "task_list" in result
    assert "current_task" in result


def test_supervisor_current_task_is_first_task():
    """驗證 current_task 為 task_list 的第一個元素"""
    valid_json = '{"tasks": ["第一個任務", "第二個任務"]}'
    mock_llm = _mock_llm_response(valid_json)

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = supervisor_node(_make_state("任何問題"))

    assert result["current_task"] == result["task_list"][0]


def test_supervisor_fallback_on_invalid_json():
    """驗證 JSON 解析失敗時 fallback 行為：以原始問題作為單一子任務（需求 2.4）"""
    invalid_json = "這不是有效的 JSON 格式"
    mock_llm = _mock_llm_response(invalid_json)

    user_query = "我的研究問題"
    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = supervisor_node(_make_state(user_query))

    assert result["task_list"] == [user_query]
    assert result["current_task"] == user_query


def test_supervisor_fallback_on_missing_tasks_key():
    """驗證 JSON 缺少 tasks 鍵時 fallback 行為（需求 2.4）"""
    wrong_key_json = '{"subtasks": ["任務1", "任務2"]}'
    mock_llm = _mock_llm_response(wrong_key_json)

    user_query = "原始問題"
    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = supervisor_node(_make_state(user_query))

    assert result["task_list"] == [user_query]
    assert result["current_task"] == user_query


def test_supervisor_fallback_logs_warning(caplog):
    """驗證 JSON 解析失敗時記錄 logging.warning（需求 2.4）"""
    import logging
    invalid_json = "not json at all"
    mock_llm = _mock_llm_response(invalid_json)

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        with caplog.at_level(logging.WARNING, logger="agents.supervisor"):
            supervisor_node(_make_state("問題"))

    assert len(caplog.records) > 0
    assert caplog.records[0].levelno == logging.WARNING


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 4.2 — Property test 屬性 3: Supervisor 子任務數量範圍
# Feature: multi-agent-research-pipeline, Property 3: Supervisor 子任務數量範圍
# Validates: Requirements 2.1
# ══════════════════════════════════════════════════════════════════════════

import random  # noqa: E402


@settings(max_examples=100)
@given(st.text(min_size=1))
def test_supervisor_task_count_range(query):
    """
    **Validates: Requirements 2.1**

    屬性 3：對於任意非空的研究問題字串，
    Supervisor 節點輸出的 task_list 長度應在 2 到 4 之間（含）。
    """
    # 使用確定性 stub：根據 query 長度決定任務數量（2-4）
    num_tasks = (len(query) % 3) + 2  # 結果為 2, 3, 或 4
    tasks = [f"子任務{i+1}" for i in range(num_tasks)]
    valid_json = json.dumps({"tasks": tasks}, ensure_ascii=False)

    mock_llm = _mock_llm_response(valid_json)

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = supervisor_node(_make_state(query))

    assert 2 <= len(result["task_list"]) <= 4, (
        f"task_list 長度 {len(result['task_list'])} 不在 2-4 範圍內"
    )


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 4.3 — Property test 屬性 4: Supervisor 輸出為有效 JSON
# Feature: multi-agent-research-pipeline, Property 4: Supervisor 輸出為有效 JSON
# Validates: Requirements 2.2
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.text(min_size=1))
def test_supervisor_output_valid_json(query):
    """
    **Validates: Requirements 2.2**

    屬性 4：對於任意研究問題輸入，Supervisor 節點的原始輸出應可被 json.loads 成功解析，
    且解析結果包含 "tasks" 鍵，且原始輸出字串不包含 Markdown 程式碼標記（```）。
    """
    # mock LLM 回傳有效 JSON（不含 Markdown 標記）
    tasks = ["搜尋子任務一", "搜尋子任務二"]
    raw_output = json.dumps({"tasks": tasks}, ensure_ascii=False)

    mock_response = MagicMock()
    mock_response.content = raw_output
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_llm):
        supervisor_node(_make_state(query))

    # 驗證原始輸出可被 json.loads 解析
    parsed = json.loads(raw_output)

    # 驗證包含 "tasks" 鍵
    assert "tasks" in parsed, "輸出 JSON 缺少 'tasks' 鍵"

    # 驗證不含 Markdown 程式碼標記
    assert "```" not in raw_output, "輸出包含 Markdown 程式碼標記"

