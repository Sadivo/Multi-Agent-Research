"""
Tests for Search Agent (Task 3.1 & 3.2).

Sub-tasks:
  3.1 — Unit tests: URL 去重邏輯、API 錯誤處理
  3.2 — Property test (屬性 5): 搜尋結果 URL 唯一性
"""

import asyncio
import sys
import pathlib
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from agents.search import search_node, deduplicate_results  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 3.1 — Unit tests
# ══════════════════════════════════════════════════════════════════════════

def _make_state(task_list):
    return {
        "user_query": "test query",
        "task_list": task_list,
        "current_task": "",
        "search_results": [],
        "analysis": "",
        "critique": "",
        "critique_count": 0,
        "final_report": "",
        "messages": [],
    }


def test_url_deduplication_removes_duplicates():
    """驗證 URL 去重邏輯：輸入含重複 URL，輸出不含重複（需求 3.3）"""
    raw = [
        {"title": "A", "url": "https://example.com", "content": "c1", "score": 0.9},
        {"title": "B", "url": "https://other.com", "content": "c2", "score": 0.8},
        {"title": "C", "url": "https://example.com", "content": "c3", "score": 0.7},  # duplicate
    ]
    result = deduplicate_results(raw)
    urls = [r["url"] for r in result]
    assert len(urls) == len(set(urls)), "去重後仍有重複 URL"
    assert len(result) == 2


def test_url_deduplication_preserves_first_occurrence():
    """驗證去重保留第一次出現的結果"""
    raw = [
        {"title": "First", "url": "https://dup.com", "content": "first", "score": 0.9},
        {"title": "Second", "url": "https://dup.com", "content": "second", "score": 0.5},
    ]
    result = deduplicate_results(raw)
    assert len(result) == 1
    assert result[0]["title"] == "First"


def test_search_node_deduplicates_across_tasks():
    """驗證 search_node 對多個子任務的結果進行跨任務去重（需求 3.3）"""
    mock_results_task1 = [
        {"title": "T1", "url": "https://a.com", "content": "c1", "score": 0.9},
        {"title": "T2", "url": "https://b.com", "content": "c2", "score": 0.8},
    ]
    mock_results_task2 = [
        {"title": "T3", "url": "https://a.com", "content": "c3", "score": 0.7},  # duplicate
        {"title": "T4", "url": "https://c.com", "content": "c4", "score": 0.6},
    ]

    with patch("agents.search.get_search_tool") as mock_get_tool:
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = [mock_results_task1, mock_results_task2]
        mock_get_tool.return_value = mock_tool

        state = _make_state(["task1", "task2"])
        result = search_node(state)

    urls = [r["url"] for r in result["search_results"]]
    assert len(urls) == len(set(urls)), "跨任務去重後仍有重複 URL"
    assert len(result["search_results"]) == 3


def test_search_node_returns_correct_format():
    """驗證 search_node 回傳結果包含 title, url, content, score 欄位（需求 3.4）"""
    mock_results = [
        {"title": "Page", "url": "https://x.com", "content": "text", "score": 0.95},
    ]

    with patch("agents.search.get_search_tool") as mock_get_tool:
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = mock_results
        mock_get_tool.return_value = mock_tool

        state = _make_state(["some task"])
        result = search_node(state)

    assert "search_results" in result
    item = result["search_results"][0]
    assert "title" in item
    assert "url" in item
    assert "content" in item
    assert "score" in item


def test_search_node_returns_empty_list_on_api_error():
    """驗證 Tavily API 錯誤時回傳空清單且不拋出例外（需求 3.5）"""
    with patch("agents.search.get_search_tool") as mock_get_tool:
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = Exception("Tavily API error")
        mock_get_tool.return_value = mock_tool

        state = _make_state(["failing task"])
        result = search_node(state)

    assert result == {"search_results": []}, "API 錯誤時應回傳空清單"


def test_search_node_partial_failure_continues():
    """驗證部分子任務失敗時，其他子任務的結果仍被保留"""
    mock_results_ok = [
        {"title": "OK", "url": "https://ok.com", "content": "ok", "score": 0.8},
    ]

    with patch("agents.search.get_search_tool") as mock_get_tool:
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = [
            Exception("first task fails"),
            mock_results_ok,
        ]
        mock_get_tool.return_value = mock_tool

        state = _make_state(["fail task", "ok task"])
        result = search_node(state)

    assert len(result["search_results"]) == 1
    assert result["search_results"][0]["url"] == "https://ok.com"


def test_search_node_empty_task_list():
    """驗證 task_list 為空時回傳空清單"""
    with patch("agents.search.get_search_tool") as mock_get_tool:
        mock_tool = MagicMock()
        mock_get_tool.return_value = mock_tool

        state = _make_state([])
        result = search_node(state)

    assert result == {"search_results": []}
    mock_tool.invoke.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 3.2 — Property test 屬性 5: 搜尋結果 URL 唯一性
# Feature: multi-agent-research-pipeline, Property 5: 搜尋結果 URL 唯一性
# Validates: Requirements 3.3, 3.4
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.lists(
    st.fixed_dictionaries({
        "title": st.text(min_size=1),
        "url": st.sampled_from(["https://a.com", "https://b.com", "https://c.com"]),
        "content": st.text(),
        "score": st.floats(min_value=0.0, max_value=1.0),
    })
))
def test_search_results_url_uniqueness(raw_results):
    """
    **Validates: Requirements 3.3, 3.4**

    屬性 5：對於任意包含重複 URL 的搜尋結果清單，
    去重後每個 URL 只出現一次，且每個元素包含 title, url, content, score 四個欄位。
    """
    deduped = deduplicate_results(raw_results)

    # 每個 URL 只出現一次
    urls = [item["url"] for item in deduped]
    assert len(urls) == len(set(urls)), "去重後仍有重複 URL"

    # 每個元素包含四個必要欄位
    for item in deduped:
        assert "title" in item
        assert "url" in item
        assert "content" in item
        assert "score" in item

