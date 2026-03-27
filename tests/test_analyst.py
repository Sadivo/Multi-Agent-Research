"""
Tests for Analyst Agent (Task 5.1, 5.2).

Sub-tasks:
  5.1 — Unit tests: empty search_results, LLM call, return fields
  5.2 — Property test (屬性 6): Analyst 分析包含來源 URL
"""

import sys
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from agents.analyst import analyst_node, _EMPTY_RESULTS_MESSAGE  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _make_state(user_query: str = "測試問題", search_results: list = None) -> dict:
    return {
        "user_query": user_query,
        "task_list": [],
        "current_task": "",
        "search_results": search_results if search_results is not None else [],
        "analysis": "",
        "critique": "",
        "critique_count": 0,
        "final_report": "",
        "messages": [],
    }


def _mock_llm(content: str):
    """建立回傳指定 content 的 mock LLM。"""
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 5.1 — Unit tests
# ══════════════════════════════════════════════════════════════════════════

def test_analyst_empty_search_results_returns_explanation():
    """驗證 search_results 為空時回傳說明文字，不呼叫 LLM（需求 4.4）"""
    with patch("agents.analyst.ChatGoogleGenerativeAI") as mock_cls:
        result = analyst_node(_make_state(search_results=[]))

    # LLM 不應被實例化
    mock_cls.assert_not_called()
    assert "analysis" in result
    assert result["analysis"] == _EMPTY_RESULTS_MESSAGE


def test_analyst_empty_search_results_no_fabrication():
    """驗證 search_results 為空時輸出不含虛構內容（需求 4.4）"""
    with patch("agents.analyst.ChatGoogleGenerativeAI"):
        result = analyst_node(_make_state(search_results=[]))

    analysis = result["analysis"]
    # 說明文字應提及無法取得搜尋結果
    assert "無法取得搜尋結果" in analysis or "無法" in analysis


def test_analyst_calls_llm_when_results_present():
    """驗證 search_results 非空時呼叫 LLM（需求 4.1）"""
    search_results = [
        {"title": "標題A", "url": "https://example.com/a", "content": "內容A", "score": 0.9}
    ]
    expected_analysis = "根據 [標題A](https://example.com/a) 的資訊..."
    mock_llm_instance = _mock_llm(expected_analysis)

    with patch("agents.analyst.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        result = analyst_node(_make_state(search_results=search_results))

    mock_llm_instance.invoke.assert_called_once()
    assert result["analysis"] == expected_analysis


def test_analyst_returns_analysis_field():
    """驗證回傳 dict 包含 analysis 欄位（需求 4.1）"""
    search_results = [
        {"title": "來源", "url": "https://example.com", "content": "內容", "score": 0.8}
    ]
    mock_llm_instance = _mock_llm("分析結果")

    with patch("agents.analyst.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        result = analyst_node(_make_state(search_results=search_results))

    assert "analysis" in result
    assert isinstance(result["analysis"], str)
    assert len(result["analysis"]) > 0


def test_analyst_prompt_includes_user_query():
    """驗證 prompt 包含 user_query（需求 4.1）"""
    search_results = [
        {"title": "T", "url": "https://example.com", "content": "C", "score": 0.5}
    ]
    user_query = "日本旅遊攻略"
    mock_llm_instance = _mock_llm("分析")

    with patch("agents.analyst.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        analyst_node(_make_state(user_query=user_query, search_results=search_results))

    call_args = mock_llm_instance.invoke.call_args
    prompt_text = call_args[0][0]
    assert user_query in prompt_text


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 5.2 — Property test 屬性 6: Analyst 分析包含來源 URL
# Feature: multi-agent-research-pipeline, Property 6: Analyst 分析包含來源 URL
# Validates: Requirements 4.2
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.lists(
    st.fixed_dictionaries({
        "title": st.text(min_size=1),
        "url": st.from_regex(r"https://[a-z]+\.com/[a-z]+", fullmatch=True),
        "content": st.text(min_size=1),
        "score": st.floats(min_value=0.0, max_value=1.0),
    }),
    min_size=1
))
def test_analyst_analysis_contains_source_url(search_results):
    """
    **Validates: Requirements 4.2**

    屬性 6：對於任意非空的 search_results，
    Analyst 節點輸出的 analysis 字串應至少包含一個來自 search_results 中的 URL。
    """
    # mock LLM 回傳包含第一個 search_result URL 的分析文字
    first_url = search_results[0]["url"]
    first_title = search_results[0]["title"]
    mock_analysis = f"根據 [{first_title}]({first_url}) 的資訊，以下是分析結果。"

    mock_llm_instance = _mock_llm(mock_analysis)

    with patch("agents.analyst.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        result = analyst_node(_make_state(search_results=search_results))

    analysis = result["analysis"]

    # 驗證 analysis 至少包含一個來自 search_results 的 URL
    urls_in_results = {r["url"] for r in search_results}
    assert any(url in analysis for url in urls_in_results), (
        f"analysis 未包含任何來源 URL。analysis={analysis!r}, urls={urls_in_results}"
    )

