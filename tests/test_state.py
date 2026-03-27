"""
Tests for graph/state.py — AgentState, SearchResult, CriticOutput.

Sub-tasks 2.1 & 2.2:
  - Unit tests: field completeness for all three TypedDicts
  - Property test (屬性 2): missing required field triggers KeyError
"""

import sys
import pathlib
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from graph.state import (  # noqa: E402
    AgentState,
    SearchResult,
    CriticOutput,
    REQUIRED_AGENT_STATE_FIELDS,
    validate_state,
)


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 2.1 — Unit tests: AgentState 欄位完整性
# ══════════════════════════════════════════════════════════════════════════

def test_agent_state_contains_all_required_fields():
    """驗證 AgentState TypedDict 包含所有必要欄位（需求 1.2）"""
    expected_fields = {
        "user_query",
        "task_list",
        "current_task",
        "search_results",
        "analysis",
        "critique",
        "critique_count",
        "final_report",
        "messages",
    }
    actual_fields = set(AgentState.__annotations__.keys())
    assert expected_fields == actual_fields, (
        f"AgentState 欄位不符。缺少：{expected_fields - actual_fields}，"
        f"多餘：{actual_fields - expected_fields}"
    )


def test_search_result_contains_all_required_fields():
    """驗證 SearchResult TypedDict 包含 title, url, content, score 欄位"""
    expected_fields = {"title", "url", "content", "score"}
    actual_fields = set(SearchResult.__annotations__.keys())
    assert expected_fields == actual_fields, (
        f"SearchResult 欄位不符。缺少：{expected_fields - actual_fields}"
    )


def test_critic_output_contains_all_required_fields():
    """驗證 CriticOutput TypedDict 包含 passed, score, feedback, missing_aspects 欄位"""
    expected_fields = {"passed", "score", "feedback", "missing_aspects"}
    actual_fields = set(CriticOutput.__annotations__.keys())
    assert expected_fields == actual_fields, (
        f"CriticOutput 欄位不符。缺少：{expected_fields - actual_fields}"
    )


def test_validate_state_passes_with_all_fields():
    """驗證 validate_state 在所有欄位存在時不拋出例外"""
    complete_state = {field: None for field in REQUIRED_AGENT_STATE_FIELDS}
    # Should not raise
    validate_state(complete_state)


def test_validate_state_raises_key_error_on_missing_field():
    """驗證 validate_state 在缺少欄位時拋出 KeyError"""
    incomplete_state = {field: None for field in REQUIRED_AGENT_STATE_FIELDS}
    del incomplete_state["user_query"]
    with pytest.raises(KeyError):
        validate_state(incomplete_state)


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 2.2 — Property test 屬性 2: 缺失欄位觸發錯誤
# Feature: multi-agent-research-pipeline, Property 2: 缺失欄位觸發錯誤
# Validates: Requirements 1.4
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.sampled_from(REQUIRED_AGENT_STATE_FIELDS))
def test_missing_field_triggers_error(missing_field):
    """
    **Validates: Requirements 1.4**

    屬性 2：對於任意缺少一個必填欄位的 AgentState 字典，
    validate_state() 應拋出 KeyError，不得靜默繼續執行。
    """
    # Build a complete state dict, then remove one field
    state = {field: None for field in REQUIRED_AGENT_STATE_FIELDS}
    del state[missing_field]

    with pytest.raises((TypeError, KeyError)):
        validate_state(state)

