"""
Tests for Critic Agent and routing logic (Task 6.1, 6.2, 6.3).

Sub-tasks:
  6.1 — Property test (屬性 7): Critic 輸出結構完整性
  6.2 — Property test (屬性 8): Critic 路由邏輯正確性
  6.3 — Property test (屬性 9): critique_count 遞增不變量
"""

import json
import sys
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from agents.critic import critic_node  # noqa: E402
from graph.edges import should_revise  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _make_state(
    user_query: str = "測試問題",
    analysis: str = "測試分析",
    critique: str = "",
    critique_count: int = 0,
) -> dict:
    return {
        "user_query": user_query,
        "task_list": [],
        "current_task": "",
        "search_results": [],
        "analysis": analysis,
        "critique": critique,
        "critique_count": critique_count,
        "final_report": "",
        "messages": [],
    }


def _mock_llm_with_json(passed: bool, score: int = 8, feedback: str = "良好", missing: list = None):
    """建立回傳有效 CriticOutput JSON 的 mock LLM。"""
    if missing is None:
        missing = []
    output = json.dumps({
        "passed": passed,
        "score": score,
        "feedback": feedback,
        "missing_aspects": missing,
    }, ensure_ascii=False)
    mock_response = MagicMock()
    mock_response.content = output
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 6.1 — Property test 屬性 7: Critic 輸出結構完整性
# Feature: multi-agent-research-pipeline, Property 7: Critic 輸出結構完整性
# Validates: Requirements 5.1
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.text(min_size=1))
def test_critic_output_structure(analysis_text):
    """
    **Validates: Requirements 5.1**

    屬性 7：對於任意 analysis 輸入，Critic 節點的輸出應包含
    passed（bool）、score（1-10 的整數）、feedback（str）、
    missing_aspects（list）四個欄位，且型別正確。
    """
    # mock LLM 回傳有效 CriticOutput JSON
    mock_llm_instance = _mock_llm_with_json(
        passed=True,
        score=7,
        feedback="分析完整",
        missing=[],
    )

    state = _make_state(analysis=analysis_text)

    with patch("agents.critic.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        result = critic_node(state)

    # 驗證回傳包含 critique 欄位
    assert "critique" in result
    assert isinstance(result["critique"], str)

    # 解析 critique JSON
    critique_data = json.loads(result["critique"])

    # 驗證四個必要欄位存在且型別正確
    assert "passed" in critique_data
    assert isinstance(critique_data["passed"], bool)

    assert "score" in critique_data
    assert isinstance(critique_data["score"], int)
    assert 1 <= critique_data["score"] <= 10

    assert "feedback" in critique_data
    assert isinstance(critique_data["feedback"], str)

    assert "missing_aspects" in critique_data
    assert isinstance(critique_data["missing_aspects"], list)


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 6.2 — Property test 屬性 8: Critic 路由邏輯正確性
# Feature: multi-agent-research-pipeline, Property 8: Critic 路由邏輯正確性
# Validates: Requirements 5.2, 5.3, 5.4
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.booleans(), st.integers(min_value=0, max_value=5))
def test_critic_routing_logic(passed, critique_count):
    """
    **Validates: Requirements 5.2, 5.3, 5.4**

    屬性 8：對於任意 AgentState，should_revise 函式的路由結果應滿足：
    - passed=False 且 critique_count < 2 → "search"
    - passed=True → "writer"
    - critique_count >= 2 → "writer"（無論 passed）
    """
    critique_json = json.dumps({
        "passed": passed,
        "score": 7,
        "feedback": "測試",
        "missing_aspects": [],
    })
    state = _make_state(critique=critique_json, critique_count=critique_count)

    result = should_revise(state)

    # 驗證三種路由情境
    if critique_count >= 2:
        # 安全閥：無論 passed，強制進入 writer
        assert result == "writer", (
            f"critique_count={critique_count} >= 2 應路由至 writer，但得到 {result!r}"
        )
    elif passed:
        # passed=True → writer
        assert result == "writer", (
            f"passed=True 應路由至 writer，但得到 {result!r}"
        )
    else:
        # passed=False 且 count < 2 → search
        assert result == "search", (
            f"passed=False 且 critique_count={critique_count} < 2 應路由至 search，但得到 {result!r}"
        )


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 6.3 — Property test 屬性 9: critique_count 遞增不變量
# Feature: multi-agent-research-pipeline, Property 9: critique_count 遞增不變量
# Validates: Requirements 5.5
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=1))
def test_critique_count_increments(initial_count):
    """
    **Validates: Requirements 5.5**

    屬性 9：對於任意觸發補充搜尋的 Pipeline 執行，
    每次路由回 Search Agent 時，AgentState.critique_count 應比前一次增加 1。

    設計說明：
    - initial_count=0：passed=False，count 從 0 → 1，should_revise 看到 count=1 → "search"
    - initial_count=1：passed=False，count 從 1 → 2，should_revise 看到 count=2 → "writer"（安全閥）
    兩種情況下，critic_node 都應將 critique_count 遞增 1。
    """
    # 設定 passed=False 以觸發計數器遞增
    mock_llm_instance = _mock_llm_with_json(
        passed=False,
        score=3,
        feedback="分析不完整",
        missing=["缺少費用資訊"],
    )

    state = _make_state(critique_count=initial_count)

    with patch("agents.critic.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        result = critic_node(state)

    # 核心不變量：passed=False 且 initial_count < 2 時，critique_count 應遞增 1
    assert result["critique_count"] == initial_count + 1, (
        f"critique_count 應從 {initial_count} 遞增至 {initial_count + 1}，"
        f"但得到 {result['critique_count']}"
    )

    # 驗證路由決策符合安全閥邏輯
    new_count = result["critique_count"]
    route = should_revise({**state, "critique": result["critique"], "critique_count": new_count})

    if new_count >= 2:
        # 安全閥：遞增後達到上限，強制進入 writer
        assert route == "writer", (
            f"new_count={new_count} >= 2 應路由至 writer，但得到 {route!r}"
        )
    else:
        # 尚未達上限，繼續路由至 search
        assert route == "search", (
            f"new_count={new_count} < 2 且 passed=False，應路由至 search，但得到 {route!r}"
        )

