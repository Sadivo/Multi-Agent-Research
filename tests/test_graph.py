"""
Tests for LangGraph StateGraph assembly (Task 8.1, 8.2).

Sub-tasks:
  8.1 — Unit tests: build_graph() 回傳已編譯的 StateGraph 實例
  8.2 — Property test (屬性 1): Agent 節點狀態不變量
"""

import json
import sys
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from graph.graph import build_graph  # noqa: E402
from agents.supervisor import supervisor_node  # noqa: E402
from agents.search import search_node  # noqa: E402
from agents.analyst import analyst_node  # noqa: E402
from agents.critic import critic_node  # noqa: E402
from agents.writer import writer_node  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _make_full_state(
    user_query: str = "測試問題",
    task_list=None,
    current_task: str = "子任務1",
    search_results=None,
    analysis: str = "測試分析",
    critique: str = "",
    critique_count: int = 0,
    final_report: str = "",
    messages=None,
) -> dict:
    return {
        "user_query": user_query,
        "task_list": task_list if task_list is not None else ["子任務1"],
        "current_task": current_task,
        "search_results": search_results if search_results is not None else [],
        "analysis": analysis,
        "critique": critique,
        "critique_count": critique_count,
        "final_report": final_report,
        "messages": messages if messages is not None else [],
    }


def _mock_llm_response(content: str):
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 8.1 — Unit tests: build_graph()
# Validates: Requirements 7.3
# ══════════════════════════════════════════════════════════════════════════

def test_build_graph_returns_non_none():
    """驗證 build_graph() 回傳非 None 的已編譯 StateGraph 實例（需求 7.3）"""
    compiled = build_graph()
    assert compiled is not None


def test_build_graph_returns_invokable():
    """驗證 build_graph() 回傳的物件有 invoke 方法（是已編譯的 graph）"""
    compiled = build_graph()
    assert hasattr(compiled, "invoke"), "build_graph() 回傳的物件缺少 invoke 方法"
    assert callable(compiled.invoke)


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 8.2 — Property test 屬性 1: Agent 節點狀態不變量
# Feature: multi-agent-research-pipeline, Property 1: Agent 節點狀態不變量
# Validates: Requirements 1.3
# ══════════════════════════════════════════════════════════════════════════

# 各節點只應更新的欄位
_NODE_UPDATED_FIELDS = {
    "supervisor": {"task_list", "current_task"},
    "search": {"search_results"},
    "analyst": {"analysis"},
    "critic": {"critique", "critique_count"},
    "writer": {"final_report"},
}

# 各節點不應修改的欄位（AgentState 所有欄位 - 該節點更新的欄位）
_ALL_STATE_FIELDS = {
    "user_query", "task_list", "current_task", "search_results",
    "analysis", "critique", "critique_count", "final_report", "messages",
}


@settings(max_examples=100)
@given(
    st.fixed_dictionaries({
        "user_query": st.text(min_size=1, max_size=50),
        "task_list": st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=4),
        "current_task": st.text(min_size=1, max_size=20),
        "search_results": st.lists(
            st.fixed_dictionaries({
                "title": st.text(max_size=30),
                "url": st.text(min_size=5, max_size=50).map(lambda s: f"https://{s}"),
                "content": st.text(max_size=100),
                "score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            }),
            max_size=3,
        ),
        "analysis": st.text(min_size=1, max_size=100),
        "critique": st.just(""),
        "critique_count": st.integers(min_value=0, max_value=1),
        "final_report": st.just(""),
        "messages": st.just([]),
    })
)
def test_agent_node_state_invariant(initial_state):
    """
    **Validates: Requirements 1.3**

    屬性 1：對於任意 AgentState 初始值與任意 Agent 節點，
    執行該節點後，節點回傳的 dict 中不應包含不應被修改的欄位。

    每個節點只回傳它更新的欄位（dict），不回傳完整 AgentState。
    測試驗證：節點回傳的 dict 中不包含不應被修改的欄位。
    """
    # ── supervisor_node ────────────────────────────────────────────────
    tasks = ["子任務1", "子任務2"]
    supervisor_json = json.dumps({"tasks": tasks}, ensure_ascii=False)
    mock_supervisor_llm = _mock_llm_response(supervisor_json)

    with patch("agents.supervisor.ChatGoogleGenerativeAI", return_value=mock_supervisor_llm):
        supervisor_result = supervisor_node(initial_state)

    supervisor_unexpected = (
        set(supervisor_result.keys()) - _NODE_UPDATED_FIELDS["supervisor"]
    )
    assert not supervisor_unexpected, (
        f"supervisor_node 回傳了不應修改的欄位：{supervisor_unexpected}"
    )

    # ── search_node ────────────────────────────────────────────────────
    import asyncio as _asyncio
    mock_search_tool = MagicMock()
    mock_search_tool.invoke.return_value = []

    with patch("agents.search.get_search_tool", return_value=mock_search_tool):
        search_result = search_node(initial_state)

    search_unexpected = (
        set(search_result.keys()) - _NODE_UPDATED_FIELDS["search"]
    )
    assert not search_unexpected, (
        f"search_node 回傳了不應修改的欄位：{search_unexpected}"
    )

    # ── analyst_node ───────────────────────────────────────────────────
    analyst_json = "根據搜尋結果，分析如下：https://example.com"
    mock_analyst_llm = _mock_llm_response(analyst_json)

    with patch("agents.analyst.ChatGoogleGenerativeAI", return_value=mock_analyst_llm):
        analyst_result = analyst_node(initial_state)

    analyst_unexpected = (
        set(analyst_result.keys()) - _NODE_UPDATED_FIELDS["analyst"]
    )
    assert not analyst_unexpected, (
        f"analyst_node 回傳了不應修改的欄位：{analyst_unexpected}"
    )

    # ── critic_node ────────────────────────────────────────────────────
    critic_json = json.dumps({
        "passed": True,
        "score": 8,
        "feedback": "良好",
        "missing_aspects": [],
    }, ensure_ascii=False)
    mock_critic_llm = _mock_llm_response(critic_json)

    with patch("agents.critic.ChatGoogleGenerativeAI", return_value=mock_critic_llm):
        critic_result = critic_node(initial_state)

    critic_unexpected = (
        set(critic_result.keys()) - _NODE_UPDATED_FIELDS["critic"]
    )
    assert not critic_unexpected, (
        f"critic_node 回傳了不應修改的欄位：{critic_unexpected}"
    )

    # ── writer_node ────────────────────────────────────────────────────
    writer_output = (
        "## 概覽\n摘要\n\n## 主要內容\n詳細\n\n"
        "## 實用資訊\n資訊\n\n## 參考來源\n[來源](https://example.com)"
    )
    mock_writer_llm = _mock_llm_response(writer_output)

    with patch("agents.writer.ChatGoogleGenerativeAI", return_value=mock_writer_llm):
        writer_result = writer_node(initial_state)

    writer_unexpected = (
        set(writer_result.keys()) - _NODE_UPDATED_FIELDS["writer"]
    )
    assert not writer_unexpected, (
        f"writer_node 回傳了不應修改的欄位：{writer_unexpected}"
    )

