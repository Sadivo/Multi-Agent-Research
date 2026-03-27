"""
LangGraph StateGraph 組裝
Task 8: 實作 build_graph() 函式

需求：7.1, 7.2, 7.3, 7.4
"""

import logging

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.edges import should_revise
from agents.supervisor import supervisor_node
from agents.search import search_node
from agents.analyst import analyst_node
from agents.critic import critic_node
from agents.writer import writer_node

logger = logging.getLogger(__name__)


def build_graph():
    """
    組裝並編譯 LangGraph StateGraph。

    節點順序：supervisor → search → analyst → critic → writer
    critic 後設置 conditional edge，根據 should_revise 路由至 search 或 writer。

    需求：
    - 7.1: 使用 StateGraph 組裝所有 Agent 節點
    - 7.2: critic 後設置 Conditional_Edge
    - 7.3: 回傳已編譯的 StateGraph 實例
    - 7.4: 組裝失敗時拋出明確錯誤訊息
    """
    try:
        graph = StateGraph(AgentState)

        graph.add_node("supervisor", supervisor_node)
        graph.add_node("search", search_node)
        graph.add_node("analyst", analyst_node)
        graph.add_node("critic", critic_node)
        graph.add_node("writer", writer_node)

        graph.set_entry_point("supervisor")
        graph.add_edge("supervisor", "search")
        graph.add_edge("search", "analyst")
        graph.add_edge("analyst", "critic")
        graph.add_conditional_edges(
            "critic",
            should_revise,
            {"search": "search", "writer": "writer"},
        )
        graph.add_edge("writer", END)

        return graph.compile()

    except Exception as e:
        raise RuntimeError(
            f"StateGraph 組裝失敗：{e}"
        ) from e
