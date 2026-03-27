"""
AgentState TypedDict 定義與核心資料模型
Task 2: 定義 AgentState 與核心資料模型
"""

from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    user_query: str
    task_list: List[str]
    current_task: str
    search_results: List[dict]
    analysis: str
    critique: str
    critique_count: int
    final_report: str
    messages: Annotated[list, add_messages]


class SearchResult(TypedDict):
    title: str
    url: str
    content: str
    score: float


class CriticOutput(TypedDict):
    passed: bool
    score: int
    feedback: str
    missing_aspects: List[str]


# 必填欄位清單（用於驗證）
REQUIRED_AGENT_STATE_FIELDS = [
    "user_query",
    "task_list",
    "current_task",
    "search_results",
    "analysis",
    "critique",
    "critique_count",
    "final_report",
    "messages",
]


def validate_state(state: dict) -> None:
    """
    驗證 AgentState 字典是否包含所有必填欄位。
    若缺少任一必填欄位，拋出 KeyError。

    需求：1.4 — IF AgentState 中任一必填欄位缺失，
    THEN THE Pipeline SHALL 在啟動時拋出型別錯誤並終止執行。
    """
    missing = [f for f in REQUIRED_AGENT_STATE_FIELDS if f not in state]
    if missing:
        raise KeyError(
            f"AgentState 缺少必填欄位：{', '.join(missing)}"
        )
