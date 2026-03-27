"""
Conditional edge 路由邏輯
Task 6: 實作 should_revise 路由函式
"""

import json
import logging

from graph.state import AgentState

logger = logging.getLogger(__name__)


def should_revise(state: AgentState) -> str:
    """
    根據 critique 結果決定路由目標。

    路由邏輯：
    - critique_count >= 2 → "writer"（安全閥）
    - critique.passed == True → "writer"
    - critique.passed == False 且 critique_count < 2 → "search"

    注意：此函式只回傳路由字串，不修改 state。
    critique_count 的遞增由 critic_node 負責。

    需求：5.2, 5.3, 5.4
    """
    critique_count: int = state.get("critique_count", 0)

    # 安全閥：已達上限，強制進入 writer
    if critique_count >= 2:
        return "writer"

    critique_str: str = state.get("critique", "{}")
    try:
        critique_data = json.loads(critique_str)
        passed = bool(critique_data.get("passed", True))
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("should_revise JSON 解析失敗，預設 passed=True：%s", e)
        passed = True

    if passed:
        return "writer"
    else:
        return "search"
