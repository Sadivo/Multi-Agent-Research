"""
Search Agent — 網路搜尋
Task 3: 實作 search_node，對 task_list 中每個子任務執行搜尋並去重
Task 10: 改為 async def，使用 asyncio.gather 平行搜尋
"""

import asyncio
import logging
from typing import List

from graph.state import AgentState
from tools.search_tools import get_search_tool

logger = logging.getLogger(__name__)


def deduplicate_results(results: list) -> list:
    """
    對搜尋結果清單進行 URL 去重。
    保留每個 URL 第一次出現的結果，確保輸出中每個 URL 只出現一次。
    """
    seen_urls: set = set()
    deduped: list = []
    for item in results:
        url = item.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(item)
    return deduped


def search_node(state: AgentState) -> dict:
    """
    對 state["task_list"] 中的每個子任務平行執行 Tavily 搜尋，
    去重後將結果寫入 search_results。

    使用 asyncio.gather 平行發出所有搜尋請求（需求 3.1）。
    TavilySearchResults 為同步 API，以 asyncio.to_thread 包裝。

    需求：3.1, 3.2, 3.3, 3.4, 3.5
    """
    task_list: List[str] = state.get("task_list", [])
    search_tool = get_search_tool()

    async def _run_parallel() -> list:
        async def search_one(task: str) -> list:
            try:
                raw = await asyncio.to_thread(search_tool.invoke, task)
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                    }
                    for r in raw
                ]
            except Exception as e:
                logger.error("Tavily API 錯誤（任務：%s）：%s", task, e)
                return []

        results_per_task = await asyncio.gather(*[search_one(t) for t in task_list])
        return [item for sublist in results_per_task for item in sublist]

    all_results = asyncio.run(_run_parallel())
    return {"search_results": deduplicate_results(all_results)}