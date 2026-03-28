"""
Tavily 搜尋工具包裝
"""

from langchain_tavily import TavilySearch


def get_search_tool() -> TavilySearch:
    """回傳已設定的 Tavily 搜尋工具實例。"""
    return TavilySearch(
        max_results=5,
        search_depth="advanced",
    )
