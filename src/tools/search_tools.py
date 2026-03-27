"""
Tavily 搜尋工具包裝
Task 3: 包裝 TavilySearchResults
"""

from langchain_community.tools.tavily_search import TavilySearchResults


def get_search_tool() -> TavilySearchResults:
    """回傳已設定的 Tavily 搜尋工具實例。"""
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
    )
