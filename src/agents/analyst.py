"""
Analyst Agent — 資訊整合分析
Task 5: 實作 analyst_node，根據搜尋結果產出 grounded 分析
"""

import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState

logger = logging.getLogger(__name__)

_EMPTY_RESULTS_MESSAGE = (
    "無法取得搜尋結果，無法進行分析。請確認 Tavily API 設定是否正確。"
)

_PROMPT_TEMPLATE = """\
你是一個嚴謹的研究分析師。請根據以下搜尋結果進行分析，嚴格禁止使用你自身的訓練知識。
所有資訊必須來自提供的搜尋結果，並以 [來源標題](URL) 格式引用來源。
若某方面資訊不足，請標注「資訊不足：[說明]」。

研究問題：{user_query}

搜尋結果：
{formatted_results}

請提供詳細分析："""


def _format_results(search_results: List[dict]) -> str:
    """將搜尋結果格式化為 prompt 可用的文字。"""
    lines = []
    for i, r in enumerate(search_results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"{i}. [{title}]({url})\n   {content}")
    return "\n\n".join(lines)


def analyst_node(state: AgentState) -> dict:
    """
    根據 state["search_results"] 產出分析。

    - 若 search_results 為空，直接回傳說明文字（不呼叫 LLM）。
    - 否則使用 grounding prompt 呼叫 LLM，要求引用來源 URL。

    需求：4.1, 4.2, 4.3, 4.4
    """
    search_results: List[dict] = state.get("search_results", [])

    if not search_results:
        logger.warning("search_results 為空，跳過 LLM 呼叫。")
        return {"analysis": _EMPTY_RESULTS_MESSAGE}

    user_query: str = state.get("user_query", "")
    formatted_results = _format_results(search_results)
    prompt = _PROMPT_TEMPLATE.format(
        user_query=user_query,
        formatted_results=formatted_results,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    response = llm.invoke(prompt)
    analysis = response.content

    return {"analysis": analysis}
