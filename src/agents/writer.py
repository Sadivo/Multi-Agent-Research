"""
Writer Agent — 報告產出
Task 7: 實作 writer_node，將分析整理為結構化 Markdown 報告

需求：6.1, 6.2, 6.3, 6.4
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
你是一個專業的研究報告撰寫員。請將以下分析整理為結構清晰的 Markdown 報告。

研究問題：{user_query}

分析內容：
{analysis}

請輸出包含以下區塊的 Markdown 報告：
1. ## 概覽 — 簡短摘要
2. ## 主要內容 — 詳細資訊
3. ## 實用資訊 — 預算、時間、注意事項等
4. ## 參考來源 — 以 [來源標題](URL) 格式列出所有引用的 URL

報告：
"""


def writer_node(state: AgentState) -> dict:
    """
    將 state["analysis"] 整理為 Markdown 格式的最終報告。

    - 使用 grounding prompt 呼叫 LLM，要求輸出包含概覽、主要內容、實用資訊、參考來源區塊。
    - 回傳 {"final_report": "..."}

    需求：6.1, 6.2, 6.3, 6.4
    """
    user_query: str = state.get("user_query", "")
    analysis: str = state.get("analysis", "")

    prompt = _PROMPT_TEMPLATE.format(
        user_query=user_query,
        analysis=analysis,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    response = llm.invoke(prompt)
    final_report = response.content

    return {"final_report": final_report}
