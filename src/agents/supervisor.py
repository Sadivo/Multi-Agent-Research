"""
Supervisor Agent — 任務分解
Task 4: 實作 supervisor_node

需求：2.1, 2.2, 2.3, 2.4
"""

import json
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import AgentState

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """你是一個研究任務分解專家。請將以下研究問題分解為 2 到 4 個具體的搜尋子任務。
必須以純 JSON 格式回應，不得包含任何 Markdown 標記或程式碼區塊。
格式：{{"tasks": ["子任務1", "子任務2"]}}

研究問題：{user_query}"""


def supervisor_node(state: AgentState) -> dict:
    """
    接收使用者研究問題，分解為 2-4 個具體搜尋子任務。

    需求：
    - 2.1: 分解為 2-4 個子任務
    - 2.2: 輸出純 JSON，不含 Markdown 標記
    - 2.4: JSON 解析失敗時以原始問題作為單一子任務繼續
    """
    user_query = state["user_query"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    prompt = SUPERVISOR_PROMPT.format(user_query=user_query)

    response = llm.invoke(prompt)
    raw_content = response.content

    # 移除 Gemini 可能回傳的 Markdown code fence
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        task_list = parsed["tasks"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            "Supervisor JSON 解析失敗（%s），以原始問題作為單一子任務繼續。原始輸出：%r",
            e,
            raw_content,
        )
        task_list = [user_query]

    return {
        "task_list": task_list,
        "current_task": task_list[0],
    }
