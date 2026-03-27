"""
Critic Agent — 品質把關
Task 6: 實作 critic_node，評估分析品質並輸出結構化反饋
"""

import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
你是一個嚴格的研究品質審核員。請評估以下研究分析的品質，並以純 JSON 格式輸出評估結果。

研究問題：{user_query}

分析內容：
{analysis}

請輸出以下格式的純 JSON（不含任何 Markdown 標記或額外說明）：
{{"passed": true/false, "score": 1-10, "feedback": "評估說明", "missing_aspects": ["缺少的面向1", "缺少的面向2"]}}

評估標準：
- passed=true：分析完整、有來源引用、涵蓋主要面向
- passed=false：分析不完整、缺少重要資訊或來源引用不足
- score：1-10 分，10 分為最高品質
- feedback：若 passed=false，說明缺少什麼
- missing_aspects：需要補充搜尋的具體面向清單"""


def critic_node(state: AgentState) -> dict:
    """
    評估 analysis 品質，輸出 CriticOutput 結構化反饋。

    - 使用 gpt-4o-mini 評估分析品質
    - 強制輸出純 JSON 格式
    - JSON 解析失敗時 fallback：passed=True, score=5
    - 若 passed=False 且 critique_count < 2，遞增 critique_count

    需求：5.1, 5.2, 5.3, 5.4, 5.5
    """
    user_query: str = state.get("user_query", "")
    analysis: str = state.get("analysis", "")
    critique_count: int = state.get("critique_count", 0)

    prompt = _PROMPT_TEMPLATE.format(
        user_query=user_query,
        analysis=analysis,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    response = llm.invoke(prompt)
    raw_output = response.content

    try:
        critique_data = json.loads(raw_output)
        passed = bool(critique_data.get("passed", True))
        score = int(critique_data.get("score", 5))
        feedback = str(critique_data.get("feedback", ""))
        missing_aspects = list(critique_data.get("missing_aspects", []))
        critique_json = json.dumps({
            "passed": passed,
            "score": score,
            "feedback": feedback,
            "missing_aspects": missing_aspects,
        }, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Critic JSON 解析失敗，使用 fallback：%s", e)
        passed = True
        critique_json = json.dumps({
            "passed": True,
            "score": 5,
            "feedback": "解析失敗",
            "missing_aspects": [],
        }, ensure_ascii=False)

    # 若 passed=False 且 critique_count < 2，遞增計數器
    new_count = critique_count
    if not passed and critique_count < 2:
        new_count = critique_count + 1

    return {
        "critique": critique_json,
        "critique_count": new_count,
    }
