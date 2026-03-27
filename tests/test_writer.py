"""
Tests for Writer Agent (Task 7.1).

Sub-tasks:
  7.1 — Property test (屬性 10): Writer 報告結構完整性
"""

import re
import sys
import pathlib
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
import hypothesis.strategies as st

ROOT = pathlib.Path(__file__).parent.parent  # project root

from agents.writer import writer_node  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

_COMPLETE_REPORT = """\
## 概覽
這是一份關於研究主題的摘要。

## 主要內容
詳細的研究內容與分析結果。

## 實用資訊
- 預算：約 NT$5,000
- 最佳時間：春季
- 注意事項：提前預訂

## 參考來源
- [範例來源](https://example.com/article)
- [另一來源](https://example.org/page)
"""


def _make_state(analysis: str = "測試分析", user_query: str = "測試問題") -> dict:
    return {
        "user_query": user_query,
        "task_list": [],
        "current_task": "",
        "search_results": [],
        "analysis": analysis,
        "critique": "",
        "critique_count": 0,
        "final_report": "",
        "messages": [],
    }


def _mock_llm_with_report(report: str):
    """建立回傳完整 Markdown 報告的 mock LLM。"""
    mock_response = MagicMock()
    mock_response.content = report
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 7.1 — Property test 屬性 10: Writer 報告結構完整性
# Feature: multi-agent-research-pipeline, Property 10: Writer 報告結構完整性
# Validates: Requirements 6.1, 6.2, 6.3, 6.4
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.text(min_size=1))
def test_writer_report_structure(analysis_text):
    """
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

    屬性 10：對於任意非空的 analysis 輸入，Writer 節點輸出的 final_report 應：
    1. 為有效的 Markdown 字串（非空）
    2. 包含概覽、主要內容、實用資訊、參考來源等區塊標題
    3. 在參考來源區塊中包含至少一個 [文字](URL) 格式的 Markdown 連結
    """
    mock_llm_instance = _mock_llm_with_report(_COMPLETE_REPORT)
    state = _make_state(analysis=analysis_text)

    with patch("agents.writer.ChatGoogleGenerativeAI", return_value=mock_llm_instance):
        result = writer_node(state)

    # 驗證回傳包含 final_report 欄位
    assert "final_report" in result

    report = result["final_report"]

    # 1. final_report 非空
    assert isinstance(report, str)
    assert len(report.strip()) > 0

    # 2. 包含必要區塊標題
    assert "## 概覽" in report, "報告應包含 ## 概覽 區塊"
    assert "## 主要內容" in report, "報告應包含 ## 主要內容 區塊"
    assert "## 實用資訊" in report, "報告應包含 ## 實用資訊 區塊"
    assert "## 參考來源" in report, "報告應包含 ## 參考來源 區塊"

    # 3. 參考來源包含至少一個 [文字](URL) 格式的 Markdown 連結
    markdown_link_pattern = re.compile(r'\[.+?\]\(https?://[^\)]+\)')
    assert markdown_link_pattern.search(report), (
        "報告的參考來源區塊應包含至少一個 [文字](URL) 格式的 Markdown 連結"
    )

