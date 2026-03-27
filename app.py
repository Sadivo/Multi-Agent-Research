"""
Streamlit 前端
Task 11: 建立 Streamlit 前端介面

需求：9.1, 9.2, 9.3, 9.4
"""

import os
import sys
import pathlib

# 確保 src/ 在 Python 路徑中（pytest 透過 pyproject.toml 設定，Streamlit 需手動加）
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import streamlit as st
from graph.graph import build_graph
from config import validate_env

st.title("Multi-Agent Research Pipeline")
st.caption("由 LangGraph 驅動的多代理人研究系統")

# 需求 9.1：文字輸入框供使用者輸入研究問題
query = st.text_input(
    "輸入研究問題",
    placeholder="例如：日本東京旅遊攻略",
)

if st.button("開始研究") and query:
    try:
        validate_env()
    except EnvironmentError as e:
        st.error(f"❌ 環境設定錯誤：{e}")
        st.stop()

    graph = build_graph()

    initial_state = {
        "user_query": query,
        "task_list": [],
        "current_task": "",
        "search_results": [],
        "analysis": "",
        "critique": "",
        "critique_count": 0,
        "final_report": "",
        "messages": [],
    }

    # 需求 9.2：即時顯示當前執行的 Agent 節點名稱
    status_placeholder = st.empty()

    final_state = None
    try:
        for chunk in graph.stream(initial_state):
            for node_name, node_output in chunk.items():
                status_placeholder.info(f"⚙️ 執行中：{node_name}")
                final_state = node_output
    except Exception as e:
        status_placeholder.error(f"❌ Pipeline 執行錯誤：{e}")
        st.stop()

    status_placeholder.success("✅ 研究完成")

    # 需求 9.3：以 Markdown 格式渲染 final_report
    if final_state and final_state.get("final_report"):
        st.markdown(final_state["final_report"])
    else:
        st.warning("⚠️ 未能產出最終報告，請重試。")

    # 需求 9.4：若 LANGCHAIN_TRACING_V2=true，顯示可點擊的 LangSmith trace 連結
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "default")
        langsmith_url = f"https://smith.langchain.com/projects/{project}"
        st.info(f"📊 LangSmith 追蹤：[{project}]({langsmith_url})")
