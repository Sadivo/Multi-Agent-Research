# Multi-Agent Research Pipeline 逐步教學

> 本教材適合：有 Python 基礎，但不熟悉 LangGraph / 多代理人系統的開發者
> 閱讀完成後，你將能夠理解：
> - LangGraph 的 StateGraph 是什麼、怎麼運作
> - 多個 AI Agent 如何分工協作
> - 條件路由（Conditional Edge）如何控制流程
> - 平行搜尋的實作方式
> - 整個 Pipeline 從輸入到輸出的完整資料流

---

## 前言

這個專案是一個「自動研究助理」：你丟一個問題進去（例如「日本東京旅遊攻略」），它會自動：

1. 把問題拆成幾個子任務
2. 平行上網搜尋每個子任務
3. 整合搜尋結果做分析
4. 評估分析品質，不夠好就重搜
5. 最後產出一份結構化的 Markdown 報告

整個流程由 **LangGraph** 驅動，核心概念是「有狀態的有向圖」——每個 Agent 是圖上的一個節點，資料在節點之間流動。

---

## Step 1：理解「狀態」是什麼（AgentState）

整個 Pipeline 的資料都存在一個叫 `AgentState` 的字典裡。你可以把它想成一個「共用白板」，每個 Agent 都從這塊白板讀資料、寫結果。

### 程式碼

```python
# src/graph/state.py
class AgentState(TypedDict):
    user_query: str        # 使用者輸入的研究問題
    task_list: List[str]   # supervisor 分解出的子任務清單
    current_task: str      # 目前執行的子任務
    search_results: List[dict]  # 搜尋結果 [{title, url, content, score}]
    analysis: str          # analyst 產出的分析文字
    critique: str          # critic 的評估結果（JSON 字串）
    critique_count: int    # 已評估幾次（安全閥用）
    final_report: str      # writer 產出的最終 Markdown 報告
    messages: Annotated[list, add_messages]  # LangGraph 訊息歷史
```

### 概念說明

`TypedDict` 是 Python 的型別提示工具，讓字典的每個 key 都有明確的型別。這裡用它定義「白板上有哪些欄位」。

`Annotated[list, add_messages]` 是 LangGraph 的特殊語法：`add_messages` 告訴 LangGraph 這個欄位要用「累加」而不是「覆蓋」的方式更新——新訊息會附加到清單末尾，不會把舊的蓋掉。

### 為什麼這樣做？

每個 Agent 節點只需要回傳它「更新的欄位」，不需要回傳完整 state。LangGraph 會自動把回傳的 partial dict 合併回完整 state。這樣每個 Agent 的職責很清楚，不會互相干擾。

---

## Step 2：理解圖的結構（build_graph）

LangGraph 的核心是 `StateGraph`——一個有方向的圖，節點是 Agent，邊是資料流向。

### 程式碼

```python
# src/graph/graph.py
def build_graph():
    graph = StateGraph(AgentState)

    # 新增節點（每個節點對應一個 Agent 函式）
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search", search_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    # 設定起點
    graph.set_entry_point("supervisor")

    # 設定固定邊（A 執行完一定去 B）
    graph.add_edge("supervisor", "search")
    graph.add_edge("search", "analyst")
    graph.add_edge("analyst", "critic")

    # 設定條件邊（critic 執行完，根據結果決定去哪）
    graph.add_conditional_edges(
        "critic",
        should_revise,                              # 路由函式
        {"search": "search", "writer": "writer"},   # 可能的目標
    )

    graph.add_edge("writer", END)

    return graph.compile()
```

### 概念說明

```
supervisor → search → analyst → critic ──(passed 或 count>=2)──→ writer → END
                         ↑              │
                         └──(failed)────┘
```

`add_edge` 是固定路徑，`add_conditional_edges` 是分叉路口——由 `should_revise` 函式決定走哪條路。

`graph.compile()` 會把這個圖「鎖定」成可執行的物件，之後就可以用 `.stream()` 或 `.invoke()` 執行。

### 為什麼這樣做？

把流程定義成圖，而不是一堆 if/else，有幾個好處：
- 流程一目了然，容易修改
- LangGraph 自動處理狀態傳遞，不需要手動傳參數
- 可以用 `.stream()` 逐節點取得輸出，方便即時顯示進度

---

## Step 3：第一個 Agent — Supervisor（任務分解）

Supervisor 的工作是把一個大問題拆成 2-4 個具體的搜尋子任務。

### 程式碼

```python
# src/agents/supervisor.py
SUPERVISOR_PROMPT = """你是一個研究任務分解專家。請將以下研究問題分解為 2 到 4 個具體的搜尋子任務。
必須以純 JSON 格式回應，不得包含任何 Markdown 標記或程式碼區塊。
格式：{{"tasks": ["子任務1", "子任務2"]}}

研究問題：{user_query}"""

def supervisor_node(state: AgentState) -> dict:
    user_query = state["user_query"]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    response = llm.invoke(SUPERVISOR_PROMPT.format(user_query=user_query))

    try:
        parsed = json.loads(response.content)
        task_list = parsed["tasks"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Fallback：解析失敗就把原始問題當成單一子任務
        task_list = [user_query]

    return {
        "task_list": task_list,
        "current_task": task_list[0],
    }
```

### 概念說明

這裡用了一個重要的設計模式：**JSON Fallback**。

LLM 的輸出不是 100% 可靠的——有時候它會在 JSON 外面包一層 Markdown 的 ` ```json ``` `，或者格式稍微不對。所以每次解析 LLM 的 JSON 輸出，都要有 `try/except` 保底。

`ChatGoogleGenerativeAI` 是 LangChain 對 Google Gemini API 的封裝，`llm.invoke(prompt)` 就是送出一個請求並等待回應。

### 為什麼在函式內部初始化 LLM？

```python
# 每次呼叫都建立新的 LLM 實例
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
```

這個專案刻意不用全域單例，而是在每個 Agent 函式內部初始化 LLM。這樣每個 Agent 的設定互相獨立，未來要換不同模型或參數時，只需要改那個 Agent，不影響其他人。

---

## Step 4：平行搜尋（Search Agent）

Search Agent 對每個子任務同時發出搜尋請求，而不是一個一個等。

### 程式碼

```python
# src/agents/search.py
def search_node(state: AgentState) -> dict:
    task_list = state.get("task_list", [])
    search_tool = get_search_tool()  # 取得 Tavily 搜尋工具

    async def _run_parallel() -> list:
        async def search_one(task: str) -> list:
            try:
                # asyncio.to_thread：把同步函式包成非同步，避免阻塞
                raw = await asyncio.to_thread(search_tool.invoke, task)
                return [{"title": r.get("title", ""), "url": r.get("url", ""),
                         "content": r.get("content", ""), "score": r.get("score", 0.0)}
                        for r in raw]
            except Exception as e:
                logger.error("Tavily API 錯誤：%s", e)
                return []

        # asyncio.gather：同時執行所有搜尋，等全部完成
        results_per_task = await asyncio.gather(*[search_one(t) for t in task_list])
        return [item for sublist in results_per_task for item in sublist]

    all_results = asyncio.run(_run_parallel())
    return {"search_results": deduplicate_results(all_results)}
```

### 概念說明

**為什麼需要平行搜尋？**

假設有 3 個子任務，每個搜尋需要 2 秒。
- 循序執行：3 × 2 = 6 秒
- 平行執行：max(2, 2, 2) = 2 秒

`asyncio.gather` 就是「同時發出所有請求，等全部回來」的工具。

**為什麼用 `asyncio.to_thread`？**

Tavily 的 Python SDK 是同步 API（`search_tool.invoke` 會阻塞等待）。在 async 環境裡直接呼叫同步函式會卡住整個事件迴圈。`asyncio.to_thread` 把同步函式丟到另一個執行緒去跑，讓主執行緒可以繼續處理其他任務。

**URL 去重**

多個子任務可能搜到同一個網頁，`deduplicate_results` 用 `set` 追蹤已見過的 URL，確保每個 URL 只出現一次。

---

## Step 5：Grounded 分析（Analyst Agent）

Analyst 的特別之處在於它被明確禁止使用 LLM 自身的訓練知識，只能根據搜尋結果分析。

### 程式碼

```python
# src/agents/analyst.py
_PROMPT_TEMPLATE = """\
你是一個嚴謹的研究分析師。請根據以下搜尋結果進行分析，嚴格禁止使用你自身的訓練知識。
所有資訊必須來自提供的搜尋結果，並以 [來源標題](URL) 格式引用來源。
若某方面資訊不足，請標注「資訊不足：[說明]」。
...
"""

def analyst_node(state: AgentState) -> dict:
    search_results = state.get("search_results", [])

    if not search_results:
        # 沒有搜尋結果就直接回傳說明，不呼叫 LLM
        return {"analysis": "無法取得搜尋結果，無法進行分析。"}

    # 把搜尋結果格式化成 prompt 可讀的文字
    formatted_results = _format_results(search_results)
    prompt = _PROMPT_TEMPLATE.format(user_query=..., formatted_results=formatted_results)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    response = llm.invoke(prompt)
    return {"analysis": response.content}
```

### 概念說明

這是 **RAG（Retrieval-Augmented Generation）** 的核心思想：先取得外部資料（搜尋結果），再讓 LLM 根據這些資料生成回答，而不是憑空回答。

Prompt 裡明確寫「嚴格禁止使用你自身的訓練知識」，這是一種 **Grounding** 技術——把 LLM 的回答「錨定」在提供的資料上，減少幻覺（hallucination）。

---

## Step 6：品質把關與計數器（Critic Agent）

Critic 評估分析品質，並決定是否需要重新搜尋。

### 程式碼

```python
# src/agents/critic.py
def critic_node(state: AgentState) -> dict:
    critique_count = state.get("critique_count", 0)

    # 呼叫 LLM 評估，要求輸出純 JSON
    response = llm.invoke(prompt)

    try:
        critique_data = json.loads(response.content)
        passed = bool(critique_data.get("passed", True))
        # ... 解析其他欄位
        critique_json = json.dumps({...}, ensure_ascii=False)
    except (json.JSONDecodeError, ...):
        # Fallback：解析失敗就當作通過
        passed = True
        critique_json = json.dumps({"passed": True, "score": 5, ...})

    # 只有 passed=False 且還沒到上限，才遞增計數器
    new_count = critique_count
    if not passed and critique_count < 2:
        new_count = critique_count + 1

    return {
        "critique": critique_json,
        "critique_count": new_count,
    }
```

### 概念說明

**為什麼 critique 存成 JSON 字串而不是 dict？**

`AgentState` 是 `TypedDict`，`critique` 欄位型別是 `str`。把結構化資料序列化成 JSON 字串存進去，需要時再 `json.loads` 解析，是一種在型別限制下傳遞結構化資料的常見做法。

**計數器的設計**

注意：`critic_node` 只負責「遞增計數器」，不負責「根據計數器決定去哪」。路由決策完全交給下一步的 `should_revise`。這是單一職責原則的體現。

---

## Step 7：條件路由（should_revise）

這是整個 Pipeline 最關鍵的控制邏輯——決定要重搜還是直接寫報告。

### 程式碼

```python
# src/graph/edges.py
def should_revise(state: AgentState) -> str:
    critique_count = state.get("critique_count", 0)

    # 安全閥：已評估 2 次，強制進入 writer，防止無限循環
    if critique_count >= 2:
        return "writer"

    critique_str = state.get("critique", "{}")
    try:
        critique_data = json.loads(critique_str)
        passed = bool(critique_data.get("passed", True))
    except (json.JSONDecodeError, ...):
        passed = True  # 解析失敗預設通過

    return "writer" if passed else "search"
```

### 概念說明

```
critic_node 執行完
        ↓
should_revise(state) 被呼叫
        ↓
回傳 "writer" 或 "search"
        ↓
LangGraph 根據回傳值決定下一個節點
```

**安全閥（Safety Valve）**

`critique_count >= 2` 這個條件是防止無限循環的保險絲。如果 Critic 一直說「不夠好」，最多重搜 2 次就強制進入 Writer，確保 Pipeline 一定會結束。

**重要：`should_revise` 只讀不寫**

這個函式只讀取 state，不修改任何欄位。這是刻意的設計——路由邏輯和狀態修改分開，讓程式碼更容易測試和理解。

---

## Step 8：最終報告（Writer Agent）

Writer 把分析整理成結構化的 Markdown 報告。

### 程式碼

```python
# src/agents/writer.py
_PROMPT_TEMPLATE = """\
...
請輸出包含以下區塊的 Markdown 報告：
1. ## 概覽 — 簡短摘要
2. ## 主要內容 — 詳細資訊
3. ## 實用資訊 — 預算、時間、注意事項等
4. ## 參考來源 — 以 [來源標題](URL) 格式列出所有引用的 URL
"""

def writer_node(state: AgentState) -> dict:
    response = llm.invoke(prompt)
    return {"final_report": response.content}
```

### 概念說明

Writer 是最簡單的節點——它只做一件事：把 `analysis` 轉換成格式化的 Markdown。Prompt 裡明確規定了報告結構，確保輸出一致。

---

## Step 9：工具層（Tavily 搜尋封裝）

### 程式碼

```python
# src/tools/search_tools.py
from langchain_community.tools.tavily_search import TavilySearchResults

def get_search_tool() -> TavilySearchResults:
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
    )
```

### 概念說明

`TavilySearchResults` 是 LangChain 對 Tavily Search API 的封裝。`max_results=5` 表示每次搜尋最多回傳 5 筆結果，`search_depth="advanced"` 表示使用更深入的搜尋模式（會比較慢但結果更豐富）。

用 `get_search_tool()` 工廠函式而不是直接實例化，方便在測試時 mock 掉這個函式，不需要真的發出 API 請求。

---

## Step 10：環境設定與驗證（config.py）

### 程式碼

```python
# src/config.py
from dotenv import load_dotenv

load_dotenv()  # 模組載入時自動執行，從 .env 讀取環境變數

REQUIRED_ENV_VARS = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY"]

def validate_env():
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"缺少必要環境變數：{', '.join(missing)}")
```

### 概念說明

`load_dotenv()` 在模組層級執行（不在函式內），所以只要 `import config` 就會自動載入 `.env` 檔案。

`validate_env()` 則是在 Streamlit 啟動時才呼叫，提供明確的錯誤訊息，告訴使用者缺少哪些設定。

---

## Step 11：前端（Streamlit app.py）

### 程式碼

```python
# app.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))  # 手動加 src/ 到路徑

import streamlit as st
from graph.graph import build_graph

query = st.text_input("輸入研究問題")

if st.button("開始研究") and query:
    validate_env()
    graph = build_graph()

    initial_state = {
        "user_query": query,
        "task_list": [], "current_task": "",
        "search_results": [], "analysis": "",
        "critique": "", "critique_count": 0,
        "final_report": "", "messages": [],
    }

    status_placeholder = st.empty()

    # graph.stream() 逐節點回傳輸出，可以即時顯示進度
    for chunk in graph.stream(initial_state):
        for node_name, node_output in chunk.items():
            status_placeholder.info(f"⚙️ 執行中：{node_name}")
            final_state = node_output

    st.markdown(final_state["final_report"])
```

### 概念說明

**為什麼要手動加 `sys.path`？**

`pyproject.toml` 裡設定了 `pythonpath = ["src"]`，這個設定只在 pytest 執行時生效。Streamlit 直接執行 `app.py` 時不走 pytest 的設定，所以需要手動把 `src/` 加到 Python 的模組搜尋路徑。

**`graph.stream()` vs `graph.invoke()`**

- `invoke()`：等整個 Pipeline 跑完才回傳最終結果
- `stream()`：每個節點執行完就立刻回傳那個節點的輸出

用 `stream()` 可以即時更新 UI，讓使用者看到「現在在執行哪個 Agent」，體驗更好。

---

## 完整資料流總覽

```
使用者輸入："日本東京旅遊攻略"
        ↓
[supervisor_node]
  讀取：user_query
  寫入：task_list = ["東京景點推薦", "東京交通攻略", "東京美食指南"]
        current_task = "東京景點推薦"
        ↓
[search_node]
  讀取：task_list
  平行搜尋 3 個子任務（asyncio.gather）
  寫入：search_results = [{title, url, content, score}, ...]
        ↓
[analyst_node]
  讀取：user_query, search_results
  寫入：analysis = "根據搜尋結果的詳細分析..."
        ↓
[critic_node]
  讀取：user_query, analysis, critique_count
  寫入：critique = '{"passed": false, "score": 6, ...}'
        critique_count = 1
        ↓
[should_revise] ← 條件路由函式
  critique_count=1 < 2，且 passed=false → 回傳 "search"
        ↓
[search_node] ← 重新搜尋
  ...（同上）
        ↓
[analyst_node] → [critic_node]
  這次 passed=true → should_revise 回傳 "writer"
        ↓
[writer_node]
  讀取：user_query, analysis
  寫入：final_report = "# 東京旅遊完整攻略\n## 概覽..."
        ↓
END → Streamlit 顯示 final_report
```

---

## 延伸思考

1. **目前 `search_node` 每次重搜都搜同樣的 `task_list`**，沒有利用 `critique` 裡的 `missing_aspects` 來針對性補搜。這是一個可以改進的地方。

2. **`current_task` 欄位**在 supervisor 設定後，後續節點都沒有用到它，實際上 search_node 是搜尋整個 `task_list`。這個欄位目前有點多餘。

3. **LangSmith 追蹤**：設定 `LANGCHAIN_TRACING_V2=true` 後，所有 LLM 呼叫都會自動被記錄到 LangSmith，可以在 dashboard 上看到每個節點的輸入輸出和延遲，對 debug 很有幫助。
