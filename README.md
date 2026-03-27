# Multi-Agent Research Pipeline

基於 LangGraph 的多代理人研究系統，能自動將研究問題分解為子任務、平行搜尋、整合分析，並產出結構化 Markdown 報告。

## 功能特色

- 自動任務分解：Supervisor Agent 將問題拆解為 2–4 個搜尋子任務
- 平行搜尋：使用 `asyncio.gather` 同時執行所有子任務搜尋，大幅縮短等待時間
- 品質把關：Critic Agent 評估分析品質，不通過則自動打回重做（最多 2 次）
- Grounded 輸出：所有分析強制引用搜尋來源，降低 hallucination
- 可觀測性：整合 LangSmith，追蹤每個節點的輸入輸出、token 消耗與延遲
- Streamlit 前端：即時顯示 Agent 執行狀態，最終以 Markdown 渲染報告

---

## 技術架構

### 技術棧

| 元件 | 技術 |
|------|------|
| LLM | Google Gemini 2.5 Flash Lite |
| 搜尋 | Tavily Search API |
| Agent 框架 | LangGraph |
| 前端 | Streamlit |
| 可觀測性 | LangSmith |
| 執行環境 | Python 3.11+ |

### 專案結構

```
research-agent/
├── src/
│   ├── agents/
│   │   ├── supervisor.py   # 任務分解
│   │   ├── search.py       # 平行搜尋
│   │   ├── analyst.py      # 資訊整合分析
│   │   ├── critic.py       # 品質把關
│   │   └── writer.py       # 報告產出
│   ├── graph/
│   │   ├── state.py        # AgentState 定義
│   │   ├── edges.py        # Conditional edge 路由邏輯
│   │   └── graph.py        # StateGraph 組裝
│   ├── tools/
│   │   └── search_tools.py # Tavily 工具封裝
│   ├── config.py           # 環境變數載入與驗證
│   └── main.py             # CLI 進入點
├── app.py                  # Streamlit 前端
├── tests/                  # 測試套件
├── .env.example
└── pyproject.toml
```

### Pipeline 流程

```
START
  │
  ▼
supervisor          ← 將問題分解為 2-4 個子任務
  │
  ▼
search              ← asyncio.gather 平行搜尋所有子任務
  │
  ▼
analyst             ← 整合搜尋結果，禁止使用 LLM 自身知識
  │
  ▼
critic              ← 評估分析品質（1-10 分）
  │
  ├─ passed=False & count < 2 ──→ search（補充搜尋）
  │
  └─ passed=True or count >= 2 ──→ writer
                                      │
                                      ▼
                                    END
```

### AgentState

所有節點共用一個狀態物件，在 LangGraph 節點間傳遞：

```python
class AgentState(TypedDict):
    user_query: str           # 使用者輸入的研究問題
    task_list: List[str]      # Supervisor 分解出的子任務清單
    current_task: str         # 當前執行中的子任務
    search_results: List[dict]# [{title, url, content, score}]
    analysis: str             # Analyst 的整合分析
    critique: str             # Critic 的 JSON 格式反饋
    critique_count: int       # 已打回重做次數（安全閥上限為 2）
    final_report: str         # 最終 Markdown 報告
    messages: Annotated[list, add_messages]
```

### Agent 設計說明

**Supervisor Agent**
接收使用者問題，輸出純 JSON 格式的子任務清單。JSON 解析失敗時 fallback 為以原始問題作為單一子任務繼續執行。

**Search Agent**
使用 `asyncio.to_thread` 將同步的 Tavily API 包裝為非同步，再以 `asyncio.gather` 平行執行所有子任務搜尋。結果自動 URL 去重。

**Analyst Agent**
Prompt 明確禁止使用 LLM 自身訓練知識，所有資訊必須來自搜尋結果並以 `[標題](URL)` 格式引用。資訊不足時標注「資訊不足：[說明]」。

**Critic Agent**
輸出結構化 JSON 評估（`passed`, `score`, `feedback`, `missing_aspects`）。JSON 解析失敗時 fallback 為 `passed=True`，避免 pipeline 卡住。

**Writer Agent**
將分析整理為包含概覽、主要內容、實用資訊、參考來源四個區塊的 Markdown 報告。

---

## 安裝與設定

### 前置需求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)（推薦）或 pip

### 安裝依賴

```bash
# 使用 uv（推薦）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 環境變數設定

複製 `.env.example` 並填入 API 金鑰：

```bash
cp .env.example .env
```

```env
# 必填
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# 選填（啟用 LangSmith 追蹤）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=multi-agent-research-pipeline
```

API 金鑰取得：
- Google Gemini：[Google AI Studio](https://aistudio.google.com/)
- Tavily：[Tavily Dashboard](https://app.tavily.com/)
- LangSmith：[LangSmith](https://smith.langchain.com/)（選填）

---

## 使用方式

### Streamlit 前端（推薦）

```bash
streamlit run app.py
```

開啟瀏覽器後輸入研究問題，點擊「開始研究」即可看到各 Agent 即時執行狀態，完成後顯示 Markdown 格式報告。

### CLI

```bash
cd src
python main.py
```

預設執行「台北旅遊攻略」範例查詢，印出搜尋結果。

### 程式呼叫

```python
import sys
sys.path.insert(0, "src")

from graph.graph import build_graph

graph = build_graph()

result = graph.invoke({
    "user_query": "京都 3 天旅遊攻略",
    "task_list": [],
    "current_task": "",
    "search_results": [],
    "analysis": "",
    "critique": "",
    "critique_count": 0,
    "final_report": "",
    "messages": [],
})

print(result["final_report"])
```

---

## 測試

```bash
# 執行所有測試
pytest

# 執行特定測試檔案
pytest tests/test_agents.py
pytest tests/test_graph.py
```

測試套件涵蓋各 Agent 節點、Graph 組裝、State 驗證、Conditional edge 路由邏輯，並使用 Hypothesis 進行 property-based testing。

---

## LangSmith 可觀測性

設定 `LANGCHAIN_TRACING_V2=true` 後，LangSmith 自動追蹤：

| 指標 | 說明 |
|------|------|
| 每個節點的輸入/輸出 | 完整 prompt 與 response |
| Token 消耗 | 各節點用量明細 |
| 執行延遲 | 找出瓶頸節點 |
| Critic 打回次數 | 品質把關執行情況 |
| 錯誤追蹤 | 失敗節點與原因 |

Streamlit 前端在追蹤啟用時會顯示可點擊的 LangSmith 專案連結。

---

## 設計決策

**為什麼用 LangGraph 而不是直接 chain？**
需要 Critic 打回重做的 conditional loop，StateGraph 讓流程明確可視覺化，也方便後續擴充節點。

**平行搜尋的設計**
子任務間無依賴關係，`asyncio.gather` 將搜尋時間從 N×t 降為 t，對多子任務場景效益顯著。

**Critic 的終止條件**
`critique_count >= 2` 是安全閥，防止無限循環。Production agent 必須有明確的終止條件。

**強制 grounding 的 prompt 設計**
明確禁止 LLM 使用自身知識，確保輸出可追溯到搜尋結果，降低 hallucination 風險。
