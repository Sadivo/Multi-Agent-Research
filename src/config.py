import os
from dotenv import load_dotenv

# load_dotenv() 在模組載入時執行，自動從 .env 讀取環境變數。
# LangSmith 追蹤相關環境變數（無需修改其他模組即可啟用）：
#   LANGCHAIN_TRACING_V2=true   — 啟用 LangSmith 自動追蹤（需求 8.1）
#   LANGCHAIN_API_KEY=<key>     — LangSmith API 金鑰
#   LANGCHAIN_PROJECT=<name>    — 追蹤歸屬的專案名稱（需求 8.3）
load_dotenv()

REQUIRED_ENV_VARS = [
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
    "LANGCHAIN_API_KEY",
]


def validate_env():
    """驗證必要環境變數是否已設定，若缺少則拋出明確錯誤訊息。"""
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"缺少必要環境變數：{', '.join(missing)}。請參考 .env.example 設定。"
        )
