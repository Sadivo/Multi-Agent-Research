"""
Tests for Multi-Agent Research Pipeline.

Sub-tasks 1.1 & 1.2:
  - Unit tests: requirements.txt and .env.example content validation
  - Property test (屬性 11): missing env var produces clear error message
"""

import os
import pathlib
import pytest
from unittest import mock
from hypothesis import given, settings
import hypothesis.strategies as st

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent  # project root

from config import REQUIRED_ENV_VARS, validate_env  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 1.1 — Unit tests: requirements.txt & .env.example
# ══════════════════════════════════════════════════════════════════════════

REQUIRED_PACKAGES = [
    "langgraph",
    "langchain",
    "langchain-openai",
    "langchain-community",
    "tavily-python",
    "langsmith",
    "streamlit",
    "python-dotenv",
    "langchain-google-genai",
]

REQUIRED_ENV_KEYS = [
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
]


def test_requirements_txt_contains_all_packages():
    """驗證 requirements.txt 包含所有必要套件（需求 10.2）"""
    content = (ROOT / "requirements.txt").read_text()
    for pkg in REQUIRED_PACKAGES:
        assert pkg in content, f"requirements.txt 缺少套件：{pkg}"


def test_env_example_contains_all_keys():
    """驗證 .env.example 包含所有必要環境變數鍵（需求 10.1）"""
    content = (ROOT / ".env.example").read_text()
    for key in REQUIRED_ENV_KEYS:
        assert key in content, f".env.example 缺少環境變數：{key}"


# ══════════════════════════════════════════════════════════════════════════
# Sub-task 1.2 — Property test 屬性 11: missing env var → clear error
# Validates: Requirements 10.3
# ══════════════════════════════════════════════════════════════════════════

@settings(max_examples=100)
@given(st.sampled_from(REQUIRED_ENV_VARS))
def test_missing_env_var_produces_clear_error(missing_var):
    """
    **Validates: Requirements 10.3**

    屬性 11：對於任意缺少一個必要環境變數的執行環境，
    validate_env() 拋出的錯誤訊息應包含該缺少的環境變數名稱。
    """
    env_with_all = {v: "dummy_value" for v in REQUIRED_ENV_VARS}
    env_with_all.pop(missing_var)

    with mock.patch.dict(os.environ, env_with_all, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            validate_env()

    assert missing_var in str(exc_info.value), (
        f"錯誤訊息未包含缺少的環境變數名稱：{missing_var}"
    )

