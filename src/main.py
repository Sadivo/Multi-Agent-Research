"""
CLI 進入點 — 最小可執行示範
Task 3: 以單一子任務呼叫 search_node 並印出結果
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config  # noqa: F401 — 觸發 load_dotenv()
from agents.search import search_node


def main():
    state = {
        "user_query": "台北旅遊攻略",
        "task_list": ["台北必去景點推薦"],
        "current_task": "",
        "search_results": [],
        "analysis": "",
        "critique": "",
        "critique_count": 0,
        "final_report": "",
        "messages": [],
    }

    print("執行 Search Agent...")
    result = search_node(state)
    print(f"搜尋結果數量：{len(result['search_results'])}")
    print(json.dumps(result["search_results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
