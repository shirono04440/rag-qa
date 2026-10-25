# -*- coding: utf-8 -*-
"""检索评测：在人工标注的问题集上计算 Hit@1 / Hit@3 / MRR 与平均耗时。"""

from __future__ import annotations

import json
import statistics
import time

from rag.answer import ask
from rag.index import Index

INDEX_PATH = "data/index.json"

# (问题, 期望命中的文件, 答案里应出现的关键词)
GOLD = [
    ("今日完成是怎么判断的", ["README.md", "src/monitor.py"], "轮询"),
    ("数据库存在哪里，用了什么模式", ["README.md", "src/database.py"], "WAL"),
    ("为什么以凌晨 4 点为界", ["README.md", "src/monitor.py"], "4"),
    ("窗口无边框和圆角是怎么做的", ["src/gui/_resize.py", "README.md"], "SetWindowRgn"),
    ("系统托盘怎么实现的", ["src/gui/_tray.py", "README.md"], "托盘"),
    ("怎么打包成 exe", ["README.md"], "pyinstaller"),
    ("exe 的图标是怎么提取的", ["src/icon_utils.py", "README.md"], "图标"),
    ("拖拽排序在哪实现", ["src/gui/_drag_drop.py", "README.md"], "拖拽"),
    ("开机自动启动怎么做的", ["src/autostart.py", "README.md"], "注册表"),
    ("测试覆盖了哪些内容", ["README.md", "tests/test_all.py"], "测试"),
]


def evaluate(top_k: int = 3) -> dict:
    index = Index.load(INDEX_PATH)
    hits1 = hits3 = 0
    reciprocal = []
    latencies = []
    keyword_ok = 0
    details = []

    for question, expect_files, keyword in GOLD:
        t0 = time.perf_counter()
        result = index.search(question, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        paths = [c.doc_path for c, *_ in result]
        rank = next((i + 1 for i, p in enumerate(paths) if p in expect_files), None)

        hits1 += 1 if rank == 1 else 0
        hits3 += 1 if rank and rank <= 3 else 0
        reciprocal.append(1.0 / rank if rank else 0.0)

        ans = ask(index, question, top_k=top_k, use_llm=False)
        ok = keyword.lower() in ans.text.lower()
        keyword_ok += 1 if ok else 0

        details.append({
            "question": question,
            "rank": rank,
            "top": paths[:3],
            "keyword_ok": ok,
        })

    n = len(GOLD)
    return {
        "questions": n,
        "hit@1": round(hits1 / n, 3),
        "hit@3": round(hits3 / n, 3),
        "mrr": round(sum(reciprocal) / n, 3),
        "keyword_hit": round(keyword_ok / n, 3),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps({k: v for k, v in report.items() if k != "details"},
                     ensure_ascii=False, indent=2))
    if "-v" in __import__("sys").argv:
        print(json.dumps(report["details"], ensure_ascii=False, indent=2))
