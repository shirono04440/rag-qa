# -*- coding: utf-8 -*-
"""命令行问答：python -m rag.cli "今日完成是怎么判断的？" """

from __future__ import annotations

import sys

from .answer import ask
from .index import Index

DEFAULT_INDEX = "data/index.json"


def format_answer(ans) -> str:
    lines = [f"问：{ans.question}", f"答：{ans.text}", ""]
    for c in ans.citations:
        lines.append(f"  · {c.doc_path}:{c.start_line}  (score {c.score})")
    lines.append(f"[{ans.mode}] {ans.latency_ms} ms")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python -m rag.cli '你的问题'")
        raise SystemExit(1)

    question = " ".join(sys.argv[1:])
    index = Index.load(DEFAULT_INDEX)
    print(format_answer(ask(index, question)))


if __name__ == "__main__":
    main()
