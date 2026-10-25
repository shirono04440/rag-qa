# -*- coding: utf-8 -*-
"""语料加载：把目录下的文本类文件读成文档，供后续切分与建索引。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

TEXT_EXT = {".md", ".txt", ".py", ".json", ".html", ".css", ".js", ".yaml", ".yml"}
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", "build", "dist",
    "node_modules", ".idea", ".vscode", ".claude", "data",
}


@dataclass
class Doc:
    path: str          # 相对路径，作为引用来源
    text: str


def _iter_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in TEXT_EXT:
                yield os.path.join(dirpath, name)


def load_documents(root: str, max_file_bytes: int = 400_000) -> List[Doc]:
    """读取 root 下所有文本文件，返回 Doc 列表（跳过超大文件与空文件）。"""
    root = os.path.abspath(root)
    docs: List[Doc] = []
    for full in _iter_files(root):
        try:
            if os.path.getsize(full) > max_file_bytes:
                continue
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        if not text.strip():
            continue
        rel = os.path.relpath(full, root).replace("\\", "/")
        docs.append(Doc(path=rel, text=text))
    docs.sort(key=lambda d: d.path)
    return docs
