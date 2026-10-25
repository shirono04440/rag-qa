# -*- coding: utf-8 -*-
"""切分与分词：中文按字符 bigram，英文/数字按词，兼顾零依赖与召回效果。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

CJK = re.compile(r"[\u4e00-\u9fff]")
ASCII_WORD = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?")
SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")


@dataclass
class Chunk:
    id: int
    doc_path: str
    text: str
    start_line: int


def tokenize(text: str) -> List[str]:
    """中文 uni+bi-gram，英文按词；返回 token 列表。"""
    tokens: List[str] = []
    tokens.extend(w.lower() for w in ASCII_WORD.findall(text))

    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        tokens.extend(run)                                   # unigram
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))  # bigram
    return tokens


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def chunk_documents(docs, target_size: int = 420, overlap: int = 80) -> List[Chunk]:
    """按段落聚合成 ~target_size 字的块，块间保留 overlap 字的重叠。"""
    chunks: List[Chunk] = []
    cid = 0
    for doc in docs:
        lines = doc.text.splitlines()
        buf: List[str] = []
        buf_len = 0
        start_line = 1

        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if buf_len == 0:
                start_line = lineno
            buf.append(stripped)
            buf_len += len(stripped)

            if buf_len >= target_size:
                text = "\n".join(buf)
                chunks.append(Chunk(cid, doc.path, text, start_line))
                cid += 1
                tail = text[-overlap:] if overlap > 0 else ""
                buf = [tail] if tail else []
                buf_len = len(tail)
                start_line = lineno

        if buf:
            text = "\n".join(buf)
            if text.strip():
                chunks.append(Chunk(cid, doc.path, text, start_line))
                cid += 1
    return chunks
