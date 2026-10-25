# -*- coding: utf-8 -*-
"""问答层：检索 →（可选）大模型生成 → 摘录式兜底，并给出引用来源与耗时。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from . import llm
from .chunker import split_sentences, tokenize
from .index import Index


@dataclass
class Citation:
    doc_path: str
    start_line: int
    score: float


@dataclass
class Answer:
    question: str
    text: str
    mode: str                     # "llm" 或 "extractive"
    citations: List[Citation] = field(default_factory=list)
    latency_ms: int = 0


def _context_block(hits, max_chars: int = 3000) -> str:
    parts, used = [], 0
    for chunk, *_ in hits:
        snippet = chunk.text[:1200]
        if used + len(snippet) > max_chars:
            break
        parts.append(f"[{chunk.doc_path}:{chunk.start_line}]\n{snippet}")
        used += len(snippet)
    return "\n\n".join(parts)


def _clean(sent: str) -> str:
    """去掉 Markdown 标记与多余空白。"""
    s = sent.strip().lstrip("#-*+> ").replace("**", "").replace("`", "").replace('"""', "").strip('"\'')
    s = s.replace("　", " ")
    return " ".join(s.split())


def _is_prose(sent: str) -> bool:
    """过滤掉代码行，只保留可读的中文句子。"""
    cjk = sum(1 for ch in sent if "\u4e00" <= ch <= "\u9fff")
    if cjk < 8:
        return False
    code_marks = ("self.", "def ", "import ", "return ", "config[", "print(")
    if any(m in sent for m in code_marks):
        return False
    symbols = sum(1 for ch in sent if ch in "{}[]()=<>;:_|\\")
    return symbols <= max(2, len(sent) // 12)


def _extractive_answer(question: str, hits) -> str:
    """没有大模型时：从命中块里挑与问题重合度最高的句子，过滤代码行后拼接。"""
    q_tokens = set(tokenize(question))
    scored = []
    for rank, (chunk, *_rest) in enumerate(hits[:5]):
        for sent in split_sentences(chunk.text):
            cleaned = _clean(sent)
            if len(cleaned) < 10 or not _is_prose(cleaned):
                continue
            overlap = len(q_tokens & set(tokenize(cleaned)))
            if overlap:
                # 排名靠前的块加权，避免引用到低相关片段
                scored.append((overlap - rank * 0.9, cleaned))
    if not scored:
        return "资料中没有找到相关内容。"
    scored.sort(key=lambda x: x[0], reverse=True)

    picked, seen = [], set()
    for _, sent in scored:
        key = sent[:24]
        if key in seen:
            continue
        seen.add(key)
        picked.append(sent if sent.endswith(("。", "！", "？", "；", "，")) else sent + "。")
        if sum(len(s) for s in picked) > 230:
            break
    return "".join(picked[:3])


def ask(index: Index, question: str, top_k: int = 5, use_llm: bool = True) -> Answer:
    t0 = time.perf_counter()
    hits = index.search(question, top_k=top_k)
    if not hits:
        return Answer(question, "资料中没有找到相关内容。", "extractive", [], 1)

    citations = [
        Citation(chunk.doc_path, chunk.start_line, round(score, 4))
        for chunk, score, _bm, _cos in hits[:3]
    ]

    text, mode = None, "extractive"
    if use_llm and llm.available():
        text = llm.generate(question, _context_block(hits))
        if text:
            mode = "llm"
    if not text:
        text = _extractive_answer(question, hits)

    return Answer(question, text, mode, citations,
                  int((time.perf_counter() - t0) * 1000))
