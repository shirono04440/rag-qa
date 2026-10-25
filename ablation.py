# -*- coding: utf-8 -*-
"""切块大小消融实验：对比不同 chunk 尺寸下的检索效果。

用法：python ablation.py
"""

from __future__ import annotations

import statistics
import time

from eval import GOLD
from rag.chunker import chunk_documents
from rag.corpus import load_documents
from rag.index import Index

SOURCE = r"D:\todo"
SIZES = [200, 320, 420, 600, 800]


def run_one(size: int) -> dict:
    docs = load_documents(SOURCE)
    chunks = chunk_documents(docs, target_size=size, overlap=max(20, size // 5))
    index = Index(chunks)

    hits1 = hits3 = 0
    rr = []
    latencies = []
    for question, expect, _kw in GOLD:
        t0 = time.perf_counter()
        result = index.search(question, top_k=3)
        latencies.append((time.perf_counter() - t0) * 1000)
        paths = [c.doc_path for c, *_ in result]
        rank = next((i + 1 for i, p in enumerate(paths) if p in expect), None)
        hits1 += 1 if rank == 1 else 0
        hits3 += 1 if rank and rank <= 3 else 0
        rr.append(1.0 / rank if rank else 0.0)

    n = len(GOLD)
    return {
        "chunk_size": size,
        "chunks": len(chunks),
        "hit@1": round(hits1 / n, 3),
        "hit@3": round(hits3 / n, 3),
        "mrr": round(sum(rr) / n, 3),
        "ms": round(statistics.mean(latencies), 2),
    }


if __name__ == "__main__":
    rows = [run_one(s) for s in SIZES]
    header = f"{'chunk':>6}{'块数':>7}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>7}{'ms':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['chunk_size']:>6}{r['chunks']:>7}{r['hit@1']:>8}{r['hit@3']:>8}"
              f"{r['mrr']:>7}{r['ms']:>7}")
