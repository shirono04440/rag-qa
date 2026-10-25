# -*- coding: utf-8 -*-
"""检索索引：BM25 与 TF-IDF 余弦的混合检索，纯 Python 实现，索引可落盘为 JSON。"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Dict, List, Tuple

from .chunker import Chunk, tokenize

K1 = 1.5
B = 0.75
BM25_WEIGHT = 0.65
COS_WEIGHT = 0.35


class Index:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tf: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        self.df: Counter = Counter()

        for ch in chunks:
            counts = Counter(tokenize(ch.text))
            self.tf.append(dict(counts))
            self.doc_len.append(sum(counts.values()))
            self.df.update(counts.keys())

        self.n = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.idf = {
            t: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }
        self.vectors = [self._tfidf_vector(i) for i in range(self.n)]

    # ---------------- 向量化 ----------------
    def _tfidf_vector(self, i: int) -> Dict[str, float]:
        vec = {}
        for t, c in self.tf[i].items():
            w = (1 + math.log(c)) * self.idf.get(t, 0.0)
            if w:
                vec[t] = w
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm:
            vec = {t: v / norm for t, v in vec.items()}
        return vec

    # ---------------- 打分 ----------------
    def bm25_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.n
        for t in set(query_tokens):
            if t not in self.df:
                continue
            idf = self.idf[t]
            for i in range(self.n):
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                dl = self.doc_len[i] or 1
                denom = f + K1 * (1 - B + B * dl / (self.avgdl or 1))
                scores[i] += idf * f * (K1 + 1) / denom
        return scores

    def cosine_scores(self, query_tokens: List[str]) -> List[float]:
        qcount = Counter(query_tokens)
        qvec = {}
        for t, c in qcount.items():
            w = (1 + math.log(c)) * self.idf.get(t, 0.0)
            if w:
                qvec[t] = w
        norm = math.sqrt(sum(v * v for v in qvec.values()))
        if not norm:
            return [0.0] * self.n
        qvec = {t: v / norm for t, v in qvec.items()}

        scores = []
        for vec in self.vectors:
            small, big = (qvec, vec) if len(qvec) < len(vec) else (vec, qvec)
            scores.append(sum(w * big.get(t, 0.0) for t, w in small.items()))
        return scores

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float, float, float]]:
        """返回 [(chunk, 混合分, bm25 分, 余弦分)]，按混合分降序。"""
        qt = tokenize(query)
        if not qt or not self.n:
            return []
        bm = self.bm25_scores(qt)
        cos = self.cosine_scores(qt)

        bm_max = max(bm) or 1.0
        cos_max = max(cos) or 1.0
        hybrid = [
            BM25_WEIGHT * (bm[i] / bm_max) + COS_WEIGHT * (cos[i] / cos_max)
            for i in range(self.n)
        ]

        order = sorted(range(self.n), key=lambda i: hybrid[i], reverse=True)[:top_k]
        return [(self.chunks[i], hybrid[i], bm[i], cos[i]) for i in order if hybrid[i] > 0]

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        payload = {
            "chunks": [
                {"id": c.id, "doc_path": c.doc_path, "text": c.text, "start_line": c.start_line}
                for c in self.chunks
            ]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Index":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return cls(chunks)
