# -*- coding: utf-8 -*-
"""建库入口：python -m rag.build --source D:\\todo --out data/index.json"""

from __future__ import annotations

import argparse
import json
import time

from .chunker import chunk_documents
from .corpus import load_documents
from .index import Index


def main() -> None:
    ap = argparse.ArgumentParser(description="为指定目录构建检索索引")
    ap.add_argument("--source", required=True, help="要索引的目录")
    ap.add_argument("--out", default="data/index.json")
    ap.add_argument("--size", type=int, default=600, help="块大小（字符）")
    ap.add_argument("--overlap", type=int, default=120, help="块间重叠（字符）")
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = load_documents(args.source)
    chunks = chunk_documents(docs, target_size=args.size, overlap=args.overlap)
    index = Index(chunks)
    index.save(args.out)

    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "vocabulary": len(index.idf),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / max(len(chunks), 1), 1),
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
