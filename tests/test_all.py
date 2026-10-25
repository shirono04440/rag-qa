# -*- coding: utf-8 -*-
"""基础自测：python tests/test_all.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.answer import ask
from rag.chunker import chunk_documents, tokenize
from rag.corpus import Doc, load_documents
from rag.index import Index


def test_tokenize():
    tokens = tokenize("深度学习 PyTorch")
    assert "深" in tokens and "深度" in tokens, "中文应拆出 unigram 与 bigram"
    assert "pytorch" in tokens, "英文应转小写"
    print("✓ tokenize")


def test_chunking():
    docs = [Doc(path="a.md", text="第一段内容。" * 60 + "\n" + "第二段内容。" * 60)]
    chunks = chunk_documents(docs, target_size=120, overlap=20)
    assert len(chunks) >= 2, "长文档应被切成多块"
    assert all(c.doc_path == "a.md" for c in chunks)
    print(f"✓ chunking（{len(chunks)} 块）")


def test_index_roundtrip(tmp_path: str = "data/_test_index.json"):
    docs = [Doc(path="x.md", text="数据库使用 SQLite 并开启 WAL 模式。窗口使用 Tkinter 自绘。")]
    chunks = chunk_documents(docs, target_size=200, overlap=0)
    index = Index(chunks)
    index.save(tmp_path)
    loaded = Index.load(tmp_path)

    assert len(loaded.chunks) == len(chunks)
    hits = loaded.search("数据库 模式", top_k=1)
    assert hits and hits[0][1] > 0, "应能检索到相关块"
    os.remove(tmp_path)
    print("✓ index 存取与检索")


def test_answer_has_citation():
    if not os.path.exists("data/index.json"):
        print("… 跳过 answer 测试（尚未建库）")
        return
    index = Index.load("data/index.json")
    ans = ask(index, "今日完成是怎么判断的", use_llm=False)
    assert ans.text and ans.text != "资料中没有找到相关内容。"
    assert ans.citations, "答案必须带引用"
    assert ans.latency_ms >= 0
    print(f"✓ answer（{ans.latency_ms} ms，{len(ans.citations)} 条引用）")


if __name__ == "__main__":
    test_tokenize()
    test_chunking()
    test_index_roundtrip()
    test_answer_has_citation()
    print("\n全部通过")
