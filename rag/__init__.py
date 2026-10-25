# -*- coding: utf-8 -*-
"""零依赖的检索增强问答（RAG）实现。"""

from .answer import Answer, Citation, ask
from .chunker import Chunk, chunk_documents, split_sentences, tokenize
from .corpus import Doc, load_documents
from .index import Index

__all__ = [
    "Answer", "Citation", "ask",
    "Chunk", "chunk_documents", "split_sentences", "tokenize",
    "Doc", "load_documents",
    "Index",
]
__version__ = "1.0.0"
