# -*- coding: utf-8 -*-
"""可插拔的大模型调用：有 API key 就走生成，没有就返回 None 由上层降级为摘录式回答。

只用标准库 urllib，兼容任何 OpenAI 风格的 /chat/completions 接口。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def generate(question: str, context: str, timeout: int = 40) -> Optional[str]:
    """调用 DeepSeek 生成答案；未配置 key 或调用失败时返回 None。"""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None

    system = (
        "你是文档问答助手。只能依据给定资料回答，资料中没有的信息直接说不知道。"
        "回答用中文，控制在 120 字以内，不要编造出处。"
    )
    user = f"资料：\n{context}\n\n问题：{question}"

    body = json.dumps({
        "model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, ValueError, TimeoutError):
        return None
