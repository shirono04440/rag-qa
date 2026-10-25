# -*- coding: utf-8 -*-
"""Web 界面：python app.py 后访问 http://127.0.0.1:8000

只用标准库 http.server，不依赖 Flask/Streamlit。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag.answer import ask
from rag.index import Index

INDEX_PATH = "data/index.json"
_index = None

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemoTodo 文档问答 · RAG Demo</title>
<style>
  :root { --navy:#142b45; --ink:#222; --gray:#6a7280; --line:#e2e7f0; --bg:#f7f8fa; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  .wrap { max-width: 780px; margin: 0 auto; padding: 56px 24px 80px; }
  h1 { font-size: 26px; color: var(--navy); margin: 0 0 6px; letter-spacing: 1px; }
  .sub { color: var(--gray); font-size: 13px; margin-bottom: 28px; }
  .box { background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px; }
  .row { display:flex; gap:10px; }
  input[type=text] { flex:1; padding:13px 14px; font-size:15px; border:1px solid var(--line);
        border-radius:8px; outline:none; font-family:inherit; }
  input[type=text]:focus { border-color:#9db3cf; }
  button { padding:13px 22px; font-size:15px; border:0; border-radius:8px; cursor:pointer;
        background:var(--navy); color:#fff; font-family:inherit; }
  button:disabled { opacity:.5; cursor:default; }
  .examples { margin:14px 0 0; font-size:12.5px; color:var(--gray); }
  .examples span { cursor:pointer; text-decoration:underline dotted; margin-right:14px; }
  .answer { margin-top:26px; }
  .meta { font-size:12px; color:var(--gray); margin-bottom:8px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:20px; background:#eef3fa;
        color:#3b6ea5; margin-right:8px; }
  .text { font-size:15.5px; line-height:1.85; white-space:pre-wrap; }
  .cites { margin-top:16px; border-top:1px dashed var(--line); padding-top:12px; }
  .cite { font-family: ui-monospace, Consolas, monospace; font-size:12.5px; color:#3b6ea5;
          margin:5px 0; }
  .cite b { color: var(--navy); font-weight:600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>MemoTodo 文档问答</h1>
  <div class="sub">BM25 + TF-IDF 混合检索 · 答案带原文出处 · 零第三方依赖</div>

  <div class="box">
    <div class="row">
      <input id="q" type="text" placeholder="问点什么，比如：今日完成是怎么判断的？" autofocus>
      <button id="go" onclick="ask()">提问</button>
    </div>
    <div class="examples">
      试试：<span onclick="fill(this)">今日完成是怎么判断的</span>
             <span onclick="fill(this)">数据库存在哪里，用了什么模式</span>
             <span onclick="fill(this)">怎么打包成 exe</span>
    </div>
  </div>

  <div class="answer" id="answer"></div>
</div>
<script>
function fill(el){ document.getElementById('q').value = el.textContent; ask(); }
async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;
  const btn = document.getElementById('go');
  btn.disabled = true;
  document.getElementById('answer').innerHTML = '';
  try{
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
                                       body: JSON.stringify({question:q})});
    const d = await r.json();
    const cites = d.citations.map(c => `<div class="cite"><b>${c.doc_path}</b>:${c.start_line}
        <span style="color:#98a2b3">（score ${c.score}）</span></div>`).join('');
    document.getElementById('answer').innerHTML = `
      <div class="meta"><span class="badge">${d.mode === 'llm' ? '大模型生成' : '原文摘录'}</span>
        ${d.latency_ms} ms · 检索 ${d.hit_count} 个片段</div>
      <div class="text">${d.text}</div>
      <div class="cites">${cites}</div>`;
  }catch(e){
    document.getElementById('answer').textContent = '出错了：' + e;
  }finally{ btn.disabled = false; }
}
document.getElementById('q').addEventListener('keydown', e => { if(e.key === 'Enter') ask(); });
</script>
</body>
</html>
"""


def get_index() -> Index:
    global _index
    if _index is None:
        _index = Index.load(INDEX_PATH)
    return _index


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = (payload.get("question") or "").strip()
        except (ValueError, TypeError):
            self._send(400, b"bad request", "text/plain")
            return

        index = get_index()
        ans = ask(index, question)
        hits = index.search(question, top_k=5)
        body = json.dumps({
            "question": question,
            "text": ans.text,
            "mode": ans.mode,
            "latency_ms": ans.latency_ms,
            "hit_count": len(hits),
            "citations": [
                {"doc_path": c.doc_path, "start_line": c.start_line, "score": c.score}
                for c in ans.citations
            ],
        }, ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("serving on http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
