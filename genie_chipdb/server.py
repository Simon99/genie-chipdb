from __future__ import annotations

import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

from .database import ChipDatabase


def create_app(data_dir: str = "./chroma_data", lm_studio_url: str = "http://localhost:1234/v1") -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app)

    db = ChipDatabase(data_dir=data_dir, lm_studio_url=lm_studio_url)

    frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"

    @app.route("/api/ask", methods=["POST"])
    def ask():
        data = request.get_json()
        question = data.get("question", "")
        model = data.get("model", None)
        if not question:
            return jsonify({"error": "question is required"}), 400
        result = db.ask(question, model=model)
        return jsonify(result)

    @app.route("/api/search", methods=["POST"])
    def search():
        data = request.get_json()
        query = data.get("query", "")
        n = data.get("n", 5)
        if not query:
            return jsonify({"error": "query is required"}), 400
        results = db.search(query, n_results=n)
        return jsonify({"results": results})

    @app.route("/api/ingest", methods=["POST"])
    def ingest():
        data = request.get_json()
        source_type = data.get("type", "")

        if source_type == "meeting":
            db.ingest_meeting_report(data["path"])
        elif source_type == "pdf":
            db.ingest_pdf(data["path"], description=data.get("description", ""))
        elif source_type == "text":
            db.ingest_text(data["text"], data.get("source", "manual"))
        else:
            return jsonify({"error": "type must be meeting, pdf, or text"}), 400

        return jsonify({"status": "ok", "stats": db.stats()})

    @app.route("/api/stats", methods=["GET"])
    def stats():
        return jsonify(db.stats())

    @app.route("/")
    def index():
        if frontend_dir.exists():
            return send_from_directory(str(frontend_dir), "index.html")
        return _fallback_ui()

    @app.route("/<path:path>")
    def static_files(path):
        if frontend_dir.exists() and (frontend_dir / path).exists():
            return send_from_directory(str(frontend_dir), path)
        return _fallback_ui()

    return app


def _fallback_ui():
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Genie ChipDB</title>
<style>
body{font-family:sans-serif;max-width:800px;margin:0 auto;padding:40px;background:#f5f5f5}
h1{color:#1a1a2e}
#q{width:100%;padding:12px;font-size:16px;border:2px solid #3498db;border-radius:8px;box-sizing:border-box}
#ask{padding:12px 24px;background:#3498db;color:white;border:none;border-radius:8px;cursor:pointer;font-size:16px;margin-top:10px}
#result{margin-top:20px;padding:20px;background:white;border-radius:8px;white-space:pre-wrap;min-height:100px}
.source{font-size:0.85em;color:#888;margin-top:10px}
</style></head><body>
<h1>Genie ChipDB</h1>
<input id="q" placeholder="Ask about chip specifications..." autofocus>
<button id="ask" onclick="doAsk()">Ask</button>
<div id="result">Results will appear here...</div>
<script>
async function doAsk(){
  const q=document.getElementById('q').value;
  if(!q)return;
  document.getElementById('result').textContent='Thinking...';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  let html=d.answer+'\\n\\n';
  if(d.sources)d.sources.forEach(s=>{html+='<div class="source">Source: '+s.source_file+(s.page?' p.'+s.page:'')+'</div>';});
  document.getElementById('result').innerHTML=html;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')doAsk();});
</script></body></html>"""
