from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

from .database import ChipDatabase


def create_app(data_dir: str = "./chroma_data",
               lm_studio_url: str = "http://localhost:1234/v1",
               ingest_root: str = None) -> Flask:
    """Create the ChipDB Flask app.

    ingest_root: directory whitelist for path-based ingest (/api/ingest with
    type meeting/pdf). Paths resolving outside it are rejected with 403.
    Defaults to the user's home directory.
    """
    app = Flask(__name__, static_folder=None)

    db = ChipDatabase(data_dir=data_dir, lm_studio_url=lm_studio_url)
    root = Path(ingest_root).resolve() if ingest_root else Path.home().resolve()

    frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"

    def _check_ingest_path(raw_path):
        """Return (resolved_path, error_response). Whitelist check against root."""
        resolved = Path(raw_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None, (jsonify({"error": "path is outside the allowed ingest root"}), 403)
        return resolved, None

    @app.route("/api/ask", methods=["POST"])
    def ask():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "")
        model = data.get("model", None)
        if not question:
            return jsonify({"error": "question is required"}), 400
        result = db.ask(question, model=model)
        return jsonify(result)

    @app.route("/api/search", methods=["POST"])
    def search():
        data = request.get_json(silent=True) or {}
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "query is required"}), 400
        try:
            n = int(data.get("n", 5))
        except (TypeError, ValueError):
            return jsonify({"error": "n must be an integer"}), 400
        n = max(1, min(n, 50))
        results = db.search(query, n_results=n)
        return jsonify({"results": results})

    @app.route("/api/ingest", methods=["POST"])
    def ingest():
        data = request.get_json(silent=True) or {}
        source_type = data.get("type", "")

        if source_type == "meeting":
            if not data.get("path"):
                return jsonify({"error": "path is required for type=meeting"}), 400
            path, err = _check_ingest_path(data["path"])
            if err:
                return err
            db.ingest_meeting_report(str(path))
        elif source_type == "pdf":
            if not data.get("path"):
                return jsonify({"error": "path is required for type=pdf"}), 400
            path, err = _check_ingest_path(data["path"])
            if err:
                return err
            result = db.ingest_pdf(str(path), description=data.get("description", ""))
            return jsonify({
                "status": "ok",
                "pages": result["pages"],
                "failed_pages": result["failed_pages"],
                "chunks": result["chunks"],
                "stats": db.stats(),
            })
        elif source_type == "text":
            if not data.get("text"):
                return jsonify({"error": "text is required for type=text"}), 400
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
  const result=document.getElementById('result');
  result.textContent='Thinking...';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  result.textContent='';
  const answer=document.createElement('div');
  answer.textContent=d.error?('Error: '+d.error):(d.answer||'');
  result.appendChild(answer);
  if(d.sources){
    d.sources.forEach(s=>{
      const el=document.createElement('div');
      el.className='source';
      el.textContent='Source: '+s.source_file+(s.page?' p.'+s.page:'');
      result.appendChild(el);
    });
  }
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')doAsk();});
</script></body></html>"""
