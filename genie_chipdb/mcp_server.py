from __future__ import annotations

"""MCP (Model Context Protocol) server for ChipDB.

Exposes chip database as tools that other AI agents can call.
Run with: python -m genie_chipdb.mcp_server [--data-dir ./chroma_data] [--port 3100]
"""

import json
import sys
from pathlib import Path

from .database import ChipDatabase


def handle_request(db: ChipDatabase, request: dict) -> dict:
    """Handle a JSON-RPC style MCP request. Returns None for notifications."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    # Notifications (no id, method namespaced under notifications/) must not
    # be answered per JSON-RPC / MCP spec.
    if req_id is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return _response(req_id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "genie-chipdb", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        })

    if method == "tools/list":
        return _response(req_id, {"tools": [
            {
                "name": "chip_search",
                "description": "Search chip database for relevant information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query about chip specs, features, test results"},
                        "n": {"type": "integer", "description": "Number of results (1-50)", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "chip_ask",
                "description": "Ask a question about chips and get an AI-generated answer with sources",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Question about chip specifications or features"},
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "chip_ingest",
                "description": "Add new chip information to the database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Chip information text to add"},
                        "source": {"type": "string", "description": "Source identifier"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "chip_stats",
                "description": "Get database statistics",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]})

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name == "chip_search":
            query = args.get("query", "")
            if not isinstance(query, str) or not query.strip():
                return _tool_error(req_id, "query must be a non-empty string")
            try:
                n = int(args.get("n", 5))
            except (TypeError, ValueError):
                return _tool_error(req_id, "n must be an integer")
            n = max(1, min(n, 50))
            results = db.search(query, n_results=n)
            return _response(req_id, {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False, indent=2)}]})

        if tool_name == "chip_ask":
            question = args.get("question", "")
            if not isinstance(question, str) or not question.strip():
                return _tool_error(req_id, "question must be a non-empty string")
            result = db.ask(question)
            return _response(req_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]})

        if tool_name == "chip_ingest":
            text = args.get("text", "")
            if not isinstance(text, str) or not text.strip():
                return _tool_error(req_id, "text must be a non-empty string")
            chunks = db.ingest_text(text, args.get("source", "mcp"))
            return _response(req_id, {"content": [{"type": "text", "text": "Ingested successfully (%d chunks)" % chunks}]})

        if tool_name == "chip_stats":
            stats = db.stats()
            return _response(req_id, {"content": [{"type": "text", "text": json.dumps(stats)}]})

        return _error(req_id, -32601, "Unknown tool: %s" % tool_name)

    return _error(req_id, -32601, "Unknown method: %s" % method)


def _response(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _tool_error(req_id, message):
    """MCP tool-level error: a successful JSON-RPC response flagged isError."""
    return _response(req_id, {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    })


def run_stdio(data_dir: str = "./chroma_data", lm_studio_url: str = "http://localhost:1234/v1"):
    """Run MCP server over stdin/stdout (standard MCP transport)."""
    db = ChipDatabase(data_dir=data_dir, lm_studio_url=lm_studio_url)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            _write(_error(None, -32700, "Parse error"))
            continue

        if not isinstance(request, dict):
            _write(_error(None, -32600, "Invalid request: expected object"))
            continue

        try:
            response = handle_request(db, request)
        except Exception as e:
            # Never let a tool exception (LM Studio down, bad args, ChromaDB IO,
            # ...) kill the server loop.
            response = _error(request.get("id"), -32603, str(e))

        if response is not None:
            _write(response)


def _write(response: dict):
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./chroma_data")
    parser.add_argument("--url", default="http://localhost:1234/v1")
    args = parser.parse_args()
    run_stdio(data_dir=args.data_dir, lm_studio_url=args.url)
