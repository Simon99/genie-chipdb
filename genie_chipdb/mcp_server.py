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
    """Handle a JSON-RPC style MCP request."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

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
                        "n": {"type": "integer", "description": "Number of results", "default": 5},
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
            results = db.search(args.get("query", ""), n_results=args.get("n", 5))
            return _response(req_id, {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False, indent=2)}]})

        if tool_name == "chip_ask":
            result = db.ask(args.get("question", ""))
            return _response(req_id, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]})

        if tool_name == "chip_ingest":
            db.ingest_text(args.get("text", ""), args.get("source", "mcp"))
            return _response(req_id, {"content": [{"type": "text", "text": "Ingested successfully"}]})

        if tool_name == "chip_stats":
            stats = db.stats()
            return _response(req_id, {"content": [{"type": "text", "text": json.dumps(stats)}]})

        return _error(req_id, -32601, "Unknown tool: %s" % tool_name)

    return _error(req_id, -32601, "Unknown method: %s" % method)


def _response(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def run_stdio(data_dir: str = "./chroma_data", lm_studio_url: str = "http://localhost:1234/v1"):
    """Run MCP server over stdin/stdout (standard MCP transport)."""
    db = ChipDatabase(data_dir=data_dir, lm_studio_url=lm_studio_url)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(db, request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            err = _error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./chroma_data")
    parser.add_argument("--url", default="http://localhost:1234/v1")
    args = parser.parse_args()
    run_stdio(data_dir=args.data_dir, lm_studio_url=args.url)
