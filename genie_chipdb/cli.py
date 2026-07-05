from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Genie ChipDB - Chip information database")
    sub = parser.add_subparsers(dest="command")

    # serve command
    serve = sub.add_parser("serve", help="Start web server")
    serve.add_argument("--port", type=int, default=5100)
    serve.add_argument("--data-dir", default="./chroma_data")
    serve.add_argument("--url", default="http://localhost:1234/v1")

    # mcp command
    mcp = sub.add_parser("mcp", help="Start MCP server (stdio)")
    mcp.add_argument("--data-dir", default="./chroma_data")
    mcp.add_argument("--url", default="http://localhost:1234/v1")

    # ingest command
    ing = sub.add_parser("ingest", help="Ingest data into database")
    ing.add_argument("path", help="Path to file (meeting report JSON, PDF, or text file)")
    ing.add_argument("--type", choices=["meeting", "pdf", "text"], default="pdf")
    ing.add_argument("--description", default="")
    ing.add_argument("--data-dir", default="./chroma_data")
    ing.add_argument("--url", default="http://localhost:1234/v1")

    # ask command
    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--data-dir", default="./chroma_data")
    ask.add_argument("--url", default="http://localhost:1234/v1")

    args = parser.parse_args()

    if args.command == "serve":
        from .server import create_app
        app = create_app(data_dir=args.data_dir, lm_studio_url=args.url)
        print("ChipDB server starting on http://localhost:%d" % args.port)
        app.run(host="0.0.0.0", port=args.port, debug=False)

    elif args.command == "mcp":
        from .mcp_server import run_stdio
        run_stdio(data_dir=args.data_dir, lm_studio_url=args.url)

    elif args.command == "ingest":
        from .database import ChipDatabase
        db = ChipDatabase(data_dir=args.data_dir, lm_studio_url=args.url)
        if args.type == "meeting":
            db.ingest_meeting_report(args.path)
        elif args.type == "pdf":
            db.ingest_pdf(args.path, description=args.description)
        elif args.type == "text":
            text = open(args.path, "r", encoding="utf-8").read()
            db.ingest_text(text, args.path)
        print("Ingested: %s" % args.path)
        print("Stats: %s" % db.stats())

    elif args.command == "ask":
        from .database import ChipDatabase
        db = ChipDatabase(data_dir=args.data_dir, lm_studio_url=args.url)
        result = db.ask(" ".join(args.question))
        print(result["answer"])
        print("\nSources:")
        for s in result.get("sources", []):
            print("  - %s %s" % (s.get("source_file", ""), "p.%s" % s["page"] if s.get("page") else ""))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
