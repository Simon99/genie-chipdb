from __future__ import annotations

import json
import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings

from genie_core.llm import LMStudioClient


class ChipDatabase:
    """ChromaDB-backed chip information database with semantic search."""

    def __init__(self, data_dir: str = "./chroma_data", lm_studio_url: str = "http://localhost:1234/v1"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.data_dir))
        self.collection = self.client.get_or_create_collection(
            name="chips",
            metadata={"hnsw:space": "cosine"},
        )
        self.llm = LMStudioClient(base_url=lm_studio_url)

    def ingest_meeting_report(self, report_path: str):
        """Import structured meeting report (from genie-meeting) into the database."""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        for topic in report.get("topics", []):
            doc_id = _make_id(report_path, topic.get("title", ""))
            text = _topic_to_text(topic)
            metadata = {
                "source_type": "meeting",
                "source_file": Path(report_path).name,
                "topic": topic.get("title", ""),
            }

            sources = topic.get("sources", [])
            if sources:
                metadata["sources_json"] = json.dumps(sources, ensure_ascii=False)

            self.collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
            )

    def ingest_pdf(self, pdf_path: str, description: str = ""):
        """Import a PDF document (supplier datasheet, test report, etc.)."""
        from genie_core.pdf.split import split_pdf_to_images
        from genie_core.llm import LMStudioClient

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = split_pdf_to_images(pdf_path, tmpdir)
            vision_llm = LMStudioClient(
                base_url=self.llm.base_url.replace("/v1", "/v1"),
            )

            for page in pages:
                try:
                    text = vision_llm.vision(
                        prompt="Extract all text and data from this document page as plain text.",
                        image_path=page["path"],
                        temperature=0.1,
                    )
                except Exception:
                    continue

                doc_id = _make_id(pdf_path, str(page["page"]))
                metadata = {
                    "source_type": "pdf",
                    "source_file": Path(pdf_path).name,
                    "page": page["page"],
                    "description": description,
                }

                self.collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[metadata],
                )

    def ingest_text(self, text: str, source_file: str, source_type: str = "manual", metadata: dict = None):
        """Import raw text content."""
        doc_id = _make_id(source_file, text[:50])
        meta = {
            "source_type": source_type,
            "source_file": source_file,
        }
        if metadata:
            meta.update(metadata)

        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search across all chip data."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })
        return items

    def ask(self, question: str, model: str = None) -> dict:
        """Answer a question using RAG (retrieve + generate)."""
        results = self.search(question, n_results=5)

        context_parts = []
        sources = []
        for r in results:
            context_parts.append(r["text"])
            sources.append({
                "source_file": r["metadata"].get("source_file", ""),
                "source_type": r["metadata"].get("source_type", ""),
                "page": r["metadata"].get("page", ""),
            })

        context = "\n\n---\n\n".join(context_parts)

        llm = LMStudioClient(base_url=self.llm.base_url, model=model) if model else self.llm
        answer = llm.complete(
            prompt="Based on the following chip information, answer this question: %s\n\nContext:\n%s" % (
                question, context),
            system="You are a chip/semiconductor expert. Answer based ONLY on the provided context. "
                   "Cite sources. If the answer is not in the context, say so.",
            temperature=0.2,
        )

        return {
            "answer": answer,
            "sources": sources,
            "query": question,
        }

    def stats(self) -> dict:
        """Get database statistics."""
        return {
            "total_documents": self.collection.count(),
            "data_dir": str(self.data_dir),
        }


def _make_id(source: str, key: str) -> str:
    raw = "%s::%s" % (source, key)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _topic_to_text(topic: dict) -> str:
    parts = [topic.get("title", "")]
    parts.append(topic.get("summary", ""))
    for p in topic.get("key_points", []):
        parts.append("- %s" % p)
    for d in topic.get("decisions", []):
        parts.append("Decision: %s" % d)
    return "\n".join(parts)
