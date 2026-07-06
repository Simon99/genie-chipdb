from __future__ import annotations

import json
import hashlib
from pathlib import Path

import requests
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from genie_core.llm import LMStudioClient

# Approximate chunk size in characters; chunks are cut at sentence/newline boundaries.
CHUNK_SIZE = 600

# Sentence/paragraph boundary characters, in rough order of preference.
_BOUNDARY_CHARS = ("\n", "。", "．", ".", "!", "?", "！", "？", ";", "；")


class LMStudioEmbeddingFunction(EmbeddingFunction):
    """chromadb EmbeddingFunction backed by LM Studio's /v1/embeddings endpoint.

    Use with a multilingual embedding model (e.g. bge-m3) loaded in LM Studio.
    Note: base_url is expected to already include the /v1 prefix
    (e.g. "http://localhost:1234/v1"), matching LMStudioClient.
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model: str = "text-embedding-bge-m3",
                 timeout=(5, 120)):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        resp = requests.post(
            "%s/embeddings" % self.base_url,
            json={"model": self.model, "input": list(input)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]

    @staticmethod
    def name() -> str:
        return "lmstudio"


class ChipDatabase:
    """ChromaDB-backed chip information database with semantic search.

    embedding: "default" uses chromadb's built-in embedding (English MiniLM);
    "lmstudio" uses LMStudioEmbeddingFunction against lm_studio_url (recommended
    for Chinese content, e.g. with bge-m3).

    WARNING: switching embedding on an existing collection produces mixed/invalid
    vectors. To change embedding, delete the collection (or data_dir) and
    re-ingest everything.
    """

    def __init__(self, data_dir: str = "./chroma_data",
                 lm_studio_url: str = "http://localhost:1234/v1",
                 embedding: str = "default",
                 embedding_model: str = "text-embedding-bge-m3"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.data_dir))
        collection_kwargs = {
            "name": "chips",
            "metadata": {"hnsw:space": "cosine"},
        }
        if embedding == "lmstudio":
            collection_kwargs["embedding_function"] = LMStudioEmbeddingFunction(
                base_url=lm_studio_url, model=embedding_model,
            )
        elif embedding != "default":
            raise ValueError("embedding must be 'default' or 'lmstudio', got: %r" % embedding)
        self.collection = self.client.get_or_create_collection(**collection_kwargs)
        self.llm = LMStudioClient(base_url=lm_studio_url)

    def ingest_meeting_report(self, report_path: str):
        """Import structured meeting report (from genie-meeting) into the database."""
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        for i, topic in enumerate(report.get("topics", [])):
            text = _topic_to_text(topic)
            # Include the topic index so same-titled topics do not overwrite each other.
            doc_id = _make_id(report_path, "topic%d::%s" % (i, text))
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

    def ingest_pdf(self, pdf_path: str, description: str = "") -> dict:
        """Import a PDF document (supplier datasheet, test report, etc.).

        Returns {"pages": total, "failed_pages": [page numbers where vision failed],
        "chunks": number of chunks ingested}.
        """
        from genie_core.pdf.split import split_pdf_to_images

        failed_pages = []
        chunk_count = 0

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = split_pdf_to_images(pdf_path, tmpdir)

            for page in pages:
                try:
                    text = self.llm.vision(
                        prompt="Extract all text and data from this document page as plain text.",
                        image_path=page["path"],
                        temperature=0.1,
                    )
                except Exception:
                    failed_pages.append(page["page"])
                    continue

                for ci, chunk in enumerate(_chunk_text(text)):
                    doc_id = _make_id(pdf_path, "p%d-c%d::%s" % (page["page"], ci, chunk))
                    metadata = {
                        "source_type": "pdf",
                        "source_file": Path(pdf_path).name,
                        "page": page["page"],
                        "chunk": ci,
                        "description": description,
                    }

                    self.collection.upsert(
                        ids=[doc_id],
                        documents=[chunk],
                        metadatas=[metadata],
                    )
                    chunk_count += 1

        return {
            "pages": len(pages),
            "failed_pages": failed_pages,
            "chunks": chunk_count,
        }

    def ingest_text(self, text: str, source_file: str, source_type: str = "manual", metadata: dict = None) -> int:
        """Import raw text content, split into ~600-char chunks. Returns chunk count."""
        chunks = _chunk_text(text)
        for ci, chunk in enumerate(chunks):
            doc_id = _make_id(source_file, "c%d::%s" % (ci, chunk))
            meta = {
                "source_type": source_type,
                "source_file": source_file,
                "chunk": ci,
            }
            if metadata:
                meta.update(metadata)

            self.collection.upsert(
                ids=[doc_id],
                documents=[chunk],
                metadatas=[meta],
            )
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search across all chip data."""
        count = self.collection.count()
        n = min(max(1, int(n_results)), max(count, 1))
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
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
        seen = set()
        for r in results:
            context_parts.append(r["text"])
            meta = r.get("metadata") or {}
            source = {
                "source_file": meta.get("source_file", ""),
                "source_type": meta.get("source_type", ""),
            }
            page = meta.get("page")
            if page not in (None, ""):
                source["page"] = page
            key = (source["source_file"], source.get("page"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)

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


def _make_id(source: str, text: str) -> str:
    """Stable doc id from source identifier + full content text (sha256)."""
    raw = "%s::%s" % (source, text)
    return hashlib.sha256(raw.encode()).hexdigest()


def _chunk_text(text: str, target: int = CHUNK_SIZE) -> list[str]:
    """Split text into ~target-char chunks, cutting at sentence/newline boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + target, n)
        if end < n:
            window = text[start:end]
            cut = -1
            for ch in _BOUNDARY_CHARS:
                idx = window.rfind(ch)
                if idx > cut:
                    cut = idx
            # Only honor the boundary if it leaves a reasonably sized chunk.
            if cut >= target // 3:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _topic_to_text(topic: dict) -> str:
    parts = [topic.get("title", "")]
    parts.append(topic.get("summary", ""))
    for p in topic.get("key_points", []):
        parts.append("- %s" % p)
    for d in topic.get("decisions", []):
        parts.append("Decision: %s" % d)
    return "\n".join(parts)
