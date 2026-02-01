# src/services/retrieval_service.py
# ChromaDB RAG – FINAL (Jetson SAFE, NO CUDA CONFLICT)

import re

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from src.config.settings import settings
from src.utils.text_normalizer import normalize_text


# ================= CONFIG =================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TUITION_KEYWORDS = [
    "học phí", "hoc phi", "bao nhiêu tiền",
    "chi phí", "đóng tiền", "phí"
]

MONEY_PATTERN = re.compile(
    r"\b(\d+(\.\d+)?\s?(triệu|tr|vnd|vnđ|đ))\b",
    re.IGNORECASE
)


class RetrievalService:
    """
    - Semantic search (Chroma)
    - Embedding CPU-only (NO CUDA TOUCH)
    - Optimized for Jetson voice loop
    """

    def __init__(self):
        print("🔎 Retrieval embedding device: CPU (explicit)")

        # ❗ CPU ONLY – tuyệt đối không init CUDA
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME,
            device="cpu"
        )

        self.client = PersistentClient(
            path=settings.VECTOR_DB_DIR
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        self.score_threshold = settings.RETRIEVAL_SCORE_THRESHOLD

    # ================= PUBLIC =================

    def retrieve(self, query: str, top_k: int = 5):
        query_norm = normalize_text(query)

        # ASR alias fix
        query_norm = (
            query_norm
            .replace("mpt", "fpt")
            .replace("mbt", "fpt")
        )

        is_tuition_query = self._detect_tuition_intent(query_norm)

        results = self.collection.query(
            query_texts=[query_norm],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )

        return self._rerank_results(
            query_norm,
            results,
            is_tuition_query,
            top_k
        )

    # ================= INTENT =================

    def _detect_tuition_intent(self, query: str) -> bool:
        return any(k in query for k in TUITION_KEYWORDS)

    # ================= RERANK =================

    def _rerank_results(self, query, results, is_tuition_query, top_k):
        if not results.get("documents") or not results["documents"][0]:
            return self._empty_result()

        candidates = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = max(0.0, 1.0 / (1.0 + dist))

            # ---- intent boost ----
            if is_tuition_query:
                if meta.get("doc_type") == "tuition":
                    score += 0.30
                elif meta.get("doc_type") == "tuition_note":
                    score += 0.15

            # ---- money signal ----
            if MONEY_PATTERN.search(doc):
                score += 0.10

            # ---- availability ----
            if meta.get("available") is True:
                score += 0.05

            # ---- keyword soft boost ----
            doc_norm = normalize_text(doc)
            hits = sum(1 for k in TUITION_KEYWORDS if k in doc_norm)
            score += min(hits * 0.03, 0.09)

            if score < self.score_threshold:
                continue

            candidates.append({
                "document": self._trim_doc(doc),
                "metadata": meta,
                "score": round(score, 4)
            })

            if len(candidates) >= top_k * 2:
                break

        # ================= FALLBACK MỀM (FIX QUAN TRỌNG) =================
        if not candidates:
            # Lấy doc tốt nhất dù score thấp để LLM vẫn có context
            doc = results["documents"][0][0]
            meta = results["metadatas"][0][0]

            return {
                "documents": [[self._trim_doc(doc)]],
                "metadatas": [[meta]],
                "scores": [[0.01]]
            }

        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:top_k]

        return {
            "documents": [[c["document"] for c in candidates]],
            "metadatas": [[c["metadata"] for c in candidates]],
            "scores": [[c["score"] for c in candidates]]
        }

    # ================= UTIL =================

    def _trim_doc(self, doc: str, max_chars: int = 800):
        if len(doc) <= max_chars:
            return doc.strip()

        match = MONEY_PATTERN.search(doc)
        if match:
            start = max(0, match.start() - 200)
            end = min(len(doc), match.end() + 300)
            return doc[start:end].strip()

        return doc[:max_chars].strip() + "..."

    def _empty_result(self):
        return {
            "documents": [[]],
            "metadatas": [[]],
            "scores": [[]]
        }
