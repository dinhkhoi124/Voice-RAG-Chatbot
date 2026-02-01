# src/rag/rag_system.py
# FINAL – Stable & Fast for Jetson Orin Nano

import chromadb
from chromadb.utils import embedding_functions
import json
import os
import hashlib
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch

from src.config.settings import settings


# ================= CONFIG =================

CRAWLED_DATA_FILE = os.path.join("data", "fpt_data.json")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RAGSystem:
    def __init__(self):
        # ⚠️ ÉP CPU cho ổn định Jetson
        device = "cpu"
        print(f"🔄 Loading Embedding Model on {device.upper()}")

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME,
            device=device
        )

        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_DIR
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=180,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        print("✅ RAG System ready")

    # ================= SPLIT =================

    def split_text_smart(self, text: str) -> List[str]:
        if not text:
            return []
        if len(text) < 200:
            return [text]
        return self.text_splitter.split_text(text)

    # ================= INDEX =================

    def index_documents(self):
        if not os.path.exists(CRAWLED_DATA_FILE):
            print("❌ Không tìm thấy file dữ liệu.")
            return

        with open(CRAWLED_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            existing_ids = set(self.collection.get()["ids"])
        except Exception:
            existing_ids = set()

        seen_ids = set()
        batch_texts, batch_ids, batch_metadatas = [], [], []
        BATCH_SIZE = 20
        count_new = 0

        print(f"🔍 Indexing {len(data)} documents...")

        for item in data:
            raw = item.get("content", "")
            if item.get("description"):
                raw = f"Tóm tắt: {item['description']}\n{raw}"

            chunks = self.split_text_smart(raw)

            for idx, chunk in enumerate(chunks):
                final_chunk = (
                    f"Tiêu đề: {item.get('title', '')}\n"
                    f"Nội dung: {chunk}"
                )

                # ❗ KHÔNG normalize khi index
                chunk_id = hashlib.md5(final_chunk.encode("utf-8")).hexdigest()

                if chunk_id in existing_ids or chunk_id in seen_ids:
                    continue

                seen_ids.add(chunk_id)

                meta = {
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "doc_type": item.get("type", "general"),
                    "available": True,
                    "chunk_index": idx
                }

                batch_texts.append(final_chunk)
                batch_ids.append(chunk_id)
                batch_metadatas.append(meta)
                count_new += 1

                print(f"Indexing chunk {count_new}", end="\r")

                if len(batch_texts) >= BATCH_SIZE:
                    self._save_batch(batch_texts, batch_ids, batch_metadatas)
                    batch_texts, batch_ids, batch_metadatas = [], [], []

        if batch_texts:
            self._save_batch(batch_texts, batch_ids, batch_metadatas)

        print(f"\n🎉 Index xong: {count_new} chunks mới")

    def _save_batch(self, texts, ids, metas):
        try:
            self.collection.add(
                documents=texts,
                ids=ids,
                metadatas=metas
            )
            print(f"   -> saved {len(texts)}")
        except Exception as e:
            print(f"\n⚠️ Skip batch: {e}")


# ================= RUN =================

if __name__ == "__main__":
    rag = RAGSystem()
    rag.index_documents()
