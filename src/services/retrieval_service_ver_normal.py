# ver 10/1
from chromadb import PersistentClient
from src.config.settings import settings
from src.utils.text_normalizer import normalize_text


class RetrievalService:
    """
    PRO Retrieval Service:
    - Hybrid Search (Semantic + Keyword-lite)
    - Metadata / Intent Boost
    - Lightweight Re-ranking
    """

    # Keyword map để detect intent
    INTENT_KEYWORDS = {
        "hoc_phi": ["học phí", "chi phí", "bao nhiêu tiền"],
        "tuyen_sinh": ["tuyển sinh", "xét tuyển", "điều kiện"],
        "nganh_ai": ["trí tuệ nhân tạo", "ai", "ngành ai"]
    }

    def __init__(self):
        self.client = PersistentClient(path="vector_db")
        self.collection = self.client.get_or_create_collection(
            name="fpt_university"
        )
        self.score_threshold = settings.RETRIEVAL_SCORE_THRESHOLD

    # ------------------ INTENT DETECTION ------------------

    def _detect_intent(self, query: str):
        query = query.lower()
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    return intent
        return None

    # ------------------ KEYWORD / METADATA BOOST ------------------

    def _boost_score(self, doc: str, meta: dict, intent: str):
        boost = 0.0
        doc_lower = doc.lower()

        if intent == "hoc_phi":
            if "học phí" in doc_lower or "chi phí" in doc_lower:
                boost += 0.25
            if meta.get("type") == "tuition":
                boost += 0.30

        elif intent == "tuyen_sinh":
            if meta.get("type") == "admission":
                boost += 0.20

        elif intent == "nganh_ai":
            if "trí tuệ nhân tạo" in doc_lower or "ai" in doc_lower:
                boost += 0.15

        return boost

    # ------------------ MAIN RETRIEVE ------------------

    def retrieve(self, query: str, top_k: int = 3):
        query = normalize_text(query)
        intent = self._detect_intent(query)

        # 1️⃣ Lấy nhiều kết quả để rerank
        results = self.collection.query(
            query_texts=[query],
            n_results=10,
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"]:
            return self._empty_result()

        ranked = []

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            semantic_score = 1 - dist

            if semantic_score < self.score_threshold:
                continue

            boost = self._boost_score(doc, meta, intent)
            final_score = semantic_score + boost

            ranked.append({
                "doc": doc,
                "meta": meta,
                "score": round(final_score, 3)
            })

        # 2️⃣ Re-rank theo score tổng hợp
        ranked.sort(key=lambda x: x["score"], reverse=True)

        # 3️⃣ Trả về format chuẩn
        return {
            "documents": [[r["doc"] for r in ranked[:top_k]]],
            "metadatas": [[r["meta"] for r in ranked[:top_k]]],
            "scores": [[r["score"] for r in ranked[:top_k]]],
        }

    def _empty_result(self):
        return {
            "documents": [[]],
            "metadatas": [[]],
            "scores": [[]],
        }


# ver gốc
# from chromadb import PersistentClient
# from src.config.settings import settings
# from src.utils.text_normalizer import normalize_text

# class RetrievalService:
#     def __init__(self):
#         self.client = PersistentClient(path="vector_db")
#         self.collection = self.client.get_or_create_collection(
#             name="fpt_university"
#         )
#         self.score_threshold = settings.RETRIEVAL_SCORE_THRESHOLD

#     def retrieve(self, query: str, top_k: int = 3):
#         query = normalize_text(query)

#         results = self.collection.query(
#             query_texts=[query],
#             n_results=top_k,
#             include=["documents", "metadatas", "distances"]
#         )

#         return self._apply_threshold(results)

#     def _apply_threshold(self, results):
#         filtered = {
#             "documents": [[]],
#             "metadatas": [[]],
#             "scores": [[]]
#         }

#         if not results["documents"]:
#             return filtered

#         for doc, meta, dist in zip(
#             results["documents"][0],
#             results["metadatas"][0],
#             results["distances"][0]
#         ):
#             score = 1 - dist
#             if score >= self.score_threshold:
#                 filtered["documents"][0].append(doc)
#                 filtered["metadatas"][0].append(meta)
#                 filtered["scores"][0].append(score)

#         return filtered
