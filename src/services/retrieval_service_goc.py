#  Hybrid search, threshold

from src.rag.rag_system import RAGSystem
from src.config.settings import settings


class RetrievalService:
    """
    Service xử lý:
    - Retrieval
    - Score threshold
    - Chuẩn bị context cho LLM
    """

    def __init__(self):
        self.rag = RAGSystem()
        self.score_threshold = settings.RETRIEVAL_SCORE_THRESHOLD

    def search(self, query: str, top_k: int = 5) -> dict:
        """
        Thực hiện search + filter theo score threshold
        """
        raw_results = self.rag.search(query, top_k=top_k)

        if not raw_results.get("documents") or not raw_results["documents"][0]:
            return self._empty_results()

        filtered_docs = []
        filtered_metas = []

        distances = raw_results.get("distances", [[]])[0]

        for idx, distance in enumerate(distances):
            # cosine distance: càng nhỏ càng tốt
            if distance <= self.score_threshold:
                filtered_docs.append(raw_results["documents"][0][idx])
                filtered_metas.append(raw_results["metadatas"][0][idx])

        if not filtered_docs:
            return self._empty_results()

        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_metas]
        }

    @staticmethod
    def _empty_results() -> dict:
        return {
            "documents": [[]],
            "metadatas": [[]]
        }
