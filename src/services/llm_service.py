# src/services/llm_service.py
# OpenAI LLM + RAG (FINAL – Jetson CPU / API SAFE)

from openai import OpenAI
from src.config.settings import settings
import time
import re


class LLMService:
    """
    - Build RAG prompt
    - Call OpenAI Chat API
    - Voice-safe answer generation
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.OPENAI_LLM_MODEL

        self.MAX_OUTPUT_CHARS = 600
        self.MAX_SENTENCES = 5
        self.MAX_DOC_CHARS = 450   # cắt context cho voice
        self.RETRY = 2

    # ================== CONTEXT ==================

    def build_context(self, retrieved_docs: dict) -> str:
        docs = retrieved_docs.get("documents", [[]])[0]

        if not docs:
            return ""

        blocks = []
        for doc in docs:
            clean = doc.strip()
            if len(clean) > self.MAX_DOC_CHARS:
                clean = clean[: self.MAX_DOC_CHARS].rsplit(" ", 1)[0] + "..."
            blocks.append(clean)

        return "\n\n".join(blocks)

    # ================== PROMPT ==================

    def build_prompt(self, query: str, context: str) -> str:
        if context:
            return f"""
Bạn là trợ lý tư vấn tuyển sinh của Đại học FPT.
Bạn đang TRẢ LỜI BẰNG GIỌNG NÓI cho người nghe.

QUY TẮC:
- Trả lời tự nhiên, giống nói chuyện
- Không văn phong học thuật
- Không mở đầu bằng: "Theo thông tin", "Dựa trên dữ liệu"
- Không markdown, không ký hiệu

YÊU CẦU:
- Tối đa 5 câu
- Mỗi câu ngắn, dễ nghe
- Có thể dùng "mình", "bạn"

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI:
{query}

TRẢ LỜI:
"""
        else:
            # fallback khi không có RAG
            return f"""
Bạn là trợ lý tư vấn tuyển sinh của Đại học FPT.
Bạn đang nói chuyện trực tiếp với người dùng.

YÊU CẦU:
- Trả lời trung thực
- Nếu không chắc, nói rõ là thông tin tham khảo
- Ngắn gọn, dễ hiểu
- Tối đa 4 câu

CÂU HỎI:
{query}

TRẢ LỜI:
"""

    # ================== POST PROCESS ==================

    def post_process(self, text: str) -> str:
        text = re.sub(r"[#*_>`]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = sentences[: self.MAX_SENTENCES]

        result = " ".join(sentences)

        if len(result) > self.MAX_OUTPUT_CHARS:
            result = result[: self.MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + "..."

        return result.strip()

    # ================== GENERATE ==================

    def generate_answer(self, query: str, retrieved_docs: dict) -> str:
        context = self.build_context(retrieved_docs)
        prompt = self.build_prompt(query, context)

        for attempt in range(self.RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là trợ lý tư vấn tuyển sinh của Đại học FPT."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                )

                text = response.choices[0].message.content.strip()
                if not text:
                    raise ValueError("Empty LLM response")

                return self.post_process(text)

            except Exception as e:
                print(f"❌ OpenAI LLM error (attempt {attempt + 1}): {e}")
                time.sleep(0.4)

        return (
            "Mình chưa trả lời được ngay lúc này. "
            "Bạn có thể hỏi lại hoặc nói theo cách khác nhé."
        )
