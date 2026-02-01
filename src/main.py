# src/main.py
# Voice Chatbot – FINAL VERSION (Jetson SAFE)
# OpenAI ASR + Gemini LLM

import os
import time

# 🔒 SAFE FOR JETSON / CPU MODE
os.environ["ORT_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from src.services.voice_service import VoiceService
from src.services.retrieval_service import RetrievalService
from src.services.llm_service import LLMService
from src.utils.text_normalizer import normalize_text


# -------- STATE --------
IDLE = "idle"
ACTIVE = "active"


# -------- INTENT KEYWORDS --------
START_KEYWORDS = [
    "bat dau", "bắt đầu",
    "bat dau tu van", "bắt đầu tư vấn",
    "tu van", "tư vấn",
    "hoi thong tin", "hỏi thông tin"
]

EXIT_KEYWORDS = [
    "thoát", "thoat",
    "kết thúc", "ket thuc",
    "dừng tư vấn", "ngừng tư vấn",
    "bye", "tạm biệt"
]

THANK_KEYWORDS = [
    "cảm ơn", "cam on",
    "thanks", "thank you",
    "ok cảm ơn", "ok cam on"
]


def contains_any(text: str, keywords: list) -> bool:
    return any(k in text for k in keywords)


def is_noise(text: str) -> bool:
    if not text:
        return True
    text = text.strip().lower()
    return len(text) < 3 or text in ["ừ", "ừm", "à", "ờ", "uh", "um"]


def run_voice_chat():
    print("🎙️ FPT AI Voice Chatbot (Jetson – FINAL)")
    print("👉 Nói: 'bắt đầu tư vấn' để bắt đầu")
    print("👉 Nói: 'dừng' để ngắt trả lời")
    print("👉 Nói: 'kết thúc', 'thoát' hoặc 'cảm ơn' để nghỉ\n")

    voice = VoiceService()
    retrieval = RetrievalService()
    llm = LLMService()

    state = IDLE

    while True:
        # ================= IDLE MODE =================
        if state == IDLE:
            user_text = voice.listen()
            if not user_text:
                continue

            normalized = normalize_text(user_text)
            print(f"👂 (idle) Nghe: {normalized}")

            if contains_any(normalized, START_KEYWORDS):
                state = ACTIVE
                voice.speak("Mình rất vui được hỗ trợ bạn. Mời bạn đặt câu hỏi.")
                print("🟢 Chuyển sang ACTIVE\n")
                time.sleep(0.5)
                continue

            if contains_any(normalized, EXIT_KEYWORDS):
                voice.speak("Tạm biệt bạn. Hẹn gặp lại.")
                break

            continue

        # ================= ACTIVE MODE =================
        user_text = voice.listen()
        if not user_text:
            continue

        normalized = normalize_text(user_text)
        print(f"👂 (active) Nghe: {normalized}")

        # ---- INTERRUPT ----
        if "dừng" in normalized:
            voice.stop()
            voice.speak("Mình đã dừng. Bạn có thể hỏi câu khác.")
            continue

        # ---- EXIT / THANK ----
        if (
            contains_any(normalized, EXIT_KEYWORDS)
            or contains_any(normalized, THANK_KEYWORDS)
        ):
            voice.stop()
            voice.speak(
                "Mình rất vui vì đã được hỗ trợ bạn. Khi cần tư vấn tiếp, hãy nói bắt đầu tư vấn nhé."
            )
            state = IDLE
            print("🔴 Quay về IDLE\n")
            continue

        # ---- NOISE ----
        if is_noise(normalized):
            continue

        # ---- RETRIEVAL ----
        retrieved = retrieval.retrieve(query=normalized, top_k=3)

        # ---- LLM ----
        try:
            answer = llm.generate_answer(
                query=normalized,
                retrieved_docs=retrieved
            )
        except Exception as e:
            print("❌ LLM error:", e)
            answer = "Mình chưa trả lời được ngay lúc này."

        print("\n🤖 Bot:", answer)
        voice.speak(answer)
        time.sleep(0.4)
        print("-" * 60)


if __name__ == "__main__":
    run_voice_chat()
