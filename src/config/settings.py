# src/config/settings.py
# Centralized settings – FINAL
# Jetson Orin Nano | Local ASR + OpenAI ASR | Switchable

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ================= PROVIDER SWITCH =================
    ASR_PROVIDER = os.getenv("ASR_PROVIDER", "local")   # local | openai
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # gemini | openai

    # ================= LLM =================
    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

    # OpenAI LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    # ================= AUDIO =================
    SAMPLE_RATE = 16000

    # ⬅️ Buffer lớn để tránh ALSA underrun (Jetson)
    BLOCK_SIZE = 4096

    # Energy-based VAD
    SILENCE_THRESHOLD = 0.012
    SILENCE_DURATION = 0.6       # seconds
    MAX_RECORD_TIME = 15         # seconds

    INPUT_AUDIO_FILE = "assets/input.wav"
    OUTPUT_AUDIO_FILE = "assets/output.wav"

    # ================= MICROPHONE =================
    MIC_DEVICE_INDEX = None      # Auto-detect

    # ================= ASR =================
    # ---- Local Whisper / Faster-Whisper ----
    ASR_MODEL = "base"           # base is safe, small dễ OOM
    ASR_DEVICE = "cuda"
    ASR_COMPUTE_TYPE = "float16"

    WHISPER_VAD = True
    WHISPER_VAD_MIN_SILENCE_MS = 500

    # ---- OpenAI ASR ----
    OPENAI_ASR_MODEL = os.getenv("OPENAI_ASR_MODEL", "whisper-1")

    # ================= TTS =================
    TTS_VOICE = "vi-VN-HoaiMyNeural"

    # ================= RETRIEVAL / RAG =================
    VECTOR_DB_DIR = "vector_db"
    COLLECTION_NAME = "fpt_university"
    RETRIEVAL_SCORE_THRESHOLD = 0.15

    # ================= VOICE UX =================
    MAX_VOICE_CHARS = 600
    MAX_VOICE_SENTENCES = 5

    # ================= DEMO / DEBUG =================
    DEMO_MODE = False
    LOG_LATENCY = True


settings = Settings()
