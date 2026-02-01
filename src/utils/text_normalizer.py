# src/utils/text_normalizer.py
# Normalize text for VOICE + RAG (Vietnamese)

import re
import unicodedata


# ===== ABBREVIATION =====
ABBREVIATION_MAP = {
    "ai": "trí tuệ nhân tạo",
    "cntt": "công nghệ thông tin",
    "it": "công nghệ thông tin",
    "ml": "machine learning",
    "dl": "deep learning",
}

# ===== ASR ERROR CORRECTION (VOICE COMMAND) =====
COMMAND_FIX = {
    # thoát
    "thoc": "thoát",
    "thóc": "thoát",
    "thop": "thoát",
    "thopp": "thoát",
    "thoat": "thoát",

    # kết thúc
    "ket thoc": "kết thúc",
    "ket thuc": "kết thúc",
    "kết thóc": "kết thúc",
    "kết thuc": "kết thúc",
}


def normalize_text(text: str) -> str:
    """
    Chuẩn hoá văn bản cho:
    - Voice command (thoát / kết thúc)
    - Retrieval (RAG)
    """

    if not text:
        return ""

    # 1️⃣ Lowercase + strip
    text = text.lower().strip()

    # 2️⃣ Chuẩn hoá unicode tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # 3️⃣ ASR COMMAND FIX (PHẢI LÀM TRƯỚC)
    for k, v in COMMAND_FIX.items():
        if k in text:
            text = v
            return text  # ⛔ Ưu tiên command, không normalize thêm

    # 4️⃣ Remove punctuation (giữ chữ + số)
    text = re.sub(r"[^\w\s]", " ", text)

    # 5️⃣ Normalize abbreviations
    words = text.split()
    words = [ABBREVIATION_MAP.get(w, w) for w in words]
    text = " ".join(words)

    # 6️⃣ Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
