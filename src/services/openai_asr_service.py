import os
import tempfile
import soundfile as sf
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OpenAIASRService:
    def __init__(self, model=None, language="vi"):
        self.model = model or os.getenv("OPENAI_ASR_MODEL", "gpt-4o-transcribe")
        self.language = language

    def transcribe(self, audio_np, sample_rate):
        """
        audio_np: numpy array (float32 or int16)
        sample_rate: int
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_np, sample_rate)

            with open(tmp.name, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    language=self.language
                )

        return result.text.strip()
