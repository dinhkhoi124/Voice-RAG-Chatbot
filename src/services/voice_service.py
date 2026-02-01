# src/services/voice_service.py
# OpenAI Whisper-1 ASR + Edge-TTS
# Jetson SAFE – AUTO MIC – PRODUCTION GRADE (NO WORD LOSS – FINAL)

import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import time
import os
import threading
import asyncio
import edge_tts
from collections import deque

from openai import OpenAI
from src.config.settings import settings


class VoiceService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

        self.sample_rate = 16000
        self.frame_duration = 0.03
        self.frame_size = int(self.sample_rate * self.frame_duration)

        self.max_record_seconds = 8
        self.min_voice_frames = 6
        self.preroll_frames = int(0.4 / self.frame_duration)

        self.voice = settings.TTS_VOICE
        self.output_file = settings.OUTPUT_AUDIO_FILE

        self.is_speaking = False
        print("🔥 OpenAI ASR ready (whisper-1)")

    # ======================================================
    # RECORD AUDIO WITH VAD + PRE-ROLL
    # ======================================================
    def record_audio_with_vad(self):
        if self.is_speaking:
            return None

        ring_buffer = deque(maxlen=self.preroll_frames)
        frames = []

        voiced = 0
        silence = 0
        max_silence = int(0.6 / self.frame_duration)

        print("🎤 Mời bạn nói...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
        ) as stream:

            start_time = time.time()
            triggered = False

            while True:
                if self.is_speaking:
                    return None

                indata, overflow = stream.read(self.frame_size)
                if overflow:
                    continue

                frame = indata[:, 0]
                ring_buffer.append(frame.copy())

                rms = np.sqrt(np.mean(frame ** 2))
                zcr = np.mean(np.abs(np.diff(np.sign(frame))))

                if rms > settings.SILENCE_THRESHOLD and zcr > 0.02:
                    voiced += 1
                    silence = 0
                else:
                    silence += 1

                if not triggered and voiced >= self.min_voice_frames:
                    triggered = True
                    frames.extend(ring_buffer)

                if triggered:
                    frames.append(frame.copy())

                if triggered and silence > max_silence:
                    break

                if time.time() - start_time > self.max_record_seconds:
                    break

        if not frames:
            return None

        return np.concatenate(frames)

    # ======================================================
    # SPEECH TO TEXT (🔥 REAL FIX HERE)
    # ======================================================
    def speech_to_text(self):
        if self.is_speaking:
            return None

        audio = self.record_audio_with_vad()
        if audio is None:
            return None

        # 🔑 ABSOLUTE FIX: prepend 300ms silence
        silence = np.zeros(int(self.sample_rate * 0.3), dtype=np.float32)
        audio = np.concatenate([silence, audio])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.sample_rate)
            wav_path = f.name

        try:
            t0 = time.time()
            with open(wav_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="vi",
                    temperature=0.0,
                )

            print(f"⚡ ASR latency: {int((time.time() - t0) * 1000)} ms")

            text = result.text.strip()
            if not text:
                return None

            blacklist = [
                "subscribe",
                "ghiền mì gõ",
                "like và share",
                "video hấp dẫn",
            ]
            if any(b in text.lower() for b in blacklist):
                print(f"🚫 Reject ASR hallucination: {text}")
                return None

            return text

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    # ======================================================
    # TTS (STABLE)
    # ======================================================
    def speak(self, text: str):
        if not text:
            return

        def worker():
            self.is_speaking = True
            try:
                async def run():
                    tts = edge_tts.Communicate(text, self.voice)
                    await tts.save(self.output_file)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run())

                data, fs = sf.read(self.output_file, dtype="float32")

                fade = int(fs * 0.05)
                data[:fade] *= np.linspace(0, 1, fade)

                silence = np.zeros((int(fs * 0.03),), dtype=np.float32)

                with sd.OutputStream(
                    samplerate=fs,
                    channels=1,
                    dtype="float32",
                    blocksize=512,
                ) as stream:
                    stream.write(silence.reshape(-1, 1))
                    stream.write(data.reshape(-1, 1))

            finally:
                self.is_speaking = False

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        try:
            sd.stop()
        except Exception:
            pass
        self.is_speaking = False

    def listen(self):
        if self.is_speaking:
            time.sleep(0.1)
            return None
        return self.speech_to_text()
