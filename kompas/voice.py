"""Распознавание голосовых сообщений (опционально).

Используется локальная модель faster-whisper: голос не уходит сторонним сервисам,
а ключи дополнительных API не нужны. Если пакет не установлен или VOICE_ENABLED=false,
бот вежливо просит написать текстом.

Возвращаются не только текст, но и таймкоды слов, по ним считается разбор подачи (speech.py).
Подсказка initial_prompt с запинками нужна, чтобы модель не «вычищала» слова-паразиты из расшифровки.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_PROMPT = "Ну, э-э, типа, короче, как бы, в общем, вот. Ответ на вопрос собеседования."


def decode_audio(path: Path, sampling_rate: int = 16000):
    """Декодирует голосовое Telegram (OGG/Opus) в моно float32 16 кГц через PyAV.

    Своя функция вместо встроенной в faster-whisper: в новых версиях PyAV (19+) у av.open
    больше нет аргумента metadata_errors, и встроенный декодер faster-whisper падает с TypeError.
    """
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


@dataclass
class Transcription:
    text: str
    words: list[tuple[float, float, str]] = field(default_factory=list)
    duration: float = 0.0


class VoiceTranscriber:
    def __init__(self, enabled: bool, model_size: str = "small"):
        self.model_size = model_size
        self._model = None
        self._lock = asyncio.Lock()
        self._enabled = False
        if enabled:
            try:
                import faster_whisper  # noqa: F401

                self._enabled = True
            except ImportError:
                log.warning("faster-whisper не установлен, голосовые сообщения выключены")

    @property
    def available(self) -> bool:
        return self._enabled

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Загружаю модель Whisper %s…", self.model_size)
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def _transcribe_sync(self, path: Path) -> Transcription:
        model = self._load()
        audio = decode_audio(path)
        segments, info = model.transcribe(
            audio, language="ru", vad_filter=False, word_timestamps=True, initial_prompt=_PROMPT,
            condition_on_previous_text=False,
        )
        words, texts = [], []
        for seg in segments:
            texts.append(seg.text.strip())
            for w in seg.words or []:
                words.append((float(w.start), float(w.end), w.word.strip()))
        return Transcription(text=" ".join(texts).strip(), words=words, duration=float(getattr(info, "duration", 0.0)))

    async def transcribe(self, path: Path) -> Transcription:
        async with self._lock:  # одна модель, одно распознавание за раз
            return await asyncio.get_running_loop().run_in_executor(None, self._transcribe_sync, path)
