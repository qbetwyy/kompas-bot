"""Разбор подачи в голосовом ответе: слова-паразиты, темп, паузы, задержка перед ответом.

Входные данные, слова с таймкодами от faster-whisper. Анализ полностью локальный и детерминированный:
одна и та же запись всегда даёт один и тот же результат, поэтому динамику между попытками честно видно.
Пороги, настройки бота, ориентиры взяты из записи B14 базы знаний.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

# «Жёсткие» паразиты считаются всегда.
HARD_FILLERS = {"ну": "ну", "типа": "типа", "короче": "короче", "блин": "блин", "походу": "походу", "чисто": "чисто"}
# «Мягкие», нормальные слова, которые становятся паразитами при повторении: считаем, если встретились 2+ раза.
SOFT_FILLERS = {"вот": "вот", "значит": "значит", "собственно": "собственно", "реально": "реально"}
HARD_PHRASES = ["как бы", "в общем", "это самое", "так сказать", "грубо говоря"]
SOFT_PHRASES = ["в принципе", "на самом деле", "то есть", "получается"]
_HESITATION = re.compile(r"^(э+|м+|эм+|мэ+|а-а+)$")
_TOKEN = re.compile(r"[а-яёa-z]+", re.IGNORECASE)

# Пороги (слов в минуту, секунды, паразитов на 100 слов)
PACE_SLOW = 90
PACE_FAST = 170
LONG_PAUSE = 2.5
SLOW_START = 3.0
FILLERS_OK = 2.0
FILLERS_MANY = 6.0
FILLERS_TERRIBLE = 15.0


@dataclass
class SpeechReport:
    duration: float
    words: int
    wpm: float | None
    fillers: dict[str, int] = field(default_factory=dict)
    fillers_total: int = 0
    fillers_per100: float = 0.0
    long_pauses: int = 0
    longest_pause: float = 0.0
    start_delay: float = 0.0
    pace_verdict: str = ""
    score: int = 10
    tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpeechReport":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _normalize(text: str) -> list[str]:
    return [t.lower().replace("ё", "е") for t in _TOKEN.findall((text or "").replace("-", " "))]


def count_fillers(text: str) -> Counter:
    tokens = _normalize(text)
    found: Counter = Counter()
    prev_hes = False
    for t in tokens:
        if _HESITATION.match(t):
            if not prev_hes:  # «э-э-э» → одна запинка, а не три
                found["э-э / м-м"] += 1
            prev_hes = True
            continue
        prev_hes = False
        if t in HARD_FILLERS:
            found[HARD_FILLERS[t]] += 1
    soft = Counter(SOFT_FILLERS[t] for t in tokens if t in SOFT_FILLERS)
    joined = " " + " ".join(tokens) + " "
    for ph in HARD_PHRASES:
        n = joined.count(" " + ph + " ")
        if n:
            found[ph] += n
    for ph in SOFT_PHRASES:
        n = joined.count(" " + ph + " ")
        if n:
            soft[ph] += n
    for word, n in soft.items():
        if n >= 2:
            found[word] += n
    return found


def analyze(text: str, words: list[tuple[float, float, str]], duration: float | None = None) -> SpeechReport:
    """words, список (начало, конец, слово) в секундах."""
    tokens = [t for t in _normalize(text) if not _HESITATION.match(t)]
    n_words = len(tokens)
    fillers = count_fillers(text)
    total = sum(fillers.values())
    per100 = round(100 * total / max(n_words, 1), 1)

    wpm = None
    long_pauses, longest, start_delay = 0, 0.0, 0.0
    if words:
        start_delay = max(words[0][0], 0.0)
        speech_time = words[-1][1] - words[0][0]
        if speech_time >= 3:
            wpm = round(n_words / speech_time * 60)
        for (_, prev_end, _), (start, _, _) in zip(words, words[1:]):
            gap = start - prev_end
            if gap >= LONG_PAUSE:
                long_pauses += 1
            longest = max(longest, gap)
    elif duration and duration >= 3:
        wpm = round(n_words / duration * 60)

    tips: list[str] = []
    score = 10
    if wpm is None:
        pace = "ответ слишком короткий, чтобы оценить темп"
    elif wpm < PACE_SLOW:
        pace = "медленно"
        score -= 2
        tips.append("Можно чуть бодрее. Помогает заранее держать в голове 2-3 главные мысли, тогда не приходится подбирать слова на ходу.")
    elif wpm > PACE_FAST:
        pace = "быстро"
        score -= 2
        tips.append("Попробуй чуть медленнее, особенно на главной мысли. Короткая пауза перед ней работает лучше, чем скорость.")
    else:
        pace = "комфортно"

    if per100 >= FILLERS_TERRIBLE:
        score -= 6
    elif per100 >= FILLERS_MANY:
        score -= 4
    elif per100 >= FILLERS_OK:
        score -= 2
    if total and per100 >= FILLERS_OK:
        top = fillers.most_common(1)[0][0]
        tips.append(f"Чаще всего проскакивает «{top}». Когда тянет его сказать, просто помолчи секунду: пауза звучит увереннее.")

    if long_pauses:
        score -= min(2, long_pauses)
        tips.append("Были долгие паузы. Если нужно подумать, так и скажи: «секунду, подумаю». Это нормально.")
    if start_delay >= SLOW_START:
        score -= 1
        tips.append("Начинай чуть быстрее. Первой фразой можно просто повторить вопрос, пока собираешься с мыслями.")

    return SpeechReport(
        duration=round(duration or (words[-1][1] if words else 0.0), 1),
        words=n_words,
        wpm=wpm,
        fillers=dict(fillers.most_common()),
        fillers_total=total,
        fillers_per100=per100,
        long_pauses=long_pauses,
        longest_pause=round(longest, 1),
        start_delay=round(start_delay, 1),
        pace_verdict=pace,
        score=max(0, score),
        tips=tips[:3],
    )
