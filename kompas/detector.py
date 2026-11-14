"""Правиловый детектор типовых ошибок.

Работает до языковой модели и независимо от неё:
* «сильные» сигналы (например, «ничего не умею») засчитываются всегда;
* «слабые» сигналы учитываются, только если кейс проверяет эту ошибку;
* модель получает сигналы как подсказку и может их подтвердить или отклонить,
  но сильные сигналы добавляются к итогу в любом случае.
"""
from __future__ import annotations

from dataclasses import dataclass

from .kb import Case, KnowledgeBase


@dataclass(frozen=True)
class Signal:
    code: str
    strength: str
    match: str


def detect(text: str, case: Case | None, kb: KnowledgeBase) -> list[Signal]:
    signals: list[Signal] = []
    seen: set[str] = set()
    tested = set(case.tests) if case else set()
    low = (text or "").lower()
    for err in kb.errors.values():
        for pattern in err.patterns:
            m = pattern.regex.search(low)
            if not m:
                continue
            if pattern.strength == "weak" and err.code not in tested:
                continue
            if err.code in seen:
                continue
            seen.add(err.code)
            signals.append(Signal(err.code, pattern.strength, m.group(0)))
    # Слишком короткий ответ, почти всегда «общие слова».
    # Не дублируем, если уже нашли явную («сильную») ошибку, она и так объясняет слабость ответа.
    has_strong = any(sig.strength == "strong" for sig in signals)
    if case and "VAGUE" in tested and "VAGUE" not in seen and not has_strong and len(low.split()) < 12:
        signals.append(Signal("VAGUE", "weak", "ответ короче 12 слов"))
    return signals


def strong_codes(signals: list[Signal]) -> list[str]:
    return [s.code for s in signals if s.strength == "strong"]
