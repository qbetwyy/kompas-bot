"""Адаптивное обучение (этап 4).

Слабая зона, тип ошибки, который пользователь допускал недавно и ещё не «закрыл».
Вес каждой ошибки затухает со временем (период полураспада 7 дней), а чистые попытки
в кейсах, которые проверяют эту ошибку, уменьшают вес. Следующий кейс выбирается так,
чтобы проверить самые тяжёлые слабые зоны, при этом не повторяя только что пройденные кейсы.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone

from .kb import Case, KnowledgeBase

HALF_LIFE_DAYS = 7.0
CLEAN_CREDIT = 0.5
THRESHOLD = 0.3


def _age_days(iso: str, now: datetime) -> float:
    try:
        ts = datetime.fromisoformat(iso)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max((now - ts).total_seconds() / 86400, 0.0)


def weak_zones(attempts: list[dict], kb: KnowledgeBase, now: datetime | None = None) -> list[tuple[str, float]]:
    now = now or datetime.now(timezone.utc)
    score: dict[str, float] = {}
    for a in attempts:
        w = 0.5 ** (_age_days(a["created_at"], now) / HALF_LIFE_DAYS)
        errors = set(a.get("errors", []))
        for code in errors:
            score[code] = score.get(code, 0.0) + w
        case = kb.cases.get(a["case_id"])
        if case:
            for code in case.tests:
                if code not in errors and code in score:
                    score[code] -= CLEAN_CREDIT * w
    zones = [(c, round(s, 2)) for c, s in score.items() if s >= THRESHOLD and c in kb.errors]
    zones.sort(key=lambda x: x[1], reverse=True)
    return zones


def pick_case(scenario: str, kb: KnowledgeBase, zones: list[tuple[str, float]],
              attempts: list[dict], rng: random.Random | None = None) -> Case:
    rng = rng or random.Random()
    candidates = kb.cases_for(scenario)
    if not candidates:
        raise ValueError(f"Нет кейсов для сценария {scenario}")
    weight = dict(zones)
    recent = [a["case_id"] for a in attempts[:4]]
    counts: dict[str, int] = {}
    for a in attempts:
        counts[a["case_id"]] = counts.get(a["case_id"], 0) + 1

    def value(c: Case) -> float:
        v = sum(weight.get(code, 0.0) for code in c.tests) * 3  # слабые зоны важнее разнообразия
        if recent and c.id == recent[0]:
            v -= 10                              # один и тот же вопрос два раза подряд не задаём
        elif c.id in recent[1:3]:
            v -= 0.4 * (3 - recent.index(c.id))  # недавние чуть откладываем, но к слабому месту возвращаемся
        v -= 0.25 * counts.get(c.id, 0)          # редко пройденные, вперёд
        return v + rng.random() * 0.2

    return max(candidates, key=value)


def target_zone(case: Case, zones: list[tuple[str, float]]) -> str | None:
    """Какую слабую зону проверяет выбранный кейс (для приветствия «давай проверим»)."""
    for code, _ in zones:
        if code in case.tests:
            return code
    return None
