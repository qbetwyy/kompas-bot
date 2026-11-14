"""Карта компетенций: радар-диаграмма навыков «было / стало».

Каждая ось, группа типовых ошибок. Оценка оси считается только по тем попыткам, в которых кейс
действительно проверял эту группу (иначе ось «нет данных»):
    ось = 0.5 · 10 · (1 − доля попыток с ошибкой из группы) + 0.5 · средняя оценка этих попыток
Ось «Подача» появляется, если есть голосовые ответы, это средняя оценка подачи.
"""
from __future__ import annotations

import math
from pathlib import Path

from .kb import KnowledgeBase

AXES: list[tuple[str, tuple[str, ...]]] = [
    ("Конкретика", ("VAGUE", "OVERCLAIM")),
    ("Структура ответа", ("NO_STRUCTURE",)),
    ("Мотивация\nи подготовка", ("NO_MOTIVATION", "NO_RESEARCH")),
    ("Уверенность", ("SELF_DEPRECATION", "CRITICAL_WEAKNESS", "NO_PORTFOLIO")),
    ("Коммуникация", ("NO_QUESTIONS", "SILENT_PROBLEMS", "NO_EXPECTATIONS", "NO_SALARY_ANCHOR")),
    ("Стратегия\nи права", ("DEADLINE_RISK", "NO_BACKUP", "PRIORITY_BY_CHANCES", "RATING_ONLY", "RIGHTS_UNAWARE", "RUSH_DECISIONS")),
]
SPEECH_AXIS = "Подача\n(голос)"

COLOR_NOW = "#2a78d6"     # проверено валидатором палитры: CVD ΔE 24.7, контраст ≥ 3:1
COLOR_BEFORE = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d3"


def profile(attempts: list[dict], kb: KnowledgeBase, speech_scores: list[int] | None = None) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, codes in AXES:
        opp = fail = 0
        scores = []
        for a in attempts:
            case = kb.cases.get(a["case_id"])
            if not case or not set(case.tests) & set(codes):
                continue
            opp += 1
            scores.append(a["score"])
            if set(a.get("errors", [])) & set(codes):
                fail += 1
        if opp == 0:
            result[name] = None
        else:
            result[name] = round(0.5 * 10 * (1 - fail / opp) + 0.5 * (sum(scores) / len(scores)), 1)
    if speech_scores:
        result[SPEECH_AXIS] = round(sum(speech_scores) / len(speech_scores), 1)
    return result


def strongest_weakest(p: dict[str, float | None]) -> tuple[str | None, str | None]:
    known = {k: v for k, v in p.items() if v is not None}
    if not known:
        return None, None
    best = max(known, key=known.get)
    worst = min(known, key=known.get)
    return best.replace("\n", " "), (worst.replace("\n", " ") if known[worst] < known[best] else None)


def render_radar(now: dict[str, float | None], before: dict[str, float | None] | None, path: Path,
                 title: str = "Карта компетенций") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(now.keys())
    if before:
        for k in before:
            if k not in now:
                labels.append(k)
    # Оси без данных не рисуем (иначе «0» выглядит как провал), если осей с данными хватает для фигуры.
    with_data = [k for k in labels if now.get(k) is not None or (before and before.get(k) is not None)]
    if len(with_data) >= 3:
        labels = with_data
    n = len(labels)
    angles = [i / n * 2 * math.pi for i in range(n)] + [0]

    def values(p):
        return [(p.get(k) if p.get(k) is not None else 0) for k in labels] + [(p.get(labels[0]) or 0)]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(6.4, 6.4), dpi=150, facecolor=SURFACE)
    ax = fig.add_subplot(111, polar=True, facecolor=SURFACE)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color=TEXT_SECONDARY, fontsize=7)
    ax.set_rlabel_position(180 / n)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)

    tick_labels = []
    for k in labels:
        no_data = now.get(k) is None and (not before or before.get(k) is None)
        tick_labels.append(k + ("\n(нет данных)" if no_data else ""))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labels, color=TEXT_PRIMARY, fontsize=9)
    ax.tick_params(axis="x", pad=18)

    if before:
        vb = values(before)
        ax.plot(angles, vb, color=COLOR_BEFORE, linewidth=2, linestyle=(0, (4, 3)), label="Раньше")
        ax.fill(angles, vb, color=COLOR_BEFORE, alpha=0.08)
        ax.scatter(angles[:-1], vb[:-1], s=22, color=COLOR_BEFORE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    vn = values(now)
    ax.plot(angles, vn, color=COLOR_NOW, linewidth=2, label="Сейчас")
    ax.fill(angles, vn, color=COLOR_NOW, alpha=0.15)
    ax.scatter(angles[:-1], vn[:-1], s=30, color=COLOR_NOW, edgecolors=SURFACE, linewidths=1.5, zorder=4)

    fig.suptitle(title, fontsize=14, color=TEXT_PRIMARY, fontweight="bold", y=0.97)
    fig.text(0.5, 0.915, "шкала 0-10 · по ответам в тренажёре «Компаса»", ha="center", fontsize=8, color=TEXT_SECONDARY)
    if before:
        handles, lbls = ax.get_legend_handles_labels()
        fig.legend(handles[::-1], lbls[::-1], loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=2, frameon=False,
                   fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.subplots_adjust(top=0.76, bottom=0.08, left=0.17, right=0.83)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path
