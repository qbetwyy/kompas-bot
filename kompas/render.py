"""Форматирование ответов для Telegram (HTML)."""
from __future__ import annotations

import re
from html import escape

from .feedback import Feedback
from .kb import Case, KnowledgeBase, Source


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def llm_text(text: str) -> str:
    """Экранирует ответ модели и превращает **жирный** в <b>жирный</b>."""
    return _BOLD_RE.sub(r"<b>\1</b>", escape(text)).replace("### ", "").replace("## ", "")


def sources_block(sources: list[Source], limit: int = 5) -> str:
    if not sources:
        return ""
    lines = [f'<a href="{escape(s.url, quote=True)}">{escape(s.title)}</a>' for s in sources[:limit]]
    return "\n\n<b>Откуда это:</b>\n" + "\n".join(lines)


def case_message(case: Case, kb: KnowledgeBase, zone_title: str | None = None, number: int | None = None) -> str:
    head = ""
    if zone_title:
        head = f"В прошлый раз было слабое место: <b>{escape(zone_title.lower())}</b>. На нём и потренируемся.\n\n"
    num = f"Вопрос {number}. " if number else ""
    return (
        f"{head}<b>{num}{escape(case.title)}</b>\n"
        f"<i>Я сейчас {escape(case.role)}</i>\n\n"
        f"{escape(case.prompt)}\n\n"
        "Отвечай так, как ответил бы вживую. Голосом даже лучше."
    )


def _fmt_fillers(fillers: dict[str, int], limit: int = 4) -> str:
    items = sorted(fillers.items(), key=lambda x: x[1], reverse=True)[:limit]
    return ", ".join(f"«{escape(w)}» ×{n}" for w, n in items)


def speech_block(sp: dict, prev: dict | None = None) -> str:
    dur = int(round(sp.get("duration") or 0))
    pace_words = {"медленно": "медленновато", "быстро": "быстровато", "комфортно": "нормально"}
    lines = [f"<b>Как ты говорил: {sp.get('score', 0)}/10</b> <i>(голосовое {dur // 60}:{dur % 60:02d})</i>"]
    wpm = sp.get("wpm")
    if wpm:
        lines.append(f"Темп: {wpm} слов в минуту, {pace_words.get(sp.get('pace_verdict', ''), sp.get('pace_verdict', ''))}")
    if sp.get("fillers_total"):
        lines.append(f"Слова-паразиты: {_fmt_fillers(sp['fillers'])}, это {sp['fillers_per100']} на 100 слов")
    else:
        lines.append("Слов-паразитов не услышал")
    if sp.get("long_pauses"):
        lines.append(f"Долгих пауз: {sp['long_pauses']}, самая длинная {sp['longest_pause']} с")
    if sp.get("start_delay", 0) >= 1:
        lines.append(f"Начал отвечать через {sp['start_delay']} с")
    if prev and prev.get("fillers_per100") is not None:
        a, b = prev["fillers_per100"], sp.get("fillers_per100", 0)
        trend = "стало лучше" if b < a else ("стало хуже" if b > a else "без изменений")
        lines.append(f"Паразитов на 100 слов: было {a}, сейчас {b}, {trend}")
    for tip in sp.get("tips", [])[:2]:
        lines.append(f"<i>{escape(tip)}</i>")
    return "\n".join(lines)


def feedback_message(fb: Feedback, kb: KnowledgeBase) -> str:
    parts = [f"<b>{fb.score} из 10</b>"]
    if fb.errors:
        titles = [escape(kb.errors[c].title.lower()) for c in fb.errors if c in kb.errors]
        word = "Заметил типичную ошибку" if len(titles) == 1 else "Заметил типичные ошибки"
        parts.append(f"<i>{word}: {', '.join(titles)}</i>")
    parts.append("<b>Это верно</b>\n" + "\n".join(f"• {escape(x)}" for x in fb.correct))
    parts.append("<b>Вот что можно улучшить</b>\n" + "\n".join(f"• {escape(x)}" for x in fb.improve))
    parts.append("<b>Вот удачный пример ответа</b>\n<blockquote>" + escape(fb.improved_answer) + "</blockquote>")
    if fb.extra.get("speech"):
        parts.append(speech_block(fb.extra["speech"], fb.extra.get("speech_prev")))
    if fb.follow_up:
        parts.append("<b>И ещё вопрос:</b> " + escape(fb.follow_up))
    text = "\n\n".join(parts)
    return text + sources_block(kb.sources_for(fb.kb_refs), limit=3)


def summary_message(summary: dict, kb: KnowledgeBase) -> str:
    if not summary["attempts"]:
        return "Закончили. Ответов в этот раз не было, возвращайся, когда будешь готов."
    lines = [f"<b>Итог</b>\nОтветов: {summary['attempts']}, в среднем {summary['avg']} из 10."]
    if summary["errors"]:
        top = sorted(summary["errors"].items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append("Чаще всего мешало:\n" + "\n".join(
            f"• {escape(kb.errors[c].title.lower())}. {escape(kb.errors[c].fix)}" for c, _ in top if c in kb.errors))
    else:
        lines.append("Типичных ошибок не нашёл. Хорошо отвечаешь.")
    if summary["zones"]:
        lines.append("В следующий раз начнём с этого: <b>" + escape(kb.errors[summary["zones"][0][0]].title.lower()) + "</b>.")
    lines.append("Разбор можно скачать и перечитать перед собеседованием.")
    return "\n\n".join(lines)


def progress_message(p: dict, kb: KnowledgeBase) -> str:
    lines = [f"<b>Прогресс</b>\nВсего ответов: {p['total']}"]
    for scn, (n, avg) in p["by_scenario"].items():
        lines.append(f"{escape(kb.scenario_title(scn))}: {n}, в среднем {avg} из 10")
    recent, earlier = p["trend"]
    if earlier is not None:
        trend = "растёт" if recent > earlier else ("просела" if recent < earlier else "держится")
        lines.append(f"Последние 5 ответов: {recent} из 10, до этого {earlier}. Оценка {trend}.")
    if p["zones"]:
        lines.append("\n<b>Над чем поработать</b> (с этого начну следующую тренировку):")
        for code, w in p["zones"][:5]:
            e = kb.errors[code]
            lines.append(f"• <b>{escape(e.title.lower())}</b>. {escape(e.fix)}")
    else:
        lines.append("\nЯвных слабых мест сейчас нет.")
    return "\n".join(lines)


def competency_caption(now: dict, before: dict | None) -> str:
    from .competencies import strongest_weakest

    best, worst = strongest_weakest(now)
    lines = ["<b>Твоя карта навыков</b>" + (": сейчас и раньше" if before else "")]
    if best:
        lines.append(f"Лучше всего: {escape(best.lower())}")
    if worst:
        lines.append(f"Подтянуть: {escape(worst.lower())}")
    if before:
        ups = [k.replace(chr(10), ' ').lower() for k, v in now.items()
               if v is not None and before.get(k) is not None and v - before[k] >= 1]
        if ups:
            lines.append("Заметно выросло: " + escape(", ".join(ups)))
    return "\n".join(lines)


def split_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for para in text.split("\n\n"):
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) > limit and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
