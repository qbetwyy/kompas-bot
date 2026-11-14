"""Логика наставника, не зависящая от Telegram (её же покрывают тесты)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from . import adaptive, competencies
from .detector import detect
from .export import build_blocks, export_markdown, export_pdf
from .feedback import Feedback, merge_strong_signals, rule_based_feedback
from .kb import Case, KnowledgeBase, Source
from .llm import MentorLLM
from .speech import SpeechReport
from .storage import Storage

log = logging.getLogger(__name__)
_REF_RE = re.compile(r"\[([ABC]\d{2})\]")


@dataclass
class InfoResult:
    text: str
    sources: list[Source]
    engine: str  # "llm" | "kb" | "none"


class Mentor:
    def __init__(self, kb: KnowledgeBase, llm: MentorLLM, storage: Storage, fonts_dir: Path, export_dir: Path):
        self.kb = kb
        self.llm = llm
        self.storage = storage
        self.fonts_dir = fonts_dir
        self.export_dir = export_dir

    # ------------------------------------------------------------------ справка
    async def info_answer(self, question: str, scenario: str | None,
                          history: list[tuple[str, str]] | None = None) -> InfoResult:
        scenario = None if scenario in (None, "all") else scenario
        found = self.kb.search(question, scenario=scenario, k=4)
        if not found:
            return InfoResult(text="", sources=[], engine="none")
        entries = [e for e, _ in found]
        if self.llm.available:
            try:
                text = await self.llm.answer_info(question, entries, history)
                cited = list(dict.fromkeys(_REF_RE.findall(text)))
                ids = [i for i in cited if i in self.kb.entries] or [e.id for e in entries[:2]]
                return InfoResult(text=text, sources=self.kb.sources_for(ids), engine="llm")
            except Exception:  # сеть, лимиты, таймаут, отвечаем из базы
                log.exception("LLM info failed")
        top = entries[:2]
        text = "\n\n".join(f"[{e.id}] {e.topic}. {e.content}" + (f" {e.insider}" if e.insider else "") for e in top)
        return InfoResult(text=text, sources=self.kb.sources_for([e.id for e in top]), engine="kb")

    # ---------------------------------------------------------------- тренажёр
    async def weak_zones(self, user_id: int) -> list[tuple[str, float]]:
        attempts = await self.storage.attempts_with_errors(user_id)
        return adaptive.weak_zones(attempts, self.kb)

    async def choose_case(self, user_id: int, scenario: str) -> tuple[Case, str | None, list[tuple[str, float]]]:
        attempts = await self.storage.attempts_with_errors(user_id)
        zones = adaptive.weak_zones(attempts, self.kb)
        case = adaptive.pick_case(scenario, self.kb, zones, attempts)
        return case, adaptive.target_zone(case, zones), zones

    async def evaluate(self, user_id: int, session_id: int, case: Case, answer: str,
                       previous: tuple[str, str] | None = None, speech: SpeechReport | None = None) -> Feedback:
        signals = detect(answer, case, self.kb)
        zones = await self.weak_zones(user_id)
        fb: Feedback | None = None
        if self.llm.available:
            try:
                fb = await self.llm.evaluate(case, answer, signals, zones, previous)
            except Exception:
                log.exception("LLM evaluate failed")
        if fb is None:
            fb = rule_based_feedback(answer, case, signals, self.kb)
        fb = merge_strong_signals(fb, signals)
        fb.extra["signals"] = [s.__dict__ for s in signals]
        if speech is not None:
            history = await self.storage.speech_history(user_id, limit=1)
            fb.extra["speech"] = speech.to_dict()
            if history:
                prev = history[0]["data"]
                fb.extra["speech_prev"] = {"fillers_per100": prev.get("fillers_per100"), "wpm": prev.get("wpm"),
                                           "score": prev.get("score")}
        attempt_id = await self.storage.save_attempt(
            user_id, session_id, case.id, answer, fb.score, fb.to_dict(), fb.errors, is_followup=previous is not None
        )
        if speech is not None:
            await self.storage.save_speech(user_id, session_id, attempt_id, speech.to_dict())
        return fb

    async def demo_feedback(self, case: Case, answer: str) -> Feedback:
        """Оценка для демо-режима: та же логика, но в историю пользователя не сохраняется."""
        signals = detect(answer, case, self.kb)
        fb: Feedback | None = None
        if self.llm.available:
            try:
                fb = await self.llm.evaluate(case, answer, signals, [])
            except Exception:
                log.exception("LLM demo evaluate failed")
        if fb is None:
            fb = rule_based_feedback(answer, case, signals, self.kb)
        return merge_strong_signals(fb, signals)

    def demo_map(self) -> Path:
        now = {"Конкретика": 7.5, "Структура ответа": 6.5, "Мотивация\nи подготовка": 7.0, "Уверенность": 6.0,
               "Коммуникация": 8.0, "Стратегия\nи права": 7.0, competencies.SPEECH_AXIS: 7.0}
        before = {"Конкретика": 4.0, "Структура ответа": 3.5, "Мотивация\nи подготовка": 5.0, "Уверенность": 2.5,
                  "Коммуникация": 5.5, "Стратегия\nи права": 6.0, competencies.SPEECH_AXIS: 4.0}
        path = self.export_dir / "competency_demo.png"
        if not path.exists():
            competencies.render_radar(now, before, path, title="Карта навыков (пример)")
        return path

    async def session_summary(self, user_id: int, session_id: int) -> dict:
        attempts = [a for a in await self.storage.attempts_with_errors(user_id) if a["session_id"] == session_id]
        zones = await self.weak_zones(user_id)
        errs: dict[str, int] = {}
        for a in attempts:
            for c in a["errors"]:
                errs[c] = errs.get(c, 0) + 1
        avg = round(sum(a["score"] for a in attempts) / len(attempts), 1) if attempts else None
        return {"attempts": len(attempts), "avg": avg, "errors": errs, "zones": zones}

    async def progress(self, user_id: int) -> dict | None:
        attempts = await self.storage.attempts_with_errors(user_id, limit=500)
        if not attempts:
            return None
        by_scn: dict[str, list[int]] = {}
        for a in attempts:
            case = self.kb.cases.get(a["case_id"])
            if case:
                by_scn.setdefault(case.scenario, []).append(a["score"])
        zones = adaptive.weak_zones(attempts, self.kb)
        recent = [a["score"] for a in attempts[:5]]
        earlier = [a["score"] for a in attempts[5:10]]
        return {
            "total": len(attempts),
            "by_scenario": {k: (len(v), round(sum(v) / len(v), 1)) for k, v in by_scn.items()},
            "zones": zones,
            "trend": (round(sum(recent) / len(recent), 1), round(sum(earlier) / len(earlier), 1) if earlier else None),
        }

    # ------------------------------------------------------- карта компетенций
    async def competency_map(self, user_id: int, session_id: int | None = None) -> tuple[Path, dict, dict | None] | None:
        """Сейчас = текущая сессия (или последние 6 ответов), раньше = всё, что было до этого."""
        attempts = await self.storage.attempts_with_errors(user_id, limit=500)
        if not attempts:
            return None
        speech = await self.storage.speech_history(user_id, limit=500)
        if session_id is not None:
            now_a = [a for a in attempts if a["session_id"] == session_id]
            before_a = [a for a in attempts if a["session_id"] < session_id]
            now_s = [s["score"] for s in speech if s["session_id"] == session_id]
            before_s = [s["score"] for s in speech if s["session_id"] < session_id]
        else:
            now_a, before_a = attempts[:6], attempts[6:]
            now_s, before_s = [s["score"] for s in speech[:6]], [s["score"] for s in speech[6:]]
        if not now_a:
            return None
        now_p = competencies.profile(now_a, self.kb, now_s)
        before_p = competencies.profile(before_a, self.kb, before_s) if len(before_a) >= 2 else None
        if before_p and not any(v is not None for v in before_p.values()):
            before_p = None
        path = self.export_dir / f"competency_{user_id}_{session_id or 'all'}.png"
        competencies.render_radar(now_p, before_p, path)
        return path, now_p, before_p

    # ------------------------------------------------------------------ экспорт
    async def export(self, user_id: int, fmt: str, scope: str) -> Path | None:
        if scope == "all":
            transcript = await self.storage.transcript(user_id)
        else:
            sid = await self.storage.last_session_id(user_id)
            if sid is None:
                return None
            transcript = await self.storage.transcript(user_id, [sid])
        if not transcript:
            return None
        blocks = build_blocks(transcript, self.kb)
        zones = await self.weak_zones(user_id)
        name = f"kompas_{user_id}_{scope}"
        if fmt == "md":
            return export_markdown(self.export_dir / f"{name}.md", blocks, zones, self.kb)
        chart = None
        try:
            cm = await self.competency_map(user_id, None)
            chart = cm[0] if cm else None
        except Exception:
            log.exception("competency map failed")
        return export_pdf(self.export_dir / f"{name}.pdf", blocks, zones, self.kb, self.fonts_dir, chart=chart)
