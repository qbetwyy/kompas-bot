"""Экспорт диалога с рекомендациями в PDF и в текстовую заметку (Markdown)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from .kb import KnowledgeBase
from .texts import DISCLAIMER_PLAIN

_EMOJI_RE = re.compile("[\U00010000-\U0010FFFF️‍]")


def _clean(text: str) -> str:
    return _EMOJI_RE.sub("", text or "").strip()


def build_blocks(transcript: list[dict], kb: KnowledgeBase) -> list[dict]:
    """Превращает сырые сообщения из БД в смысловые блоки для экспорта."""
    blocks = []
    for m in transcript:
        meta = m.get("meta") or {}
        kind = m["kind"]
        if kind == "info_q":
            blocks.append({"type": "info_q", "text": m["text"]})
        elif kind == "info_a":
            blocks.append({"type": "info_a", "text": m["text"], "sources": meta.get("sources", [])})
        elif kind == "case":
            case = kb.cases.get(meta.get("case_id", ""))
            blocks.append({
                "type": "case",
                "title": case.title if case else "Кейс",
                "scenario": kb.scenario_title(case.scenario) if case else "",
                "text": m["text"],
            })
        elif kind == "answer":
            blocks.append({"type": "answer", "text": m["text"], "followup": bool(meta.get("followup"))})
        elif kind == "feedback":
            fb = meta.get("feedback", {})
            refs = fb.get("kb_refs", [])
            blocks.append({
                "type": "feedback",
                "score": fb.get("score"),
                "correct": fb.get("correct", []),
                "improve": fb.get("improve", []),
                "improved_answer": fb.get("improved_answer", ""),
                "follow_up": fb.get("follow_up", ""),
                "errors": [kb.errors[c].title for c in fb.get("errors", []) if c in kb.errors],
                "sources": [{"title": s.title, "url": s.url} for s in kb.sources_for(refs)],
                "speech": (fb.get("extra") or {}).get("speech"),
            })
    return blocks


def _zones_lines(zones: list[tuple[str, float]], kb: KnowledgeBase) -> list[tuple[str, str]]:
    return [(kb.errors[c].title, kb.errors[c].fix) for c, _ in zones if c in kb.errors]


# --------------------------------------------------------------------------- PDF
class _Pdf(FPDF):
    def __init__(self, fonts_dir: Path):
        super().__init__(format="A4")
        self.add_font("DejaVu", "", str(fonts_dir / "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", str(fonts_dir / "DejaVuSans-Bold.ttf"))
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 16, 18)

    def footer(self):
        self.set_y(-14)
        self.set_font("DejaVu", "", 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 5, f"«Компас», AI-наставник. Рекомендации носят информационный характер.   Стр. {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def _speech_lines(sp: dict) -> list[str]:
    lines = [f"Подача (голос): {sp.get('score', 0)}/10"]
    if sp.get("wpm"):
        lines.append(f"Темп: {sp['wpm']} слов/мин, {sp.get('pace_verdict', '')}")
    if sp.get("fillers_total"):
        top = ", ".join(f"«{w}» ×{n}" for w, n in list(sp["fillers"].items())[:4])
        lines.append(f"Слова-паразиты: {top}, {sp['fillers_per100']} на 100 слов")
    else:
        lines.append("Слова-паразиты: не найдены")
    if sp.get("long_pauses"):
        lines.append(f"Длинные паузы: {sp['long_pauses']} (самая длинная {sp['longest_pause']} с)")
    lines += sp.get("tips", [])[:2]
    return lines


def export_pdf(path: Path, blocks: list[dict], zones: list[tuple[str, float]], kb: KnowledgeBase, fonts_dir: Path,
               chart: Path | None = None) -> Path:
    pdf = _Pdf(fonts_dir)
    pdf.add_page()

    def para(text: str, size: int = 10, bold: bool = False, color=(0, 0, 0), indent: float = 0, gap: float = 1.5):
        pdf.set_font("DejaVu", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(0, size * 0.5, _clean(text), align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)
        pdf.set_text_color(0, 0, 0)

    def bullets(items: list[str], indent: float = 4):
        for it in items:
            para("• " + it, indent=indent, gap=0.8)

    def sources(items: list[dict]):
        if not items:
            return
        para("Источники:", size=8, bold=True, color=(90, 90, 90), gap=0.5)
        for s in items:
            pdf.set_font("DejaVu", "", 8)
            pdf.set_text_color(30, 80, 160)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 4, _clean(s["title"]), link=s["url"], align="L", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1.5)

    para("Компас, разбор сессии", size=18, bold=True, gap=1)
    para(datetime.now().strftime("Сформировано %d.%m.%Y %H:%M"), size=9, color=(110, 110, 110), gap=1)
    para(DISCLAIMER_PLAIN, size=8, color=(110, 110, 110), gap=4)

    if chart and Path(chart).exists():
        pdf.image(str(chart), x=(pdf.w - 120) / 2, w=120)
        pdf.ln(3)

    zl = _zones_lines(zones, kb)
    if zl:
        para("Твои слабые зоны, на что обратить внимание перед собеседованием", size=12, bold=True)
        for title, fix in zl:
            para("• " + title, bold=True, indent=4, gap=0.3)
            para(fix, indent=8, gap=1.2)
        pdf.ln(2)

    for b in blocks:
        t = b["type"]
        if t == "case":
            pdf.ln(2)
            para(f"Кейс: {b['title']}" + (f"  ({b['scenario']})" if b["scenario"] else ""), size=12, bold=True)
            para(b["text"], color=(60, 60, 60))
        elif t == "answer":
            para("Ответ на уточняющий вопрос:" if b["followup"] else "Твой ответ:", bold=True, gap=0.5)
            para(b["text"], indent=4)
        elif t == "feedback":
            if b.get("score") is not None:
                para(f"Оценка: {b['score']}/10", bold=True, color=(30, 110, 60))
            if b["correct"]:
                para("Это верно:", bold=True, gap=0.5)
                bullets(b["correct"])
            if b["improve"]:
                para("Вот что можно улучшить:", bold=True, gap=0.5)
                bullets(b["improve"])
            if b["improved_answer"]:
                para("Вот удачный пример ответа:", bold=True, gap=0.5)
                para(b["improved_answer"], indent=4, color=(40, 40, 120))
            if b.get("speech"):
                sl = _speech_lines(b["speech"])
                para(sl[0], bold=True, gap=0.5)
                bullets(sl[1:])
            if b["follow_up"]:
                para("Уточняющий вопрос наставника: " + b["follow_up"], color=(110, 60, 0))
            sources(b["sources"])
        elif t == "info_q":
            pdf.ln(2)
            para("Вопрос: " + b["text"], size=11, bold=True)
        elif t == "info_a":
            para(b["text"])
            sources(b["sources"])

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


# ---------------------------------------------------------------------- Markdown
def export_markdown(path: Path, blocks: list[dict], zones: list[tuple[str, float]], kb: KnowledgeBase) -> Path:
    lines = ["# Компас, разбор сессии", "", datetime.now().strftime("_Сформировано %d.%m.%Y %H:%M_"), "",
             f"> {DISCLAIMER_PLAIN}", ""]
    zl = _zones_lines(zones, kb)
    if zl:
        lines += ["## Слабые зоны", ""]
        lines += [f"- **{title}**, {fix}" for title, fix in zl]
        lines.append("")

    def src(items):
        if items:
            lines.append("")
            lines.append("Источники: " + "; ".join(f"[{s['title']}]({s['url']})" for s in items))
            lines.append("")

    for b in blocks:
        t = b["type"]
        if t == "case":
            lines += ["", f"## Кейс: {b['title']}", "", b["text"], ""]
        elif t == "answer":
            lines += [f"**{'Ответ на уточняющий вопрос' if b['followup'] else 'Твой ответ'}:** {b['text']}", ""]
        elif t == "feedback":
            if b.get("score") is not None:
                lines.append(f"**Оценка:** {b['score']}/10")
            if b["correct"]:
                lines += ["", "**Это верно:**"] + [f"- {x}" for x in b["correct"]]
            if b["improve"]:
                lines += ["", "**Вот что можно улучшить:**"] + [f"- {x}" for x in b["improve"]]
            if b["improved_answer"]:
                lines += ["", "**Вот удачный пример ответа:**", "", f"> {b['improved_answer']}"]
            if b.get("speech"):
                sl = _speech_lines(b["speech"])
                lines += ["", f"**{sl[0]}**"] + [f"- {x}" for x in sl[1:]]
            if b["follow_up"]:
                lines += ["", f"_Уточняющий вопрос:_ {b['follow_up']}"]
            src(b["sources"])
        elif t == "info_q":
            lines += ["", f"### Вопрос: {b['text']}", ""]
        elif t == "info_a":
            lines += [b["text"]]
            src(b["sources"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path
