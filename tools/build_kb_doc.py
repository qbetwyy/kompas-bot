"""Собирает базу знаний в читаемый документ для жюри: docs/knowledge_base.md"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kompas.kb import KnowledgeBase  # noqa: E402


def main():
    kb = KnowledgeBase(ROOT / "data")
    raw = json.loads((ROOT / "data" / "knowledge_base.json").read_text(encoding="utf-8"))
    lines = ["# База знаний «Компаса»", "", f"_Версия: {raw['meta']['version']}_", "", raw["meta"]["note"], "",
             f"Всего записей: {len(kb.entries)}; кейсов тренажёра: {len(kb.cases)}; типов ошибок: {len(kb.errors)}.", ""]
    for scn, title in kb.scenarios.items():
        entries = [e for e in kb.entries.values() if e.scenario == scn]
        lines += [f"## {title} ({len(entries)} записей)", ""]
        for e in entries:
            lines += [f"### [{e.id}] {e.topic}", "", e.content, ""]
            if e.insider:
                lines += [f"**Внутренняя кухня.** {e.insider}", ""]
            lines += [f"**Типичная ошибка.** {e.mistake}", "", f"**Удачный пример.** {e.good_example}", "", "**Источники:**"]
            lines += [f"- [{s.title}]({s.url})" for s in e.sources]
            lines.append("")
        cases = kb.cases_for(scn)
        lines += [f"### Кейсы тренажёра: {title}", ""]
        for c in cases:
            tests = ", ".join(kb.errors[t].title for t in c.tests)
            lines += [f"- **{c.id}. {c.title}** (роль: {c.role}). Проверяет: {tests}. Опирается на: {', '.join(c.kb_refs)}."]
        lines.append("")
    lines += ["## Каталог типовых ошибок", "", "| Код | Ошибка | Как исправить | Записи базы |", "|---|---|---|---|"]
    for er in kb.errors.values():
        lines.append(f"| {er.code} | {er.title} | {er.fix} | {', '.join(er.kb_refs)} |")
    out = ROOT / "docs" / "knowledge_base.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
