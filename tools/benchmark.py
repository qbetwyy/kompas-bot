"""Мини-бенчмарк: насколько хорошо бот находит типовые ошибки в ответах.

Режимы:
  python tools/benchmark.py           , только правила (без API, бесплатно)
  python tools/benchmark.py --llm     , полный гибрид «правила + модель» (нужен ANTHROPIC_API_KEY в .env)

Результат печатается и сохраняется в docs/benchmark_<режим>.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kompas.detector import detect  # noqa: E402
from kompas.feedback import merge_strong_signals  # noqa: E402
from kompas.kb import KnowledgeBase  # noqa: E402


def metrics(rows):
    tp = fp = fn = 0
    per_code = defaultdict(lambda: [0, 0, 0])
    clean_ok = clean_total = flagged_ok = flagged_total = 0
    for exp, pred in rows:
        exp, pred = set(exp), set(pred)
        for c in pred & exp:
            tp += 1
            per_code[c][0] += 1
        for c in pred - exp:
            fp += 1
            per_code[c][1] += 1
        for c in exp - pred:
            fn += 1
            per_code[c][2] += 1
        if not exp:
            clean_total += 1
            clean_ok += int(not pred)
        else:
            flagged_total += 1
            flagged_ok += int(bool(pred & exp))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": p, "recall": r, "f1": f1,
        "clean_acc": clean_ok / clean_total if clean_total else 0.0,
        "flagged_acc": flagged_ok / flagged_total if flagged_total else 0.0,
        "per_code": dict(per_code), "n": len(rows), "clean_total": clean_total, "flagged_total": flagged_total,
    }


def report(name: str, m: dict) -> str:
    lines = [
        f"# Мини-бенчмарк «Компаса»: {name}", "",
        f"Ответов в наборе: {m['n']} (с ошибками, {m['flagged_total']}, хороших, {m['clean_total']}).", "",
        "| Метрика | Значение |", "|---|---|",
        f"| Точность (precision) по типам ошибок | {m['precision']:.0%} |",
        f"| Полнота (recall) по типам ошибок | {m['recall']:.0%} |",
        f"| F1 | {m['f1']:.2f} |",
        f"| Плохой ответ: найдена хотя бы одна верная ошибка | {m['flagged_acc']:.0%} |",
        f"| Хороший ответ: бот не придрался зря | {m['clean_acc']:.0%} |",
        "", "| Код ошибки | Верно найдено | Лишних | Пропущено |", "|---|---|---|---|",
    ]
    for code, (tp, fp, fn) in sorted(m["per_code"].items()):
        lines.append(f"| {code} | {tp} | {fp} | {fn} |")
    return "\n".join(lines) + "\n"


async def run_llm(kb, items):
    from config import load_settings  # noqa: WPS433
    from kompas.llm import MentorLLM
    settings = load_settings()
    if not settings.anthropic_api_key:
        raise SystemExit("Нет ANTHROPIC_API_KEY в .env")
    llm = MentorLLM(settings.anthropic_api_key, settings.model, settings.llm_timeout, kb, settings.anthropic_workspace_id)
    rows = []
    for i, it in enumerate(items, 1):
        case = kb.cases[it["case"]]
        signals = detect(it["answer"], case, kb)
        fb = await llm.evaluate(case, it["answer"], signals, [])
        fb = merge_strong_signals(fb, signals)
        rows.append((it["expected"], fb.errors))
        print(f"{i}/{len(items)} {case.id}: ожидалось {it['expected']}, найдено {fb.errors}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()
    kb = KnowledgeBase(ROOT / "data")
    items = json.loads((ROOT / "data" / "eval_set.json").read_text(encoding="utf-8"))["items"]
    if args.llm:
        rows = asyncio.run(run_llm(kb, items))
        name, fname = "правила + языковая модель", "benchmark_hybrid.md"
    else:
        rows = [(it["expected"], [s.code for s in detect(it["answer"], kb.cases[it["case"]], kb)]) for it in items]
        name, fname = "только правила (без модели)", "benchmark_rules.md"
    m = metrics(rows)
    text = report(name, m)
    out = ROOT / "docs" / fname
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
