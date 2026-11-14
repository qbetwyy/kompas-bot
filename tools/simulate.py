"""Симуляция на виртуальных кандидатах: python tools/simulate.py

Это не данные живых пользователей, а проверка механики бота на синтетике. Две части:

1. Адаптивность. Виртуальный кандидат стабильно ошибается в чём-то одном (например, «нет запасного
   плана»). Его ответы берутся из размеченного набора data/eval_set.json, ошибки ищет настоящий
   правиловый детектор бота (без языковой модели, то есть по нижней границе качества). Сравниваем,
   как часто следующий вопрос попадает в слабое место: адаптивный выбор бота против случайного.

2. Подсчёт слов-паразитов. Фразы с заранее известным числом паразитов, включая ловушки, где то же
   слово употреблено по делу («вопрос такого типа», «как бы ты поступил»). Проверяем анализатор речи.

Результат печатается и сохраняется в docs/simulation.md. Случайность зафиксирована (seed), прогон
повторяемый: те же цифры получатся у любого, кто запустит скрипт.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kompas.adaptive import pick_case, weak_zones  # noqa: E402
from kompas.detector import detect  # noqa: E402
from kompas.kb import KnowledgeBase  # noqa: E402
from kompas.speech import count_fillers  # noqa: E402

RUNS = 300          # прогонов на каждого виртуального кандидата
QUESTIONS = 6       # вопросов за одну тренировку
P_ERROR = 0.8       # насколько стабильно кандидат допускает свою ошибку, если вопрос её проверяет
P_NOISE = 0.15      # шанс случайно ошибиться в чём-то другом
SEED = 2026


def load_eval():
    data = json.loads((ROOT / "data" / "eval_set.json").read_text(encoding="utf-8"))["items"]
    good, bad = {}, defaultdict(list)
    for it in data:
        if it["expected"]:
            bad[it["case"]].append(it)
        else:
            good[it["case"]] = it["answer"]
    return good, bad


def personas(kb: KnowledgeBase, bad):
    """Слабые места, на которых адаптивность вообще можно проверить: ошибку проверяют хотя бы 2 вопроса
    сценария, и в размеченном наборе есть ответ с этой ошибкой."""
    out = []
    for scenario in ("admission", "job", "onboarding"):
        cases = kb.cases_for(scenario)
        codes = defaultdict(list)
        for c in cases:
            for code in c.tests:
                if code != "VAGUE":
                    codes[code].append(c.id)
        for code, ids in codes.items():
            has_example = any(code in it["expected"] for cid in ids for it in bad.get(cid, []))
            if len(ids) >= 2 and has_example:
                out.append((scenario, code, len(ids), len(cases)))
    return out


def answer(case, weakness, good, bad, rng):
    """Ответ виртуального кандидата на вопрос: (текст, ошибки по ручной разметке)."""
    if weakness in case.tests and rng.random() < P_ERROR:
        pool = [it for it in bad.get(case.id, []) if weakness in it["expected"]]
        if pool:
            it = rng.choice(pool)
            return it["answer"], it["expected"]
    if rng.random() < P_NOISE and bad.get(case.id):
        it = rng.choice(bad[case.id])
        return it["answer"], it["expected"]
    return good[case.id], []


def session(kb, scenario, weakness, good, bad, rng, adaptive: bool, oracle: bool):
    attempts: list[dict] = []  # как в базе бота: новые сверху
    hits = 0
    first_named = None
    cases = kb.cases_for(scenario)
    now = datetime.now(timezone.utc).isoformat()
    for q in range(QUESTIONS):
        zones = weak_zones(attempts, kb)
        if q and zones and zones[0][0] == weakness and first_named is None:
            first_named = q
        if adaptive:
            case = pick_case(scenario, kb, zones, attempts, rng)
        else:
            last = attempts[0]["case_id"] if attempts else None
            case = rng.choice([c for c in cases if c.id != last] or cases)
        if q and weakness in case.tests:
            hits += 1
        text, labels = answer(case, weakness, good, bad, rng)
        # oracle: ошибки найдены верно (ориентир для бота с Claude); иначе только правиловый детектор
        errors = sorted(labels) if oracle else sorted({s.code for s in detect(text, case, kb)})
        attempts.insert(0, {"case_id": case.id, "errors": errors, "created_at": now})
    return hits / (QUESTIONS - 1), first_named


def run_adaptivity(kb, oracle: bool):
    good, bad = load_eval()
    rng = random.Random(SEED)
    rows = []
    for scenario, code, n_testing, n_cases in personas(kb, bad):
        res = {}
        for adaptive in (True, False):
            shares, named = [], []
            for _ in range(RUNS):
                share, first = session(kb, scenario, code, good, bad, rng, adaptive, oracle)
                shares.append(share)
                named.append(first)
            res[adaptive] = (sum(shares) / RUNS, sum(1 for n in named if n is not None) / RUNS,
                             [n for n in named if n is not None])
        found = res[True][2]
        rows.append({
            "scenario": kb.scenario_title(scenario), "code": code, "title": kb.errors[code].title,
            "share": f"{n_testing} из {n_cases}",
            "adaptive": res[True][0], "random": res[False][0],
            "named": res[True][1], "named_after": (sum(found) / len(found)) if found else None,
        })
    return rows


# Фразы для проверки подсчёта паразитов: (текст, сколько паразитов должно насчитаться).
# Правила бота: «ну, типа, короче, блин, как бы, в общем» считаются всегда; «вот, значит, то есть,
# на самом деле» только при повторе (2+ раза); «э-э-э» подряд считается одной запинкой.
FILLER_SET = [
    ("Я студент первого курса, учусь на инноватике и делаю своего бота.", 0),
    ("Ну я студент первого курса, учусь на инноватике.", 1),
    ("Ну, типа, я не знаю, что сказать.", 2),
    ("Короче, блин, опыта нет, но я быстро учусь.", 2),
    ("Как бы я сделал проект, в общем, он работает.", 2),
    ("Э-э-э, ну, я думаю, что справлюсь.", 2),
    ("Вот мой проект, я делал его два месяца.", 0),
    ("Вот, я сделал проект, вот, и он работает, вот.", 3),
    ("Значит, я пришёл, значит, и сразу начал.", 2),
    ("То есть я хочу сказать, что люблю задачи на анализ данных.", 0),
    ("На самом деле, я, на самом деле, хочу развиваться в ИИ.", 2),
    ("Мне нравятся задачи такого типа, где есть данные и метрики.", 1),   # ловушка: «типа» по делу
    ("Как бы ты поступил на моём месте, спросил я наставника.", 1),       # ловушка: «как бы» по делу
    ("Ну что ж, давайте начнём с моего проекта.", 1),
    ("Я подготовился: изучил компанию, вакансию и отзывы стажёров.", 0),
    ("Типа я умею Python, ну, и немного SQL, короче, всё.", 3),
    ("Э-э, м-м, я не уверен.", 2),
    ("Короче говоря, у меня есть опыт с Python.", 1),                    # ловушка: оборот вместо паразита
    ("В общем и целом, мне подходят ваши условия.", 1),                  # ловушка: оборот вместо паразита
    ("Блин, забыл, ну, типа, как бы, ладно.", 4),
]
# Сколько паразитов «на самом деле» в ловушках с точки зрения человека: 0 у «типа»/«как бы»/«короче
# говоря»/«в общем и целом» по делу, 1 у «Ну что ж» спорно. Ниже считаем расхождение с ручной разметкой.
HUMAN = {11: 0, 12: 0, 13: 1, 17: 0, 18: 0}


def run_fillers():
    rows = []
    for i, (text, rule_expected) in enumerate(FILLER_SET):
        got = sum(count_fillers(text).values())
        human = HUMAN.get(i, rule_expected)
        rows.append({"text": text, "got": got, "human": human, "trap": i in HUMAN})
    return rows


def adapt_table(adapt) -> list[str]:
    lines = [
        "| Сценарий | Слабое место | Вопросов на эту тему | Адаптивный выбор | Случайный выбор | Бот назвал слабое место | После скольких вопросов |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in adapt:
        after = f"{r['named_after']:.1f}" if r["named_after"] is not None else "-"
        lines.append(f"| {r['scenario']} | {r['title']} | {r['share']} | **{r['adaptive']*100:.0f}%** | "
                     f"{r['random']*100:.0f}% | {r['named']*100:.0f}% прогонов | {after} |")
    return lines


def summary(adapt):
    avg_a = sum(r["adaptive"] for r in adapt) / len(adapt)
    avg_r = sum(r["random"] for r in adapt) / len(adapt)
    named = [r["named_after"] for r in adapt if r["named_after"] is not None]
    return avg_a, avg_r, (sum(named) / len(named) if named else None)


def render(oracle, rules, fillers) -> str:
    oa, orr, on = summary(oracle)
    ra, rr, _ = summary(rules)
    exact = sum(1 for r in fillers if r["got"] == r["human"])
    traps = [r for r in fillers if r["trap"]]
    traps_ok = sum(1 for r in traps if r["got"] == r["human"])
    plain = [r for r in fillers if not r["trap"]]
    plain_ok = sum(1 for r in plain if r["got"] == r["human"])
    lines = [
        "# Симуляция на виртуальных кандидатах",
        "",
        "Это **не данные живых пользователей**, а проверка механики бота на синтетике. "
        "Скрипт: `tools/simulate.py`, прогон повторяемый (фиксированный seed).",
        "",
        "## 1. Попадает ли бот в слабое место",
        "",
        f"Виртуальный кандидат стабильно ({int(P_ERROR*100)}% случаев) ошибается в чём-то одном. "
        "Его ответы взяты из размеченного набора `data/eval_set.json`. "
        f"Тренировка из {QUESTIONS} вопросов, {RUNS} прогонов на каждого кандидата. "
        "Считаем, какая доля вопросов со 2-го по последний проверяет его слабое место. "
        "Для сравнения тот же кандидат с ботом, который задаёт вопросы случайно.",
        "",
        f"**Итог:** когда ошибки распознаны верно, адаптивный выбор попадает в слабое место в **{oa*100:.0f}%** "
        f"вопросов против **{orr*100:.0f}%** при случайном выборе (в {oa/orr:.1f} раза чаще)"
        + (f", а само слабое место бот называет в среднем после {on:.1f} вопроса." if on else "."),
        "",
        "### Ошибки распознаны верно (так работает бот с Claude)",
        "",
        "Ошибки берутся из ручной разметки ответа. Это ориентир для режима, когда разбор делает языковая модель.",
        "",
        *adapt_table(oracle),
        "",
        "### Только правила, без языковой модели (запасной режим)",
        "",
        "Ошибки ищет правиловый детектор бота. Это нижняя граница: так бот работает, если API недоступен. "
        f"Здесь адаптивный выбор даёт {ra*100:.0f}% против {rr*100:.0f}%: часть ошибок (структура ответа, "
        "мотивация, подготовка) правила не замечают, и бот просто не узнаёт о слабом месте.",
        "",
        *adapt_table(rules),
        "",
        "Почему даже в лучшем случае не 100%: бот не задаёт один и тот же вопрос два раза подряд, "
        "а вопросов на одно слабое место в сценарии всего 2-3.",
        "",
        "Что поменяли по итогам первого прогона: раньше выбор вопроса слишком сильно избегал недавних вопросов "
        "и почти не отличался от случайного. Теперь запрещён только повтор подряд, а вес слабых зон увеличен.",
        "",
        "## 2. Подсчёт слов-паразитов",
        "",
        f"{len(fillers)} фраз с заранее известным числом паразитов. {len(traps)} из них ловушки, где то же слово "
        "употреблено по делу («задачи такого типа», «как бы ты поступил», «короче говоря»).",
        "",
        f"* Обычные фразы: **{plain_ok} из {len(plain)}** посчитаны точно.",
        f"* Ловушки: **{traps_ok} из {len(traps)}**. Анализатор считает слово паразитом по форме, а не по смыслу, "
        "и это честное ограничение: «вопросы такого типа» он примет за паразита.",
        f"* Всего точное совпадение с ручной разметкой: **{exact} из {len(fillers)}**.",
        "",
        "| Фраза | Насчитал бот | Ручная разметка |",
        "|---|---|---|",
    ]
    for r in fillers:
        mark = " (ловушка)" if r["trap"] else ""
        lines.append(f"| {r['text']}{mark} | {r['got']} | {r['human']} |")
    lines += [
        "",
        "Отдельно: качество подсчёта в живом голосовом зависит ещё и от распознавания речи "
        "(faster-whisper). Если модель расслышала слово неправильно, анализатор его не увидит.",
        "",
        "## 3. Поиск типовых ошибок",
        "",
        "Отдельный тест на тех же 42 размеченных ответах, только правила без модели: "
        "см. `docs/benchmark_rules.md` (`python tools/benchmark.py`).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    kb = KnowledgeBase(ROOT / "data")
    oracle = run_adaptivity(kb, oracle=True)
    rules = run_adaptivity(kb, oracle=False)
    fillers = run_fillers()
    md = render(oracle, rules, fillers)
    (ROOT / "docs" / "simulation.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
