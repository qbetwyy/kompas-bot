"""Модель обратной связи и правиловый (резервный) оценщик.

Резервный оценщик нужен, чтобы бот не «падал», если API языковой модели недоступен:
он сравнивает ответ с ключевыми пунктами кейса по совпадению основ слов,
добавляет найденные детектором ошибки и показывает эталонный ответ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .detector import Signal
from .kb import Case, KnowledgeBase, tokenize


@dataclass
class Feedback:
    score: int
    correct: list[str]
    improve: list[str]
    improved_answer: str
    errors: list[str]
    kb_refs: list[str]
    follow_up: str = ""
    engine: str = "llm"  # "llm" | "rules"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Feedback":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k in data})


# Уточняющие «неудобные» вопросы для резервного режима.
FOLLOW_UPS = {
    "SELF_DEPRECATION": "Хорошо, а если всё-таки назвать одну вещь, которую ты сделал сам от начала до конца,, что это?",
    "NO_PORTFOLIO": "Что конкретно я могу открыть и посмотреть прямо сейчас, чтобы убедиться в твоих навыках?",
    "OVERCLAIM": "Окей, давай проверим: какую самую сложную задачу из этого ты делал сам, без туториала?",
    "VAGUE": "Звучит общо. Приведи один конкретный пример, что именно ты сделал и чем это закончилось?",
    "NO_STRUCTURE": "А какой был итог? Что изменилось благодаря именно твоим действиям?",
    "NO_RESEARCH": "А что ты знаешь о нас (или об этой программе) конкретно? Назови хотя бы один факт.",
    "NO_MOTIVATION": "Чем это место лучше трёх других похожих вариантов, лично для тебя?",
    "RATING_ONLY": "Хорошо, а что ты знаешь об учебном плане и практике на этой программе?",
    "NO_BACKUP": "А если не пройдёшь, какой у тебя план Б, конкретно?",
    "DEADLINE_RISK": "Какого числа и до скольки нужно это сделать? Что будет, если опоздаешь на час?",
    "PRIORITY_BY_CHANCES": "Представь, что прошёл на обе программы. Куда тебя зачислят при такой расстановке, и туда ли ты хотел?",
    "CRITICAL_WEAKNESS": "А если серьёзно: что у тебя реально получается хуже, чем хотелось бы, и что ты с этим делаешь?",
    "NO_SALARY_ANCHOR": "Допустим, я настаиваю: назови одну цифру. На что ты будешь опираться?",
    "NO_QUESTIONS": "Неужели тебе не интересно, по каким критериям мы будем оценивать тебя через три месяца?",
    "SILENT_PROBLEMS": "Сколько времени ты готов потратить сам, прежде чем пойти за помощью? Почему именно столько?",
    "NO_EXPECTATIONS": "А как ты поймёшь через месяц, что справляешься, если не спросишь об этом сейчас?",
    "RUSH_DECISIONS": "А ты уверен, что знаешь, почему это сделано именно так? Что, если у этого есть причина?",
    "RIGHTS_UNAWARE": "А что говорит Трудовой кодекс о длительности испытательного срока и зарплате в этот период?",
}


_META = {"назыв", "понима", "показ", "говор", "спраш", "выясн", "уточн", "предла", "есть", "имеет", "знает",
         "конкр", "хотя", "желат", "связы", "трезв", "оцени", "учиты", "обращ", "приход", "опира", "задаё", "задае"}


def _prefixes(text: str) -> set[str]:
    return {t[:5] for t in tokenize(text)}


def _coverage(answer_pref: set[str], point: str) -> float:
    pref = _prefixes(point) - _META
    if not pref:
        return 0.0
    return len(pref & answer_pref) / len(pref)


def rule_based_feedback(answer: str, case: Case, signals: list[Signal], kb: KnowledgeBase) -> Feedback:
    answer_pref = _prefixes(answer)
    covered, missed = [], []
    for point in case.ideal_points:
        (covered if _coverage(answer_pref, point) >= 0.34 else missed).append(point)
    model_pref = _prefixes(case.model_answer)
    similarity = len(answer_pref & model_pref) / max(len(model_pref), 1)

    errors = []
    for s in signals:
        if s.code not in errors:
            errors.append(s.code)

    correct = [f"Есть: {p.lower()}" for p in covered[:3]]
    if not correct:
        correct = ["Ты ответил на вопрос и не ушёл от темы это база, от неё и будем отталкиваться."]

    improve = []
    for code in errors[:2]:
        err = kb.errors[code]
        ref = f" [{err.kb_refs[0]}]" if err.kb_refs else ""
        improve.append(f"{err.title}: {err.fix}{ref}")
    for p in missed:
        if len(improve) >= 3:
            break
        improve.append(f"Не хватает: {p.lower()}.")
    if not improve:
        improve = ["Добавь одну цифру или конкретный факт, ответ станет убедительнее."]

    strong = sum(1 for s in signals if s.strength == "strong")
    ratio = len(covered) / max(len(case.ideal_points), 1)
    score = round(10 * (0.6 * ratio + 0.4 * min(1.0, similarity * 1.5))) - 2 * strong - (1 if errors else 0)
    score = max(0, min(10, score))

    follow_up = FOLLOW_UPS.get(errors[0], "") if errors else ""
    if not follow_up and missed:
        follow_up = "А что насчёт этого: " + missed[0].lower() + "?"

    refs = list(dict.fromkeys(list(case.kb_refs) + [r for c in errors for r in kb.errors[c].kb_refs]))[:4]
    return Feedback(
        score=score,
        correct=correct,
        improve=improve,
        improved_answer=case.model_answer,
        errors=errors,
        kb_refs=refs,
        follow_up=follow_up,
        engine="rules",
    )


def merge_strong_signals(fb: Feedback, signals: list[Signal]) -> Feedback:
    """Сильные сигналы детектора добавляются к ошибкам модели, даже если она их пропустила."""
    for s in signals:
        if s.strength == "strong" and s.code not in fb.errors:
            fb.errors.append(s.code)
    return fb
