"""База знаний, кейсы, каталог ошибок и поиск по базе (BM25 + русский стеммер).

Поиск сделан своими силами, без внешних сервисов: база небольшая (десятки записей),
поэтому лексический BM25 со стеммингом работает быстро, бесплатно и предсказуемо,
а найденные фрагменты затем передаются языковой модели как единственный источник фактов.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("russian")
_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так",
    "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "её", "мне", "было",
    "вот", "от", "меня", "еще", "ещё", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам",
    "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней",
    "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем",
    "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве",
    "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть",
    "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между", "это", "мне", "нужно",
}

SCENARIO_ICONS = {"admission": "🎓", "job": "💼", "onboarding": "🚀"}


def tokenize(text: str) -> list[str]:
    words = [w.lower().replace("ё", "е") for w in _WORD_RE.findall(text or "")]
    return [_STEMMER.stemWord(w) for w in words if w not in _STOPWORDS and len(w) > 1]


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class Entry:
    id: str
    scenario: str
    topic: str
    content: str
    mistake: str
    good_example: str
    tags: tuple[str, ...]
    sources: tuple[Source, ...]
    insider: str = ""

    def as_prompt_block(self, scenario_title: str) -> str:
        parts = [f"[{self.id}] ({scenario_title}) {self.topic}: {self.content}"]
        if self.insider:
            parts.append(f"Внутренняя кухня: {self.insider}")
        parts.append(f"Типичная ошибка: {self.mistake}")
        parts.append(f"Удачный пример: {self.good_example}")
        return "\n".join(parts)


@dataclass(frozen=True)
class Pattern:
    regex: re.Pattern
    strength: str  # "strong" | "weak"


@dataclass(frozen=True)
class ErrorType:
    code: str
    title: str
    description: str
    fix: str
    kb_refs: tuple[str, ...]
    patterns: tuple[Pattern, ...]


@dataclass(frozen=True)
class Case:
    id: str
    scenario: str
    title: str
    role: str
    prompt: str
    context: str
    tests: tuple[str, ...]
    ideal_points: tuple[str, ...]
    model_answer: str
    kb_refs: tuple[str, ...]


@dataclass
class _Bm25Index:
    docs: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    df: Counter = field(default_factory=Counter)
    avgdl: float = 0.0

    def __post_init__(self) -> None:
        for doc in self.docs:
            self.df.update(set(doc))
        self.avgdl = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        self._tf = [Counter(d) for d in self.docs]

    def score(self, query: list[str], idx: int) -> float:
        n = len(self.docs)
        tf = self._tf[idx]
        dl = len(self.docs[idx])
        total = 0.0
        for term in query:
            if term not in tf:
                continue
            idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
            freq = tf[term]
            total += idf * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return total


class KnowledgeBase:
    def __init__(self, data_dir: Path):
        kb_raw = json.loads((data_dir / "knowledge_base.json").read_text(encoding="utf-8"))
        cases_raw = json.loads((data_dir / "cases.json").read_text(encoding="utf-8"))
        errors_raw = json.loads((data_dir / "errors.json").read_text(encoding="utf-8"))

        self.meta: dict = kb_raw.get("meta", {})
        self.scenarios: dict[str, str] = kb_raw["scenarios"]
        self.entries: dict[str, Entry] = {}
        for e in kb_raw["entries"]:
            self.entries[e["id"]] = Entry(
                id=e["id"],
                scenario=e["scenario"],
                topic=e["topic"],
                content=e["content"],
                mistake=e["mistake"],
                good_example=e["good_example"],
                tags=tuple(e.get("tags", [])),
                sources=tuple(Source(s["title"], s["url"]) for s in e["sources"]),
                insider=e.get("insider", ""),
            )
        self.errors: dict[str, ErrorType] = {}
        for er in errors_raw["errors"]:
            self.errors[er["code"]] = ErrorType(
                code=er["code"],
                title=er["title"],
                description=er["description"],
                fix=er["fix"],
                kb_refs=tuple(er.get("kb_refs", [])),
                patterns=tuple(
                    Pattern(re.compile(p["re"], re.IGNORECASE), p.get("strength", "weak"))
                    for p in er.get("patterns", [])
                ),
            )
        self.cases: dict[str, Case] = {}
        for c in cases_raw["cases"]:
            self.cases[c["id"]] = Case(
                id=c["id"],
                scenario=c["scenario"],
                title=c["title"],
                role=c["role"],
                prompt=c["prompt"],
                context=c["context"],
                tests=tuple(c["tests"]),
                ideal_points=tuple(c["ideal_points"]),
                model_answer=c["model_answer"],
                kb_refs=tuple(c["kb_refs"]),
            )
        self._validate()

        self._order = list(self.entries.keys())
        docs = []
        for eid in self._order:
            e = self.entries[eid]
            # Заголовок и теги весят вдвое больше основного текста.
            text = " ".join([e.topic, e.topic, " ".join(e.tags), " ".join(e.tags), e.content, e.insider, e.mistake])
            docs.append(tokenize(text))
        self._index = _Bm25Index(docs)

    def _validate(self) -> None:
        for case in self.cases.values():
            for ref in case.kb_refs:
                assert ref in self.entries, f"Кейс {case.id}: нет записи базы {ref}"
            for code in case.tests:
                assert code in self.errors, f"Кейс {case.id}: нет кода ошибки {code}"
        for err in self.errors.values():
            for ref in err.kb_refs:
                assert ref in self.entries, f"Ошибка {err.code}: нет записи базы {ref}"

    # --- поиск -----------------------------------------------------------------
    def search(self, query: str, scenario: str | None = None, k: int = 4, min_score: float = 0.5) -> list[tuple[Entry, float]]:
        q = tokenize(query)
        if not q:
            return []
        scored = []
        for idx, eid in enumerate(self._order):
            entry = self.entries[eid]
            s = self._index.score(q, idx)
            if scenario and entry.scenario != scenario:
                s *= 0.6  # другие сценарии не исключаем полностью, но понижаем
            if s >= min_score:
                scored.append((entry, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # --- утилиты ---------------------------------------------------------------
    def scenario_title(self, scenario: str) -> str:
        return self.scenarios.get(scenario, scenario)

    def entries_by_ids(self, ids) -> list[Entry]:
        return [self.entries[i] for i in ids if i in self.entries]

    def sources_for(self, ids) -> list[Source]:
        seen, result = set(), []
        for entry in self.entries_by_ids(ids):
            for src in entry.sources:
                if src.url not in seen:
                    seen.add(src.url)
                    result.append(src)
        return result

    def cases_for(self, scenario: str) -> list[Case]:
        return [c for c in self.cases.values() if c.scenario == scenario]
