"""Работа с языковой моделью (Anthropic API).

Два вызова:
* answer_info, ответ в справочном режиме строго по найденным фрагментам базы;
* evaluate, оценка ответа в тренажёре со структурированным результатом
  (через принудительный вызов инструмента submit_feedback, чтобы не парсить свободный текст).

Статичная часть системного промпта помечена cache_control это prompt caching,
он заметно удешевляет повторные запросы.
"""
from __future__ import annotations

import logging
import re

from .detector import Signal
from .feedback import Feedback
from .kb import Case, Entry, KnowledgeBase

log = logging.getLogger(__name__)


_KB_REF = re.compile(r"\s*[\[(](?:[A-Z]\d{2})(?:[,;\s]+[A-Z]\d{2})*[\])]")


def plain(text: str) -> str:
    """Убирает то, что выдаёт нейросеть и служебную кухню: тире, коды статей вроде [A03], квадратные скобки."""
    t = _KB_REF.sub("", text or "")
    t = t.replace("[", "(").replace("]", ")")
    return t.replace(" — ", " - ").replace("—", "-").replace(" – ", " - ").replace("–", "-")

MENTOR_PERSONA = """Ты Компас, наставник для абитуриентов и стажёров 15-20 лет. По характеру ты старший товарищ, \
который сам недавно поступал, искал первую стажировку и знает, как всё устроено на деле, а не только по правилам.

Как писать:
- Как живой человек в мессенджере: на «ты», просто, конкретно, короткими предложениями.
- Без канцелярита и без шаблонов нейросети: никаких «отличный вопрос», «важно отметить», «не только…, но и…», \
«давай разберёмся», «в заключение», никаких троек прилагательных и пафоса.
- Без эмодзи. Списки только если без них правда хуже читается.
- Не используй длинное тире (—) и короткое (–) вообще. Вместо них точка, запятая или двоеточие.
- Не пиши в тексте коды статей базы вроде [A03] и не используй квадратные скобки: ссылки на источники бот добавит сам.
- Можно мягкую прямоту: «тут ты себя недооцениваешь», «так лучше не говорить».

Жёсткие правила:
1. Факты бери только из переданных фрагментов базы знаний. Если ответа там нет, честно скажи, что не знаешь, \
и подскажи, где проверить: правила приёма вуза, приёмная комиссия, Трудовой кодекс, HR компании.
2. После каждого факта ставь ссылку на фрагмент в квадратных скобках, например [A03]. Не выдумывай ссылки и источники \
и не пиши URL: бот сам добавит список источников.
3. Не обещай результат (поступление, оффер). Советы информационные и не заменяют официальные требования вузов \
и работодателей.
4. Не спрашивай персональные данные (ФИО, адрес, телефон, номера документов)."""

INFO_TASK = """Режим «Справка». Ответь на вопрос по фрагментам базы знаний: 3-6 коротких предложений, по делу, \
в конце один практический совет своими словами. Если вопрос не про поступление, стажировки или первые месяцы работы, \
коротко скажи, с чем можешь помочь. Без заголовков, таблиц и эмодзи."""


def _evaluation_task(kb: KnowledgeBase) -> str:
    catalog = "\n".join(f"- {e.code}: {e.title}. {e.description}" for e in kb.errors.values())
    return f"""Режим «Тренажёр-экзаменатор». Ты оцениваешь ответ пользователя на проблемный кейс и даёшь обратную связь, \
как опытный наставник: по-доброму, но честно. Ошибки называй прямо и мягко, не ругай и не льсти. \
Пиши пункты как короткие живые фразы, а не как отчёт: «Хорошо, что сразу назвал проект», а не «Корректно указан релевантный опыт».

Каталог типовых ошибок:
{catalog}

Заполни инструмент submit_feedback:
- correct, 1-3 конкретных пункта, что в ответе действительно верно. Не выдумывай достоинств: если верного почти нет, \
отметь самое полезное, что есть (например, что пользователь честно ответил).
- improve, 1-3 конкретных пункта, что улучшить; в каждом пункте ссылка на фрагмент базы, например [B05].
- improved_answer, «удачный пример ответа»: улучшенная версия ответа пользователя от первого лица, 3-6 предложений. \
Сохраняй его факты. Не приписывай пользователю опыт, которого он не упоминал. Вместо этого ставь заглушки \
в квадратных скобках, например [твой проект].
- errors, коды из каталога, которые действительно есть в ответе. Сигналы правил, подсказка, а не истина: \
подтверди их или отклони.
- kb_refs, 1-3 идентификатора фрагментов базы, на которые ты опирался.
- follow_up, один неудобный уточняющий вопрос, как задал бы живой собеседник из роли кейса, по самому слабому месту \
ответа. Если у пользователя есть слабые зоны из прошлых сессий, по возможности целься в них. \
Если ответ отличный, задай вопрос на углубление.
- score, оценка от 0 до 10."""


def _feedback_tool(kb: KnowledgeBase, ref_ids: list[str]) -> dict:
    return {
        "name": "submit_feedback",
        "description": "Вернуть структурированную обратную связь по ответу пользователя.",
        "input_schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "correct": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                "improve": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                "improved_answer": {"type": "string"},
                "errors": {"type": "array", "items": {"type": "string", "enum": list(kb.errors.keys())}},
                "kb_refs": {"type": "array", "items": {"type": "string", "enum": ref_ids}},
                "follow_up": {"type": "string"},
            },
            "required": ["score", "correct", "improve", "improved_answer", "errors", "kb_refs", "follow_up"],
        },
    }


class MentorLLM:
    def __init__(self, api_key: str | None, model: str, timeout: float, kb: KnowledgeBase,
                 workspace_id: str | None = None):
        self.kb = kb
        self.model = model
        self._client = None
        if api_key:
            from anthropic import AsyncAnthropic

            headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
            self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2, default_headers=headers)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _system(self, task: str) -> list[dict]:
        return [{"type": "text", "text": MENTOR_PERSONA + "\n\n" + task, "cache_control": {"type": "ephemeral"}}]

    async def answer_info(self, question: str, entries: list[Entry], history: list[tuple[str, str]] | None = None) -> str:
        assert self._client is not None
        kb_text = "\n\n".join(e.as_prompt_block(self.kb.scenario_title(e.scenario)) for e in entries) or "(ничего не найдено)"
        messages = []
        for q, a in (history or [])[-2:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({
            "role": "user",
            "content": f"Фрагменты базы знаний:\n{kb_text}\n\nВопрос пользователя: {question}",
        })
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self._system(INFO_TASK),
            messages=messages,
        )
        return plain("".join(getattr(b, "text", "") for b in resp.content).strip())

    async def evaluate(
        self,
        case: Case,
        answer: str,
        signals: list[Signal],
        weak_zones: list[tuple[str, float]],
        previous: tuple[str, str] | None = None,
    ) -> Feedback:
        """previous = (предыдущий ответ, уточняющий вопрос), если это ответ на follow-up."""
        assert self._client is not None
        ref_ids = list(dict.fromkeys(list(case.kb_refs) + [r for c in case.tests for r in self.kb.errors[c].kb_refs]))
        entries = self.kb.entries_by_ids(ref_ids)
        kb_text = "\n\n".join(e.as_prompt_block(self.kb.scenario_title(e.scenario)) for e in entries)
        signals_text = "\n".join(f"- {s.code} ({s.strength}): «{s.match}»" for s in signals) or "- нет"
        weak_text = "\n".join(f"- {c}: {self.kb.errors[c].title}" for c, _ in weak_zones[:3] if c in self.kb.errors) or "- нет данных"
        points = "\n".join(f"- {p}" for p in case.ideal_points)

        parts = [
            f"Сценарий: {self.kb.scenario_title(case.scenario)}",
            f"Кейс «{case.title}». Роль собеседника: {case.role}.",
            f"Вопрос пользователю:\n{case.prompt}",
            f"Что отличает сильный ответ (для тебя, не цитируй дословно): {case.context}",
            f"Ключевые пункты:\n{points}",
            f"Проверяемые ошибки в этом кейсе: {', '.join(case.tests)}",
            f"Фрагменты базы знаний:\n{kb_text}",
            f"Сигналы правил:\n{signals_text}",
            f"Слабые зоны пользователя по прошлым сессиям:\n{weak_text}",
        ]
        if previous:
            parts.append(f"Предыдущий ответ пользователя: «{previous[0]}»\nТвой уточняющий вопрос был: «{previous[1]}»\n"
                         "Сейчас пользователь отвечает на уточняющий вопрос, оценивай этот ответ в контексте предыдущего.")
        parts.append(f"Ответ пользователя: «{answer}»")
        parts.append("Верни результат только через вызов инструмента submit_feedback.")

        kwargs = dict(
            model=self.model,
            max_tokens=4000,
            system=self._system(_evaluation_task(self.kb)),
            tools=[_feedback_tool(self.kb, ref_ids)],
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
        )
        try:
            resp = await self._client.messages.create(**kwargs, tool_choice={"type": "tool", "name": "submit_feedback"})
        except Exception as exc:  # некоторые режимы модели не принимают принудительный выбор инструмента
            if "tool_choice" not in str(exc) and "thinking" not in str(exc):
                raise
            log.warning("Принудительный tool_choice отклонён (%s), повторяю с auto", exc)
            resp = await self._client.messages.create(**kwargs, tool_choice={"type": "auto"})
        data = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                data = dict(block.input)
                break
        if not data:
            raise RuntimeError("Модель не вернула submit_feedback")
        return Feedback(
            score=int(max(0, min(10, data.get("score", 5)))),
            correct=[plain(str(x)) for x in data.get("correct", [])][:3],
            improve=[plain(str(x)) for x in data.get("improve", [])][:3],
            improved_answer=plain(str(data.get("improved_answer", "")).strip()) or case.model_answer,
            errors=[c for c in data.get("errors", []) if c in self.kb.errors],
            kb_refs=[r for r in data.get("kb_refs", []) if r in self.kb.entries] or list(case.kb_refs),
            follow_up=plain(str(data.get("follow_up", "")).strip()),
            engine="llm",
        )
