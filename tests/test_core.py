"""Офлайн-тесты ядра: поиск, детектор, резервный оценщик, адаптивность, хранилище, экспорт.

Запуск: python -m pytest -q   (или просто python tests/test_core.py)
"""
import asyncio
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kompas import adaptive  # noqa: E402
from kompas.detector import detect  # noqa: E402
from kompas.feedback import Feedback, rule_based_feedback  # noqa: E402
from kompas.kb import KnowledgeBase  # noqa: E402
from kompas.llm import MentorLLM  # noqa: E402
from kompas.mentor import Mentor  # noqa: E402
from kompas.storage import Storage  # noqa: E402

KB = KnowledgeBase(ROOT / "data")


def test_kb_loaded():
    assert len(KB.entries) >= 30
    assert {e.scenario for e in KB.entries.values()} == {"admission", "job", "onboarding"}
    for e in KB.entries.values():
        assert e.sources, e.id
        for s in e.sources:
            assert s.url.startswith("https://"), s.url
    for scn in KB.scenarios:
        assert 3 <= len(KB.cases_for(scn)) <= 6, scn


def test_search_finds_relevant():
    cases = {
        "как расставить приоритеты при поступлении": "A02",
        "забыл подать согласие на зачисление": "A03",
        "испытательный срок полгода это законно?": "C01",
        "что ответить на вопрос расскажите о себе": "B04",
        "у меня нет опыта работы что говорить": "B05",
        "какие у вас слабые стороны": "B10",
        "целевой договор что проверить": "A07",
        "зарплатные ожидания на собеседовании": "B11",
    }
    for q, expected in cases.items():
        top = [e.id for e, _ in KB.search(q, k=3)]
        assert expected in top, (q, top)


def test_search_nothing_for_offtopic():
    assert KB.search("рецепт борща со сметаной") == []


def test_detector_strong_and_weak():
    case = KB.cases["CB2"]
    sig = {s.code: s.strength for s in detect("Честно, я ничего не умею, портфолио нет", case, KB)}
    assert sig.get("SELF_DEPRECATION") == "strong"
    assert sig.get("NO_PORTFOLIO") == "strong"
    # слабый сигнал «престиж» не должен срабатывать в кейсе, который его не проверяет
    assert "RATING_ONLY" not in {s.code for s in detect("Хочу в престижную компанию, давно мечтаю о работе у вас", case, KB)}
    assert "RATING_ONLY" in {s.code for s in detect("Выберу по престижу", KB.cases["CA3"], KB)}


def test_rule_feedback_bad_vs_good():
    case = KB.cases["CB5"]
    bad = rule_based_feedback("Не знаю, предложите сами. Вопросов нет.", case, detect("Не знаю, предложите сами. Вопросов нет.", case, KB), KB)
    good_text = case.model_answer
    good = rule_based_feedback(good_text, case, detect(good_text, case, KB), KB)
    assert "NO_SALARY_ANCHOR" in bad.errors and "NO_QUESTIONS" in bad.errors
    assert good.score > bad.score
    assert bad.follow_up
    assert bad.improved_answer == case.model_answer
    assert len(bad.correct) >= 1 and len(bad.improve) >= 1


def test_weak_zones_decay_and_credit():
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    old = (now - timedelta(days=60)).isoformat()
    fresh = now.isoformat()
    attempts = [
        {"case_id": "CB1", "errors": ["VAGUE"], "created_at": fresh},
        {"case_id": "CB3", "errors": ["VAGUE", "NO_STRUCTURE"], "created_at": fresh},
        {"case_id": "CB4", "errors": ["CRITICAL_WEAKNESS"], "created_at": old},
    ]
    zones = dict(adaptive.weak_zones(attempts, KB, now))
    assert zones.get("VAGUE", 0) > zones.get("NO_STRUCTURE", 0) > 0
    assert "CRITICAL_WEAKNESS" not in zones  # давняя ошибка затухла
    # чистая попытка по кейсу, который проверяет NO_STRUCTURE, снижает вес
    attempts2 = attempts + [{"case_id": "CB3", "errors": [], "created_at": fresh}] * 2
    zones2 = dict(adaptive.weak_zones(attempts2, KB, now))
    assert zones2.get("NO_STRUCTURE", 0) < zones.get("NO_STRUCTURE", 0)


def test_pick_case_targets_weak_zone():
    import random
    zones = [("NO_SALARY_ANCHOR", 2.0)]
    case = adaptive.pick_case("job", KB, zones, attempts=[], rng=random.Random(1))
    assert "NO_SALARY_ANCHOR" in case.tests
    # но не повторяет только что пройденный кейс
    case2 = adaptive.pick_case("job", KB, [], attempts=[{"case_id": case.id, "errors": [], "created_at": ""}], rng=random.Random(1))
    assert case2.id != case.id


def _run(coro):
    return asyncio.run(coro)


def test_mentor_end_to_end_without_llm():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = Storage(tmp / "t.db")
            await store.init()
            llm = MentorLLM(None, "x", 5, KB)  # без ключа → резервный режим
            mentor = Mentor(KB, llm, store, ROOT / "fonts", tmp)
            uid = 42
            await store.touch_user(uid)

            # справка
            sid_info = await store.start_session(uid, "info", "all")
            res = await mentor.info_answer("может ли испытательный срок быть полгода", None)
            assert res.engine == "kb" and res.sources and "C01" in res.text
            await store.log(sid_info, uid, "user", "info_q", "вопрос")
            await store.log(sid_info, uid, "bot", "info_a", res.text, {"sources": [s.__dict__ for s in res.sources]})

            # тренажёр
            sid = await store.start_session(uid, "train", "job")
            case, zone, _ = await mentor.choose_case(uid, "job")
            assert zone is None
            await store.log(sid, uid, "bot", "case", case.prompt, {"case_id": case.id})
            bad = "Я ничего не умею, вопросов нет, предложите сами"
            fb = await mentor.evaluate(uid, sid, case, bad)
            assert fb.engine == "rules" and "SELF_DEPRECATION" in fb.errors
            await store.log(sid, uid, "user", "answer", bad, {"case_id": case.id})
            await store.log(sid, uid, "bot", "feedback", "", {"case_id": case.id, "feedback": fb.to_dict()})

            zones = await mentor.weak_zones(uid)
            assert any(c == "SELF_DEPRECATION" for c, _ in zones)
            nxt, zone2, _ = await mentor.choose_case(uid, "job")
            assert zone2 is not None and zone2 in nxt.tests  # следующий кейс бьёт в слабую зону

            summary = await mentor.session_summary(uid, sid)
            assert summary["attempts"] == 1
            prog = await mentor.progress(uid)
            assert prog["total"] == 1

            pdf = await mentor.export(uid, "pdf", "all")
            md = await mentor.export(uid, "md", "last")
            assert pdf.exists() and pdf.stat().st_size > 5000
            assert pdf.read_bytes()[:4] == b"%PDF"
            text = md.read_text(encoding="utf-8")
            assert "Вот удачный пример ответа" in text and "Слабые зоны" in text

            await store.forget(uid)
            assert await mentor.progress(uid) is None
    _run(scenario())


def test_mentor_uses_llm_and_merges_strong_signals():
    class FakeLLM(MentorLLM):
        def __init__(self):
            super().__init__(None, "fake", 5, KB)

        @property
        def available(self):
            return True

        async def evaluate(self, case, answer, signals, weak_zones, previous=None):
            # модель «забыла» про сильный сигнал — ядро должно его добавить
            return Feedback(score=6, correct=["ok"], improve=["x [B05]"], improved_answer="y", errors=[],
                            kb_refs=["B05"], follow_up="А конкретнее?")

        async def answer_info(self, question, entries, history=None):
            return f"Ответ по базе [{entries[0].id}]."

    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            store = Storage(Path(tmp) / "t.db")
            await store.init()
            mentor = Mentor(KB, FakeLLM(), store, ROOT / "fonts", Path(tmp))
            sid = await store.start_session(1, "train", "job")
            fb = await mentor.evaluate(1, sid, KB.cases["CB2"], "я ничего не умею")
            assert fb.engine == "llm" and "SELF_DEPRECATION" in fb.errors
            res = await mentor.info_answer("как написать сопроводительное письмо", "job")
            assert res.engine == "llm" and res.sources
    _run(scenario())


if __name__ == "__main__":
    import inspect
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", name, repr(exc))
    sys.exit(1 if failed else 0)
