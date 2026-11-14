"""Тесты «ярких» фишек: разбор подачи и карта компетенций."""
import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kompas import competencies  # noqa: E402
from kompas.kb import KnowledgeBase  # noqa: E402
from kompas.llm import MentorLLM  # noqa: E402
from kompas.mentor import Mentor  # noqa: E402
from kompas.speech import analyze, count_fillers  # noqa: E402
from kompas.storage import Storage  # noqa: E402

KB = KnowledgeBase(ROOT / "data")


def _words(text, start=0.5, step=0.4, pauses=None):
    """Синтетические таймкоды: слово длится 0.3 с, шаг step; pauses = {индекс_слова: доп. пауза перед ним}."""
    out, t = [], start
    for i, w in enumerate(text.split()):
        t += (pauses or {}).get(i, 0)
        out.append((t, t + 0.3, w))
        t += step
    return out


def test_fillers_hard_soft_and_hesitations():
    c = count_fillers("Ну, э-э-э, типа я сделал бота, как бы, ну вот. Вот так.")
    assert c["ну"] == 2 and c["типа"] == 1 and c["как бы"] == 1
    assert c["э-э / м-м"] == 1          # «э-э-э» — одна запинка
    assert c["вот"] == 2                # мягкий паразит засчитан, т.к. 2+ раза
    assert "вот" not in count_fillers("Вот мой проект на GitHub.")  # один раз — нормальное слово


def test_clean_fast_and_messy_speech():
    clean_text = "Я студент первого курса и полгода делаю бота для подготовки к экзамену он проверяет ответы через языковую модель"
    clean = analyze(clean_text, _words(clean_text, step=0.42))
    assert clean.fillers_total == 0 and clean.pace_verdict == "комфортно" and clean.score == 10

    fast = analyze(clean_text, _words(clean_text, step=0.25))
    assert fast.pace_verdict == "быстро" and fast.score < 10

    messy_text = "ну короче я типа ну сделал как бы бота ну и типа всё короче"
    messy = analyze(messy_text, _words(messy_text, pauses={6: 3.2}, start=4.0))
    assert messy.fillers_per100 > 30
    assert messy.long_pauses == 1 and messy.longest_pause >= 3.0
    assert messy.start_delay >= 3.0
    assert messy.score <= 3
    assert 1 <= len(messy.tips) <= 3


def test_competency_profile_axes():
    attempts = [
        {"case_id": "CB4", "score": 2, "errors": ["CRITICAL_WEAKNESS"]},   # уверенность провалена
        {"case_id": "CB5", "score": 9, "errors": []},                      # коммуникация хорошо
    ]
    p = competencies.profile(attempts, KB, speech_scores=[6, 8])
    assert p["Уверенность"] < p["Коммуникация"]
    assert p["Структура ответа"] is None        # не проверялась
    assert p[competencies.SPEECH_AXIS] == 7.0
    best, worst = competencies.strongest_weakest(p)
    assert best == "Коммуникация" and worst == "Уверенность"


def test_radar_renders_png():
    with tempfile.TemporaryDirectory() as tmp:
        now = {name: v for (name, _), v in zip(competencies.AXES, [8, 6, None, 4, 9, 7])}
        before = {name: v for (name, _), v in zip(competencies.AXES, [5, 4, None, 3, 6, 6])}
        path = competencies.render_radar(now, before, Path(tmp) / "r.png")
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 20000


def test_mentor_speech_and_map_end_to_end():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            store = Storage(tmp / "f.db")
            await store.init()
            mentor = Mentor(KB, MentorLLM(None, "x", 5, KB), store, ROOT / "fonts", tmp)
            sid1 = await store.start_session(5, "train", "job")
            t1 = "ну короче я типа ничего не умею ну"
            fb1 = await mentor.evaluate(5, sid1, KB.cases["CB2"], t1, speech=analyze(t1, _words(t1)))
            assert "speech" in fb1.extra and "speech_prev" not in fb1.extra
            await store.log(sid1, 5, "bot", "feedback", "", {"case_id": "CB2", "feedback": fb1.to_dict()})
            t2 = "коммерческого опыта нет но я сделал бота и задеплоил его могу показать код"
            fb2 = await mentor.evaluate(5, sid1, KB.cases["CB1"], t2, speech=analyze(t2, _words(t2)))
            assert fb2.extra["speech_prev"]["fillers_per100"] > fb2.extra["speech"]["fillers_per100"]

            sid2 = await store.start_session(5, "train", "job")
            await mentor.evaluate(5, sid2, KB.cases["CB5"], KB.cases["CB5"].model_answer)
            path, now, before = await mentor.competency_map(5, sid2)
            assert path.exists() and before is not None
            pdf = await mentor.export(5, "pdf", "all")
            assert pdf.exists() and pdf.stat().st_size > 30000   # с картинкой карты
    asyncio.run(run())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
