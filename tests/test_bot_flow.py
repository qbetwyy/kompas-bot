"""Сквозной прогон Telegram-обработчиков без сети: подменяем HTTP-сессию бота заглушкой
и «скармливаем» диспетчеру апдейты, как будто пишет пользователь."""
import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aiogram import Bot, Dispatcher  # noqa: E402
from aiogram.client.default import DefaultBotProperties  # noqa: E402
from aiogram.client.session.base import BaseSession  # noqa: E402
from aiogram.fsm.storage.memory import MemoryStorage  # noqa: E402
from aiogram.types import CallbackQuery, Chat, Message, Update, User  # noqa: E402

from kompas import handlers  # noqa: E402
from kompas.kb import KnowledgeBase  # noqa: E402
from kompas.llm import MentorLLM  # noqa: E402
from kompas.mentor import Mentor  # noqa: E402
from kompas.storage import Storage  # noqa: E402
from kompas.voice import VoiceTranscriber  # noqa: E402

CHAT = Chat(id=777, type="private")
USER = User(id=777, is_bot=False, first_name="Тест")


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []
        self._mid = 100

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        name = type(method).__name__
        if name in {"SendMessage", "SendDocument", "SendPhoto", "EditMessageText", "EditMessageReplyMarkup"}:
            self._mid += 1
            return Message(message_id=self._mid, date=datetime.now(), chat=CHAT,
                           text=getattr(method, "text", None) or "file")
        return True

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield b""

    async def close(self):
        pass


def _msg(text, mid=[0]):
    mid[0] += 1
    return Update(update_id=mid[0], message=Message(message_id=mid[0], date=datetime.now(), chat=CHAT, from_user=USER, text=text))


def _cb(data, mid=[1000]):
    mid[0] += 1
    return Update(update_id=mid[0], callback_query=CallbackQuery(
        id=str(mid[0]), from_user=USER, chat_instance="ci", data=data,
        message=Message(message_id=5, date=datetime.now(), chat=CHAT, text="menu")))


def test_full_flow():
    async def run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            kb = KnowledgeBase(ROOT / "data")
            store = Storage(tmp / "b.db")
            await store.init()
            mentor = Mentor(kb, MentorLLM(None, "x", 5, kb), store, ROOT / "fonts", tmp)
            session = FakeSession()
            bot = Bot("123456:TEST", session=session, default=DefaultBotProperties(parse_mode="HTML"))
            dp = Dispatcher(storage=MemoryStorage())
            dp.include_router(handlers.router)
            dp["mentor"], dp["kb"], dp["store"], dp["voice"] = mentor, kb, store, VoiceTranscriber(False)

            def texts():
                return [getattr(c, "text", "") or "" for c in session.calls]

            await dp.feed_update(bot, _msg("/start"))
            assert "Компас" in texts()[-1]

            # демо-режим для экспертов не пишет в историю
            await dp.feed_update(bot, _msg("/demo"))
            assert any("Это верно" in t for t in texts()[-6:])
            assert type(session.calls[-2]).__name__ == "SendPhoto"
            assert await mentor.progress(777) is None

            # свободный вопрос без режима → справка
            await dp.feed_update(bot, _msg("можно ли испытательный срок полгода?"))
            assert "Откуда это" in texts()[-1]

            # тренажёр
            await dp.feed_update(bot, _msg("Тренироваться"))
            await dp.feed_update(bot, _cb("scn:train:job"))
            assert any("Отвечай так, как ответил бы вживую" in t for t in texts())
            await dp.feed_update(bot, _msg("Я ничего не умею, вопросов нет, предложите сами"))
            last = texts()[-1]
            assert "Это верно" in last and "Вот что можно улучшить" in last and "Вот удачный пример ответа" in last

            # ответ на уточняющий вопрос
            await dp.feed_update(bot, _msg("Ну ладно, я сделал бота на aiogram и задеплоил его на сервер, могу показать код"))
            assert "из 10" in texts()[-1]

            await dp.feed_update(bot, _cb("tr:next"))
            assert "Вопрос 2" in texts()[-1]
            await dp.feed_update(bot, _cb("tr:finish"))  # ещё не ответил на кейс 2 — всё равно итог
            assert "Итог" in texts()[-1]

            await dp.feed_update(bot, _msg("Мой прогресс"))
            assert "Над чем поработать" in texts()[-1]

            # новая тренировка должна начаться со слабой зоны
            await dp.feed_update(bot, _cb("scn:train:job"))
            assert any("В прошлый раз было слабое место" in t for t in texts()[-3:])

            await dp.feed_update(bot, _cb("ex:pdf:all"))
            assert type(session.calls[-1]).__name__ == "SendDocument"
            assert (tmp / "kompas_777_all.pdf").exists()

            await dp.feed_update(bot, _msg("/forget"))
            await dp.feed_update(bot, _msg("Мой прогресс"))
            assert "Пока нечего показать" in texts()[-1]
            await bot.session.close()
    asyncio.run(run())


if __name__ == "__main__":
    test_full_flow()
    print("ok")
