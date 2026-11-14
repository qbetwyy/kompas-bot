"""Точка входа: python bot.py"""
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand

from config import load_settings
from kompas.handlers import router
from kompas.kb import KnowledgeBase
from kompas.llm import MentorLLM
from kompas.mentor import Mentor
from kompas.storage import Storage
from kompas.voice import VoiceTranscriber


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()

    kb = KnowledgeBase(settings.data_dir)
    store = Storage(settings.db_path)
    await store.init()
    llm = MentorLLM(settings.anthropic_api_key, settings.model, settings.llm_timeout, kb, settings.anthropic_workspace_id)
    voice = VoiceTranscriber(settings.voice_enabled, settings.whisper_model)
    mentor = Mentor(kb, llm, store, settings.fonts_dir, settings.export_dir)

    logging.info("База знаний: %d записей, %d кейсов, %d типов ошибок", len(kb.entries), len(kb.cases), len(kb.errors))
    logging.info("LLM: %s | Голос: %s", settings.model if llm.available else "выключена (нет ключа)",
                 "вкл" if voice.available else "выкл")

    session = AiohttpSession(proxy=settings.telegram_proxy) if settings.telegram_proxy else None
    bot = Bot(settings.bot_token, session=session,
              default=DefaultBotProperties(parse_mode=ParseMode.HTML, link_preview_is_disabled=True))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    dp["mentor"] = mentor
    dp["kb"] = kb
    dp["store"] = store
    dp["voice"] = voice

    try:
        await _set_commands(bot)
    except Exception as exc:  # нет связи с Telegram на старте, не падаем, polling сам переподключится
        logging.warning("Не удалось установить меню команд (%s). Проверь интернет/VPN, бот продолжит попытки.", exc)
    await dp.start_polling(bot)


async def _set_commands(bot: Bot) -> None:
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать"),
        BotCommand(command="demo", description="Показать, что умеет бот, за минуту"),
        BotCommand(command="info", description="Спросить"),
        BotCommand(command="train", description="Тренироваться"),
        BotCommand(command="progress", description="Мой прогресс"),
        BotCommand(command="export", description="Скачать разбор"),
        BotCommand(command="help", description="Как это работает"),
        BotCommand(command="forget", description="Удалить мои данные"),
    ])


if __name__ == "__main__":
    asyncio.run(main())
