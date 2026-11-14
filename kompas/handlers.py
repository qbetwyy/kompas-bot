"""Обработчики Telegram (aiogram 3)."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, FSInputFile, Message
from aiogram.utils.chat_action import ChatActionSender

from . import keyboards as kbd
from . import render
from . import texts as T
from .kb import KnowledgeBase
from .mentor import Mentor
from .speech import analyze as analyze_speech
from .storage import Storage
from .voice import VoiceTranscriber

log = logging.getLogger(__name__)
router = Router()

MAX_FOLLOWUP_DEPTH = 2
MAX_ANSWER_LEN = 3000


class InfoSt(StatesGroup):
    asking = State()


class TrainSt(StatesGroup):
    answering = State()
    reviewing = State()


async def _send(message: Message, text: str, **kwargs) -> None:
    chunks = render.split_message(text)
    for i, chunk in enumerate(chunks):
        await message.answer(chunk, **(kwargs if i == len(chunks) - 1 else {}))


# ------------------------------------------------------------------- базовые
@router.message(CommandStart(deep_link=True, magic=F.args == "demo"))
@router.message(Command("demo"))
async def cmd_demo(message: Message, state: FSMContext, bot: Bot, mentor: Mentor, kb: KnowledgeBase, store: Storage):
    """Демо для экспертов: справка, разбор ответа, подача и карта навыков за одну минуту."""
    await state.clear()
    await store.touch_user(message.from_user.id)
    await message.answer(T.DEMO_INTRO, reply_markup=kbd.main_menu())
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        info = await mentor.info_answer(T.DEMO_QUESTION, "onboarding")
    if info.engine != "none":
        await _send(message, render.llm_text(info.text) + render.sources_block(info.sources))
    case = kb.cases["CB2"]
    await message.answer(render.case_message(case, kb, None, None))
    await message.answer(T.DEMO_TRAIN.format(answer=render.escape(T.DEMO_ANSWER)))
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        fb = await mentor.demo_feedback(case, T.DEMO_ANSWER)
    fb.follow_up = ""
    await _send(message, render.feedback_message(fb, kb))
    words, t = [], 1.2
    for i, w in enumerate(T.DEMO_ANSWER.split()):
        t += 1.6 if i == 5 else 0
        words.append((t, t + 0.25, w))
        t += 0.32
    demo_speech = analyze_speech(T.DEMO_ANSWER, words, t + 0.5).to_dict()
    await message.answer(T.DEMO_VOICE + "\n\n" + render.speech_block(demo_speech, {"fillers_per100": 21.4}))
    await message.answer_photo(FSInputFile(mentor.demo_map()), caption=T.DEMO_MAP)
    await message.answer(T.DEMO_END)


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, store: Storage):
    await state.clear()
    await store.touch_user(message.from_user.id)
    await message.answer(T.START, reply_markup=kbd.main_menu())


@router.message(Command("help"))
@router.message(F.text == T.BTN_ABOUT)
async def cmd_about(message: Message):
    await message.answer(T.ABOUT, reply_markup=kbd.main_menu())


@router.message(Command("forget"))
async def cmd_forget(message: Message, state: FSMContext, store: Storage):
    await state.clear()
    await store.forget(message.from_user.id)
    await message.answer(T.FORGET_DONE, reply_markup=kbd.main_menu())


@router.message(Command("info"))
@router.message(F.text == T.BTN_INFO)
async def menu_info(message: Message, state: FSMContext, kb: KnowledgeBase):
    await state.clear()
    await message.answer(T.CHOOSE_SCENARIO_INFO, reply_markup=kbd.scenarios(kb, "info"))


@router.message(Command("train"))
@router.message(F.text == T.BTN_TRAIN)
async def menu_train(message: Message, state: FSMContext, kb: KnowledgeBase):
    await state.clear()
    await message.answer(T.CHOOSE_SCENARIO_TRAIN, reply_markup=kbd.scenarios(kb, "train"))


async def _send_competency_map(message: Message, mentor: Mentor, user_id: int, session_id: int | None) -> None:
    try:
        result = await mentor.competency_map(user_id, session_id)
    except Exception:
        log.exception("competency map failed")
        return
    if not result:
        return
    path, now, before = result
    await message.answer_photo(FSInputFile(path), caption=render.competency_caption(now, before))


@router.message(Command("progress"))
@router.message(F.text == T.BTN_PROGRESS)
async def menu_progress(message: Message, mentor: Mentor, kb: KnowledgeBase):
    p = await mentor.progress(message.from_user.id)
    if not p:
        await message.answer(T.NO_DATA_PROGRESS)
        return
    await _send_competency_map(message, mentor, message.from_user.id, None)
    await message.answer(render.progress_message(p, kb))


@router.message(Command("export"))
@router.message(F.text == T.BTN_EXPORT)
async def menu_export(message: Message):
    await message.answer(T.EXPORT_ASK, reply_markup=kbd.export_formats())


# ------------------------------------------------------------ выбор сценария
@router.callback_query(F.data.startswith("scn:"))
async def on_scenario(cb: CallbackQuery, state: FSMContext, mentor: Mentor, kb: KnowledgeBase, store: Storage):
    _, mode, scenario = cb.data.split(":", 2)
    user_id = cb.from_user.id
    await store.touch_user(user_id)
    await cb.answer()
    if mode == "info":
        session_id = await store.start_session(user_id, "info", scenario)
        await state.set_state(InfoSt.asking)
        await state.set_data({"scenario": scenario, "session_id": session_id, "history": []})
        await cb.message.edit_text(T.ASK_QUESTION.format(example=T.INFO_EXAMPLES.get(scenario, T.INFO_EXAMPLES["all"])))
        return
    session_id = await store.start_session(user_id, "train", scenario)
    await cb.message.edit_text(T.TRAIN_START.format(title=T.SCENARIO_BUTTONS.get(scenario, kb.scenario_title(scenario)).lower()))
    await _give_case(cb.message, state, mentor, kb, store, user_id, scenario, session_id, number=1)


async def _give_case(message: Message, state: FSMContext, mentor: Mentor, kb: KnowledgeBase, store: Storage,
                     user_id: int, scenario: str, session_id: int, number: int, case_id: str | None = None):
    zone_title = None
    if case_id:
        case = kb.cases[case_id]
    else:
        case, zone, _ = await mentor.choose_case(user_id, scenario)
        zone_title = kb.errors[zone].title if zone else None
    await state.set_state(TrainSt.answering)
    await state.set_data({"scenario": scenario, "session_id": session_id, "case_id": case.id,
                          "case_no": number, "depth": 0})
    text = render.case_message(case, kb, zone_title, number)
    await store.log(session_id, user_id, "bot", "case", case.prompt, {"case_id": case.id})
    await message.answer(text)


# ------------------------------------------------------ действия после разбора
@router.callback_query(F.data.startswith("tr:"))
async def on_train_action(cb: CallbackQuery, state: FSMContext, mentor: Mentor, kb: KnowledgeBase, store: Storage):
    action = cb.data.split(":", 1)[1]
    data = await state.get_data()
    await cb.answer()
    if "session_id" not in data or "scenario" not in data:
        await cb.message.answer(T.TRAIN_OVER, reply_markup=kbd.scenarios(kb, "train"))
        return
    user_id = cb.from_user.id
    await cb.message.edit_reply_markup(reply_markup=None)
    if action == "followup":
        await cb.message.answer(T.FOLLOWUP_WAIT)
    elif action == "next":
        await _give_case(cb.message, state, mentor, kb, store, user_id, data["scenario"], data["session_id"],
                         number=data.get("case_no", 1) + 1)
    elif action == "retry":
        await _give_case(cb.message, state, mentor, kb, store, user_id, data["scenario"], data["session_id"],
                         number=data.get("case_no", 1), case_id=data.get("case_id"))
    elif action == "finish":
        summary = await mentor.session_summary(user_id, data["session_id"])
        await state.clear()
        await _send_competency_map(cb.message, mentor, user_id, data["session_id"])
        await cb.message.answer(render.summary_message(summary, kb), reply_markup=kbd.finish_actions())


# ------------------------------------------------------------------- экспорт
@router.callback_query(F.data.startswith("ex:"))
async def on_export(cb: CallbackQuery, mentor: Mentor):
    _, fmt, scope = cb.data.split(":", 2)
    await cb.answer(T.EXPORT_PREPARING)
    path = await mentor.export(cb.from_user.id, fmt, scope)
    if not path:
        await cb.message.answer(T.NO_DATA_EXPORT)
        return
    filename = f"kompas_razbor.{'md' if fmt == 'md' else 'pdf'}"
    await cb.message.answer_document(FSInputFile(path, filename=filename),
                                     caption=T.EXPORT_CAPTION)


# ------------------------------------------------------------------- голос
@router.message(F.voice)
async def on_voice(message: Message, state: FSMContext, bot: Bot, mentor: Mentor, kb: KnowledgeBase,
                   store: Storage, voice: VoiceTranscriber):
    if not voice.available:
        await message.answer(T.VOICE_OFF)
        return
    tr = None
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "voice.ogg"
                await bot.download(message.voice, destination=path)
                tr = await voice.transcribe(path)
    except Exception:
        log.exception("voice failed")
    if not tr or not tr.text:
        await message.answer(T.VOICE_FAIL)
        return
    await message.answer(T.VOICE_HEARD.format(text=render.escape(tr.text)))
    speech = None
    if await state.get_state() in (TrainSt.answering.state, TrainSt.reviewing.state):
        speech = analyze_speech(tr.text, tr.words, tr.duration or message.voice.duration)
    await _process_text(message, state, bot, mentor, kb, store, tr.text, speech=speech)


# -------------------------------------------------------------------- текст
@router.message(F.text)
async def on_text(message: Message, state: FSMContext, bot: Bot, mentor: Mentor, kb: KnowledgeBase, store: Storage):
    await _process_text(message, state, bot, mentor, kb, store, message.text)


async def _process_text(message: Message, state: FSMContext, bot: Bot, mentor: Mentor, kb: KnowledgeBase,
                        store: Storage, text: str, speech=None):
    text = (text or "").strip()[:MAX_ANSWER_LEN]
    if not text:
        return
    user_id = message.from_user.id
    await store.touch_user(user_id)
    current = await state.get_state()
    data = await state.get_data()

    if current == TrainSt.answering.state:
        await _handle_answer(message, state, bot, mentor, kb, store, text, data, previous=None, speech=speech)
        return
    if current == TrainSt.reviewing.state:
        previous = (data["last_answer"], data["follow_up"]) if data.get("follow_up") else None
        await _handle_answer(message, state, bot, mentor, kb, store, text, data, previous=previous, speech=speech)
        return
    if current != InfoSt.asking.state:
        # Свободный вопрос без выбранного режима, отвечаем как справка по всем темам.
        session_id = await store.start_session(user_id, "info", "all")
        data = {"scenario": "all", "session_id": session_id, "history": []}
        await state.set_state(InfoSt.asking)
        await state.set_data(data)
    await _handle_info(message, state, bot, mentor, store, text, data)


async def _handle_info(message: Message, state: FSMContext, bot: Bot, mentor: Mentor, store: Storage,
                       question: str, data: dict):
    user_id = message.from_user.id
    history = [tuple(x) for x in data.get("history", [])]
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        result = await mentor.info_answer(question, data.get("scenario"), history)
    await store.log(data["session_id"], user_id, "user", "info_q", question)
    if result.engine == "none":
        await message.answer(T.NOTHING_FOUND)
        await store.log(data["session_id"], user_id, "bot", "info_a", T.NOTHING_FOUND)
        return
    body = render.llm_text(result.text)
    if result.engine == "kb":
        body = T.LLM_DOWN_INFO + "\n\n" + body
    await _send(message, body + render.sources_block(result.sources))
    await store.log(data["session_id"], user_id, "bot", "info_a", result.text,
                    {"sources": [{"title": s.title, "url": s.url} for s in result.sources]})
    history.append((question, result.text))
    data["history"] = [list(x) for x in history[-3:]]
    await state.set_data(data)


async def _handle_answer(message: Message, state: FSMContext, bot: Bot, mentor: Mentor, kb: KnowledgeBase,
                         store: Storage, answer: str, data: dict, previous: tuple[str, str] | None, speech=None):
    user_id = message.from_user.id
    case = kb.cases[data["case_id"]]
    session_id = data["session_id"]
    await store.log(session_id, user_id, "user", "answer", answer,
                    {"case_id": case.id, "followup": previous is not None, "voice": speech is not None})
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        fb = await mentor.evaluate(user_id, session_id, case, answer, previous, speech=speech)
    depth = data.get("depth", 0) + (1 if previous else 0)
    if depth >= MAX_FOLLOWUP_DEPTH:
        fb.follow_up = ""
    await store.log(session_id, user_id, "bot", "feedback", "", {"case_id": case.id, "feedback": fb.to_dict()})
    text = render.feedback_message(fb, kb)
    if fb.engine == "rules":
        text = T.LLM_DOWN_TRAIN + "\n\n" + text
    await _send(message, text, reply_markup=kbd.after_feedback(bool(fb.follow_up)))
    data.update({"last_answer": answer, "follow_up": fb.follow_up, "depth": depth})
    await state.set_state(TrainSt.reviewing)
    await state.set_data(data)
