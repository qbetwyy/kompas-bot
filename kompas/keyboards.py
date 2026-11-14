"""Клавиатуры."""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

from .kb import KnowledgeBase
from .texts import ANY_TOPIC, BTN_ABOUT, BTN_EXPORT, BTN_INFO, BTN_PROGRESS, BTN_TRAIN, SCENARIO_BUTTONS


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_INFO), KeyboardButton(text=BTN_TRAIN)],
            [KeyboardButton(text=BTN_PROGRESS), KeyboardButton(text=BTN_EXPORT)],
            [KeyboardButton(text=BTN_ABOUT)],
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Пиши или говори",
    )


def scenarios(kb: KnowledgeBase, mode: str) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=SCENARIO_BUTTONS.get(k, v), callback_data=f"scn:{mode}:{k}")]
            for k, v in kb.scenarios.items()]
    if mode == "info":
        rows.append([InlineKeyboardButton(text=ANY_TOPIC, callback_data="scn:info:all")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def after_feedback(has_follow_up: bool) -> InlineKeyboardMarkup:
    rows = []
    if has_follow_up:
        rows.append([InlineKeyboardButton(text="Ответить на вопрос", callback_data="tr:followup")])
    rows.append([
        InlineKeyboardButton(text="Следующий вопрос", callback_data="tr:next"),
        InlineKeyboardButton(text="Ещё раз", callback_data="tr:retry"),
    ])
    rows.append([InlineKeyboardButton(text="Хватит, подведи итог", callback_data="tr:finish")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def export_formats() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="PDF, последняя тренировка", callback_data="ex:pdf:last")],
        [InlineKeyboardButton(text="Заметкой (.md)", callback_data="ex:md:last")],
        [InlineKeyboardButton(text="PDF, вся история", callback_data="ex:pdf:all")],
    ])


def finish_actions() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Скачать PDF", callback_data="ex:pdf:last"),
         InlineKeyboardButton(text="Заметкой", callback_data="ex:md:last")],
    ])
