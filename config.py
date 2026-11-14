"""Настройки бота. Все секреты читаются из .env (файл не коммитится)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class Settings:
    bot_token: str
    anthropic_api_key: str | None
    anthropic_workspace_id: str | None
    model: str
    llm_timeout: float
    db_path: Path
    data_dir: Path
    fonts_dir: Path
    export_dir: Path
    telegram_proxy: str | None
    voice_enabled: bool
    whisper_model: str


def load_settings() -> Settings:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN не задан. Скопируйте .env.example в .env и впишите токен от @BotFather.")
    export_dir = BASE_DIR / "exports"
    export_dir.mkdir(exist_ok=True)
    return Settings(
        bot_token=token,
        anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip() or None,
        anthropic_workspace_id=(os.getenv("ANTHROPIC_WORKSPACE_ID") or "").strip() or None,
        model=os.getenv("LLM_MODEL", "claude-sonnet-5-5").strip(),
        llm_timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        db_path=Path(os.getenv("DB_PATH", str(BASE_DIR / "kompas.db"))),
        data_dir=BASE_DIR / "data",
        fonts_dir=BASE_DIR / "fonts",
        export_dir=export_dir,
        telegram_proxy=(os.getenv("TELEGRAM_PROXY") or "").strip() or None,
        voice_enabled=_bool(os.getenv("VOICE_ENABLED"), default=True),
        whisper_model=os.getenv("WHISPER_MODEL", "small").strip(),
    )
