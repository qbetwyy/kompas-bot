"""Проверка ключей перед запуском: python tools/check_keys.py

Проверяет токен Telegram (getMe) и ключ Anthropic (короткий тестовый запрос к модели).
Сами ключи не печатает.
"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")


async def check_telegram(token: str) -> None:
    from aiogram import Bot

    if not token:
        print("❌ BOT_TOKEN пустой, впиши токен от @BotFather в .env")
        return
    bot = Bot(token)
    try:
        me = await bot.get_me()
        print(f"✅ Telegram: токен работает, это бот @{me.username} ({me.full_name})")
        if "fizya" in (me.username or "").lower():
            print("⚠️  Это токен Физи! Создай в @BotFather нового бота для «Компаса».")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Telegram: токен не подошёл, {type(exc).__name__}: {exc}")
    finally:
        await bot.session.close()


async def check_anthropic(key: str, model: str, workspace: str | None) -> None:
    from anthropic import AsyncAnthropic

    if not key:
        print("❌ ANTHROPIC_API_KEY пустой")
        return
    if not key.startswith("sk-ant-"):
        print("❌ Ключ Anthropic должен начинаться с sk-ant- (похоже, скопирован ID ключа, а не сам ключ)")
        return
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    client = AsyncAnthropic(api_key=key, default_headers=headers, timeout=30, max_retries=1)
    try:
        resp = await client.messages.create(
            model=model, max_tokens=300, messages=[{"role": "user", "content": "Ответь одним словом: работает?"}]
        )
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        kinds = ", ".join(getattr(b, "type", "?") for b in resp.content)
        print(f"✅ Anthropic: ключ работает, модель {model} ответила: «{text}» (блоки: {kinds}; stop: {resp.stop_reason})")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"❌ Anthropic: {type(exc).__name__}: {msg[:300]}")
        if "workspace" in msg.lower():
            print("   Подсказка: ключ привязан к аккаунту без рабочего пространства. Либо создай ключ с выбранным "
                  "Workspace = Default, либо впиши ANTHROPIC_WORKSPACE_ID=<id из Settings → Workspaces> в .env")
        elif "credit" in msg.lower() or "balance" in msg.lower():
            print("   Подсказка: пополни баланс API в Claude Console")
        elif "model" in msg.lower():
            print("   Подсказка: попробуй LLM_MODEL=claude-haiku-5-5 в .env")


async def main() -> None:
    await check_telegram(os.getenv("BOT_TOKEN", "").strip())
    await check_anthropic(
        os.getenv("ANTHROPIC_API_KEY", "").strip(),
        os.getenv("LLM_MODEL", "claude-sonnet-5-5").strip(),
        (os.getenv("ANTHROPIC_WORKSPACE_ID") or "").strip() or None,
    )


if __name__ == "__main__":
    asyncio.run(main())
