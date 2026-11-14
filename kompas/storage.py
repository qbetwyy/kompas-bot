"""Хранилище на SQLite (aiosqlite).

Храним минимум: Telegram user_id (без имени и username), историю сессий, ответы в тренажёре
и найденные ошибки это нужно для адаптивного обучения и экспорта. Команда /forget удаляет всё.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    scenario TEXT,
    started_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    meta TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    case_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    score INTEGER NOT NULL,
    feedback_json TEXT NOT NULL,
    is_followup INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempt_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    code TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speech_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id INTEGER NOT NULL,
    attempt_id INTEGER NOT NULL,
    wpm REAL,
    fillers_per100 REAL NOT NULL,
    long_pauses INTEGER NOT NULL,
    score INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_speech_user ON speech_reports(user_id, id);
CREATE INDEX IF NOT EXISTS idx_attempts_user ON attempts(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_errors_user ON attempt_errors(user_id, created_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path):
        self.path = str(path)

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def touch_user(self, user_id: int) -> None:
        ts = now_iso()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO users(user_id, created_at, last_seen) VALUES(?,?,?) "
                "ON CONFLICT(user_id) DO UPDATE SET last_seen=excluded.last_seen",
                (user_id, ts, ts),
            )
            await db.commit()

    async def start_session(self, user_id: int, mode: str, scenario: str | None) -> int:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "INSERT INTO sessions(user_id, mode, scenario, started_at) VALUES(?,?,?,?)",
                (user_id, mode, scenario, now_iso()),
            )
            await db.commit()
            return cur.lastrowid

    async def log(self, session_id: int, user_id: int, role: str, kind: str, text: str, meta: dict | None = None) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO messages(session_id, user_id, role, kind, text, meta, created_at) VALUES(?,?,?,?,?,?,?)",
                (session_id, user_id, role, kind, text, json.dumps(meta, ensure_ascii=False) if meta else None, now_iso()),
            )
            await db.commit()

    async def save_attempt(self, user_id: int, session_id: int, case_id: str, answer: str,
                           score: int, feedback: dict, errors: list[str], is_followup: bool) -> int:
        ts = now_iso()
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "INSERT INTO attempts(user_id, session_id, case_id, answer, score, feedback_json, is_followup, created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (user_id, session_id, case_id, answer, score, json.dumps(feedback, ensure_ascii=False), int(is_followup), ts),
            )
            attempt_id = cur.lastrowid
            await db.executemany(
                "INSERT INTO attempt_errors(attempt_id, user_id, code, created_at) VALUES(?,?,?,?)",
                [(attempt_id, user_id, code, ts) for code in errors],
            )
            await db.commit()
            return attempt_id

    async def save_speech(self, user_id: int, session_id: int, attempt_id: int, report: dict) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO speech_reports(user_id, session_id, attempt_id, wpm, fillers_per100, long_pauses, score, "
                "data_json, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (user_id, session_id, attempt_id, report.get("wpm"), report.get("fillers_per100", 0.0),
                 report.get("long_pauses", 0), report.get("score", 0), json.dumps(report, ensure_ascii=False), now_iso()),
            )
            await db.commit()

    async def speech_history(self, user_id: int, limit: int = 100) -> list[dict]:
        """Последние разборы подачи, от новых к старым."""
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(
                "SELECT session_id, attempt_id, score, data_json, created_at FROM speech_reports "
                "WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))).fetchall()
            return [{**dict(r), "data": json.loads(r["data_json"])} for r in rows]

    async def attempts_with_errors(self, user_id: int, limit: int = 200) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(
                "SELECT id, case_id, score, is_followup, created_at, session_id FROM attempts "
                "WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit))).fetchall()
            result = []
            for r in rows:
                errs = await (await db.execute("SELECT code FROM attempt_errors WHERE attempt_id=?", (r["id"],))).fetchall()
                result.append({**dict(r), "errors": [e[0] for e in errs]})
            return result

    async def last_session_id(self, user_id: int, mode: str | None = None) -> int | None:
        async with aiosqlite.connect(self.path) as db:
            if mode:
                row = await (await db.execute(
                    "SELECT s.id FROM sessions s WHERE s.user_id=? AND s.mode=? AND EXISTS "
                    "(SELECT 1 FROM messages m WHERE m.session_id=s.id) ORDER BY s.id DESC LIMIT 1", (user_id, mode))).fetchone()
            else:
                row = await (await db.execute(
                    "SELECT s.id FROM sessions s WHERE s.user_id=? AND EXISTS "
                    "(SELECT 1 FROM messages m WHERE m.session_id=s.id) ORDER BY s.id DESC LIMIT 1", (user_id,))).fetchone()
            return row[0] if row else None

    async def transcript(self, user_id: int, session_ids: list[int] | None = None) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            if session_ids:
                marks = ",".join("?" * len(session_ids))
                rows = await (await db.execute(
                    f"SELECT * FROM messages WHERE user_id=? AND session_id IN ({marks}) ORDER BY id",
                    (user_id, *session_ids))).fetchall()
            else:
                rows = await (await db.execute("SELECT * FROM messages WHERE user_id=? ORDER BY id", (user_id,))).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["meta"] = json.loads(d["meta"]) if d.get("meta") else {}
                out.append(d)
            return out

    async def session_ids_since(self, user_id: int, since_session: int) -> list[int]:
        async with aiosqlite.connect(self.path) as db:
            rows = await (await db.execute(
                "SELECT id FROM sessions WHERE user_id=? AND id>=? ORDER BY id", (user_id, since_session))).fetchall()
            return [r[0] for r in rows]

    async def forget(self, user_id: int) -> None:
        async with aiosqlite.connect(self.path) as db:
            for table in ("speech_reports", "attempt_errors", "attempts", "messages", "sessions", "users"):
                await db.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
            await db.commit()
