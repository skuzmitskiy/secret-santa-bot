#!/usr/bin/env python3
"""
Secret Santa Telegram Bot

Features
- Multiple concurrent events
- Deep-link join: shareable link lets people join an event with one tap
- Close signups and run the draw
- Ensures a SINGLE CYCLE permutation (no subloops)
- Creator-only: add "illegal" edges (disallowed giver→receiver pairs) before the draw
- DM each participant with their assigned recipient

Requirements
- Python 3.9+
- python-telegram-bot >= 21.0 (async API)

Environment
- BOT_TOKEN: Telegram bot token
- BOT_USERNAME: Your bot username WITHOUT the leading @ (for building invite links)

Run
- pip install python-telegram-bot==21.6
- python telegram_secret_santa_bot.py

Notes
- This example uses long polling for simplicity. For production, set a webhook.
- Persistence uses a lightweight SQLite DB (secretsanta.db) via the built-in sqlite3 module.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ForceReply,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# ------------------------------------------------------------
# Config & Logging
# ------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
BOT_USERNAME = os.environ.get("BOT_USERNAME", "your_bot_username")  # without @
DB_PATH = os.environ.get("SS_DB_PATH", "secretsanta.db")
LOCK_FILE = Path(os.environ.get("SS_LOCK_FILE", str(Path(DB_PATH).with_suffix(".lock"))))
_LOCK_HANDLE: Optional[IO[str]] = None


def acquire_single_instance_lock() -> bool:
    """Prevent running multiple polling loops on the same host."""
    global _LOCK_HANDLE
    if fcntl is None:
        logger.warning("fcntl module not available; skipping single-instance lock enforcement.")
        return False
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Ignore inability to create parent; open() will raise a clearer error.
        pass
    handle = open(LOCK_FILE, "w+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(
            f"Another Secret Santa bot instance appears to be running (lock file: {LOCK_FILE}). "
            "Stop the other process or remove the stale lock before starting a new instance."
        )
    handle.truncate(0)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _LOCK_HANDLE = handle
    logger.info("Acquired single-instance lock at %s", LOCK_FILE)
    return True


def release_single_instance_lock() -> None:
    global _LOCK_HANDLE
    if _LOCK_HANDLE is None:
        return
    try:
        if fcntl is not None:
            fcntl.flock(_LOCK_HANDLE.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        Path(_LOCK_HANDLE.name).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _LOCK_HANDLE.close()
    except Exception:
        pass
    _LOCK_HANDLE = None

# ------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------
SCHEMA_SQL = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    creator_id INTEGER NOT NULL,
    join_open INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_participants (
    event_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    PRIMARY KEY (event_id, user_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

-- Disallowed directed edges: giver -> receiver should not happen
CREATE TABLE IF NOT EXISTS disallowed_pairs (
    event_id INTEGER NOT NULL,
    giver_id INTEGER NOT NULL,
    receiver_id INTEGER NOT NULL,
    PRIMARY KEY (event_id, giver_id, receiver_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

-- Final assignments: single-cycle mapping giver -> receiver
CREATE TABLE IF NOT EXISTS assignments (
    event_id INTEGER NOT NULL,
    giver_id INTEGER NOT NULL,
    receiver_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (event_id, giver_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pending_name_requests (
    event_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    requested_at TEXT NOT NULL,
    PRIMARY KEY (event_id, user_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
);
"""

@contextmanager
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(SCHEMA_SQL)


# ------------------------------------------------------------
# Utility dataclasses
# ------------------------------------------------------------
@dataclass
class Event:
    id: int
    title: str
    creator_id: int
    join_open: bool
    created_at: str


@dataclass
class Participant:
    user_id: int
    display_name: str


# ------------------------------------------------------------
# DB operations
# ------------------------------------------------------------

def create_event(title: str, creator_id: int) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO events (title, creator_id, join_open, created_at) VALUES (?, ?, 1, ?)",
            (title, creator_id, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def get_event(event_id: int) -> Optional[Event]:
    with db() as conn:
        cur = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Event(
            id=row["id"],
            title=row["title"],
            creator_id=row["creator_id"],
            join_open=bool(row["join_open"]),
            created_at=row["created_at"],
        )


def list_my_events(user_id: int) -> List[Event]:
    with db() as conn:
        cur = conn.execute(
            "SELECT * FROM events WHERE creator_id = ? ORDER BY id DESC",
            (user_id,),
        )
        rows = cur.fetchall()
        return [
            Event(
                id=r["id"],
                title=r["title"],
                creator_id=r["creator_id"],
                join_open=bool(r["join_open"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]


def add_participant(event_id: int, user_id: int, display_name: str) -> bool:
    with db() as conn:
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO event_participants (event_id, user_id, display_name) VALUES (?, ?, ?)",
                (event_id, user_id, display_name),
            )
            return cur.rowcount > 0
        except sqlite3.IntegrityError:
            return False


def add_pending_name_request(event_id: int, user_id: int) -> None:
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO pending_name_requests (event_id, user_id, requested_at) VALUES (?, ?, ?)",
            (event_id, user_id, datetime.utcnow().isoformat()),
        )


def has_pending_name_request(event_id: int, user_id: int) -> bool:
    with db() as conn:
        cur = conn.execute(
            "SELECT 1 FROM pending_name_requests WHERE event_id = ? AND user_id = ?",
            (event_id, user_id),
        )
        return cur.fetchone() is not None


def list_pending_name_events(user_id: int) -> List[Tuple[int, str]]:
    with db() as conn:
        cur = conn.execute(
            """
            SELECT p.event_id, e.title
            FROM pending_name_requests AS p
            JOIN events AS e ON e.id = p.event_id
            WHERE p.user_id = ?
            ORDER BY p.requested_at ASC
            """,
            (user_id,),
        )
        return [(row["event_id"], row["title"]) for row in cur.fetchall()]


def clear_pending_name_requests(user_id: int, event_ids: Sequence[int]) -> None:
    if not event_ids:
        return
    with db() as conn:
        conn.executemany(
            "DELETE FROM pending_name_requests WHERE user_id = ? AND event_id = ?",
            [(user_id, eid) for eid in event_ids],
        )


def remove_participant(event_id: int, user_id: int) -> None:
    with db() as conn:
        conn.execute(
            "DELETE FROM event_participants WHERE event_id = ? AND user_id = ?",
            (event_id, user_id),
        )
        conn.execute(
            "DELETE FROM pending_name_requests WHERE event_id = ? AND user_id = ?",
            (event_id, user_id),
        )


def list_participants(event_id: int) -> List[Participant]:
    with db() as conn:
        cur = conn.execute(
            "SELECT user_id, display_name FROM event_participants WHERE event_id = ? ORDER BY display_name COLLATE NOCASE",
            (event_id,),
        )
        return [Participant(user_id=r["user_id"], display_name=r["display_name"]) for r in cur.fetchall()]


def list_user_memberships(user_id: int) -> List[Tuple[int, str]]:
    with db() as conn:
        cur = conn.execute(
            """
            SELECT ep.event_id, e.title
            FROM event_participants AS ep
            JOIN events AS e ON e.id = ep.event_id
            WHERE ep.user_id = ?
            ORDER BY ep.event_id DESC
            """,
            (user_id,),
        )
        return [(row["event_id"], row["title"]) for row in cur.fetchall()]


def update_participant_name(event_id: int, user_id: int, display_name: str) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE event_participants SET display_name = ? WHERE event_id = ? AND user_id = ?",
            (display_name, event_id, user_id),
        )


def set_join_open(event_id: int, open_flag: bool) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE events SET join_open = ? WHERE id = ?",
            (1 if open_flag else 0, event_id),
        )


def is_join_open(event_id: int) -> bool:
    ev = get_event(event_id)
    return bool(ev and ev.join_open)


def add_disallowed(event_id: int, giver_id: int, receiver_id: int) -> None:
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO disallowed_pairs (event_id, giver_id, receiver_id) VALUES (?, ?, ?)",
            (event_id, giver_id, receiver_id),
        )


def list_disallowed(event_id: int) -> List[Tuple[int, int]]:
    with db() as conn:
        cur = conn.execute(
            "SELECT giver_id, receiver_id FROM disallowed_pairs WHERE event_id = ?",
            (event_id,),
        )
        return [(r["giver_id"], r["receiver_id"]) for r in cur.fetchall()]


def clear_assignments(event_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM assignments WHERE event_id = ?", (event_id,))


def save_assignments(event_id: int, pairs: List[Tuple[int, int]]) -> None:
    now = datetime.utcnow().isoformat()
    with db() as conn:
        conn.executemany(
            "INSERT INTO assignments (event_id, giver_id, receiver_id, created_at) VALUES (?, ?, ?, ?)",
            [(event_id, g, r, now) for g, r in pairs],
        )


def get_assignments(event_id: int) -> List[Tuple[int, int]]:
    with db() as conn:
        cur = conn.execute(
            "SELECT giver_id, receiver_id FROM assignments WHERE event_id = ?",
            (event_id,),
        )
        return [(r["giver_id"], r["receiver_id"]) for r in cur.fetchall()]



def delete_event(event_id: int) -> None:
    with db() as conn:
        conn.execute("DELETE FROM events WHERE id = ?", (event_id,))

# ------------------------------------------------------------
# Secret Santa draw: SINGLE CYCLE with constraints
# ------------------------------------------------------------
class DrawError(Exception):
    pass


def _build_allowed_graph(participants: Sequence[int], disallowed: set[Tuple[int, int]]) -> Dict[int, List[int]]:
    allowed: Dict[int, List[int]] = {}
    pset = set(participants)
    for g in participants:
        # allowed receivers are everyone else except disallowed edges
        allowed_receivers = [r for r in participants if r != g and (g, r) not in disallowed]
        random.shuffle(allowed_receivers)
        allowed[g] = allowed_receivers
    return allowed


def _try_random_cycle(participants: List[int], disallowed: set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    # Simple fast path: shuffle into a cycle and check constraints
    order = participants[:]
    random.shuffle(order)
    pairs = []
    ok = True
    for i in range(len(order)):
        g = order[i]
        r = order[(i + 1) % len(order)]
        if (g, r) in disallowed or g == r:
            ok = False
            break
        pairs.append((g, r))
    return pairs if ok else None


def _backtracking_single_cycle(participants: List[int], disallowed: set[Tuple[int, int]], timeout_s: float = 3.0) -> Optional[List[Tuple[int, int]]]:
    """Construct a Hamiltonian cycle in the directed graph of allowed edges.
    Uses DFS with heuristics (min-remaining-values, randomized order) and a timeout.
    Returns list of (giver, receiver) pairs in cycle order if found.
    """
    import time

    start_t = time.time()
    n = len(participants)
    if n < 2:
        return None

    allowed = _build_allowed_graph(participants, disallowed)

    # Start from a random node to add variety
    start = random.choice(participants)
    path: List[int] = [start]
    used = {start}

    # Precompute sorted candidate order (smallest out-degree first to reduce branching)
    def candidates(node: int) -> List[int]:
        cands = [r for r in allowed[node] if r not in used]
        # Heuristic: sort by remaining out-degree of candidate (MRV), random tiebreak
        cands.sort(key=lambda x: (len([y for y in allowed[x] if y not in used and y != x]), random.random()))
        return cands

    def dfs(node: int) -> bool:
        nonlocal path
        if time.time() - start_t > timeout_s:
            return False
        if len(path) == n:
            # need closing edge path[-1] -> start
            last = path[-1]
            if (last, start) in disallowed or last == start:
                return False
            return True
        for nxt in candidates(node):
            path.append(nxt)
            used.add(nxt)
            if dfs(nxt):
                return True
            used.remove(nxt)
            path.pop()
        return False

    if dfs(start):
        pairs = [(path[i], path[(i + 1) % n]) for i in range(n)]
        return pairs
    return None


def compute_single_cycle_with_constraints(event_id: int) -> List[Tuple[int, int]]:
    participants = [p.user_id for p in list_participants(event_id)]
    if len(participants) < 2:
        raise DrawError("At least 2 participants are required for the draw.")

    disallowed = set(list_disallowed(event_id))

    # Try several quick random attempts first
    for _ in range(200):
        res = _try_random_cycle(participants, disallowed)
        if res is not None:
            return res

    # Fall back to backtracking with a time limit
    res = _backtracking_single_cycle(participants, disallowed, timeout_s=6.0)
    if res is None:
        raise DrawError("Unable to find a single cycle that satisfies the constraints. Try removing some constraints.")
    return res


# ------------------------------------------------------------
# Bot command handlers
# ------------------------------------------------------------
HELP_TEXT = (
    """<b>Тайный Санта</b> 🎁

Создавайте события, позволяйте участникам присоединяться по ссылке, закрывайте регистрацию и проводите розыгрыш, который генерирует единый цикл (без подциклов).

<b>Основные команды Secret Santa Bot</b>

• /newevent <i>title</i> – создать новое событие

• /myevents – список ваших событий

• /share <i>event_id</i> – ссылка для присоединения

• /list <i>event_id</i> – участники

• /leave <i>event_id</i> – покинуть событие

• /close <i>event_id</i> – закрыть регистрацию (только создатель)

• /reopen <i>event_id</i> – открыть регистрацию снова (только создатель)

• /add_illegal <i>event_id</i> <i>giver</i> <i>receiver</i> – запретить направленную пару Д→О (только создатель)

   (пользователь может быть указан как @username или по имени, как в списке)

• /view_illegal <i>event_id</i> – показать запрещённые пары

• /clear_illegal <i>event_id</i> – удалить запрещённые пары (только создатель)

• /draw <i>event_id</i> – провести жеребьёвку и отправить личные сообщения

• /debug_cycle <i>event_id</i> – показать цепочку (только создатель)

• /deleteevent <i>event_id</i> – удалить событие (только создатель)

<b>Присоединение по ссылке</b>
Используйте /share, чтобы получить ссылку вида https://t.me/%s?start=join_EVENTID. По клику пользователь зарегистрируется, если регистрация открыта.
"""
) % BOT_USERNAME


async def start_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    args = context.args

    # Deep-link payload: /start join_<event_id>
    if args and len(args) == 1 and args[0].startswith("join_"):
        try:
            event_id = int(args[0].split("_", 1)[1])
        except ValueError:
            await update.message.reply_text("Invalid link.")
            return
        ev = get_event(event_id)
        if not ev:
            await update.message.reply_text("Event not found.")
            return
        if not ev.join_open:
            await update.message.reply_text("Signups for this event are closed.")
            return
        display_name = (user.full_name or user.username or str(user.id)).strip()
        is_new_participant = add_participant(event_id, user.id, display_name)
        await update.message.reply_text(
            f"Здравствуйте!\n"
            f"\n"
            f"Вы успешно присоединились к игре «<b>{ev.title}</b>».\n"
            f"Регистрация участников открыта до 19 декабря 2025 года.\n"
            f"\n"
            f"Важный шаг: Чтобы ваш Тайный Санта знал, кому готовить подарок, пожалуйста, подтвердите своё участие, указав свои Имя и Фамилию (полностью).\n"
            f"\n"
            f"Жеребьевка состоится после окончания регистрации, а вручение подарков запланировано на 30 декабря 2025 года.\n"
            f"\n"
            f"Спасибо, что участвуете!",
            parse_mode=ParseMode.HTML,
        )
        needs_name = False
        if is_new_participant:
            add_pending_name_request(event_id, user.id)
            needs_name = True
        elif has_pending_name_request(event_id, user.id):
            needs_name = True

        if needs_name and update.message:
            pending = list(dict.fromkeys(context.user_data.get("pending_name_events", [])))
            if event_id not in pending:
                pending.append(event_id)
            context.user_data["pending_name_events"] = pending
            await update.message.reply_text(
                "Пожалуйста, ответьте на это сообщение и напишите свои Имя и Фамилию через пробел "
                "для участия в событии «{title}» (ID {eid}). Например: Иван Иванов.".format(
                    title=ev.title, eid=ev.id
                ),
                parse_mode=ParseMode.HTML,
                reply_markup=ForceReply(input_field_placeholder="Имя Фамилия"),
            )
        return

    # Default /start
    await update.message.reply_text(
        "Hi! I'm a Secret Santa bot. Use /help for commands."
    )


async def collect_name_response(update: Update, context: CallbackContext) -> None:
    message = update.message
    user = update.effective_user
    if not message or not message.text or not user:
        return
    pending_events = context.user_data.get("pending_name_events")
    pending_pairs: List[Tuple[int, Optional[str]]] = []
    if pending_events:
        unique_ids = list(dict.fromkeys(pending_events))
        for event_id in unique_ids:
            ev = get_event(event_id)
            if ev:
                pending_pairs.append((event_id, ev.title))
    else:
        db_pending = list_pending_name_events(user.id)
        if db_pending:
            pending_pairs = [(eid, title) for eid, title in db_pending]
            unique_ids = [eid for eid, _ in pending_pairs]
            context.user_data["pending_name_events"] = unique_ids[:]

    if not pending_pairs:
        memberships = list_user_memberships(user.id)
        if not memberships:
            return
        pending_pairs = memberships

    if update.effective_chat and update.effective_chat.type != "private":
        return

    text = message.text.strip()
    parts = [token for token in text.split() if token]
    if len(parts) < 2:
        await message.reply_text("Нужно указать как минимум Имя и Фамилию через пробел. Например: Иван Иванов.")
        return

    first_name = parts[0]
    last_name = " ".join(parts[1:]).strip()
    display_name = f"{first_name} {last_name}".strip()

    updated_events: List[str] = []
    updated_event_ids: List[int] = []
    for event_id, title in pending_pairs:
        update_participant_name(event_id, user.id, display_name)
        if title:
            updated_events.append(title)
        else:
            ev = get_event(event_id)
            if ev:
                updated_events.append(ev.title)
        updated_event_ids.append(event_id)

    remaining = [eid for eid in context.user_data.get("pending_name_events", []) if eid not in updated_event_ids]
    context.user_data["pending_name_events"] = remaining
    clear_pending_name_requests(user.id, updated_event_ids)

    if not updated_events:
        await message.reply_text(f"Спасибо! Имя сохранено: {display_name}.")
        return

    confirmation = (
        f"Спасибо! Зафиксировал ваше имя: {display_name}.\n"
        f"События: {', '.join(updated_events)}"
    )
    await message.reply_text(confirmation)


async def help_cmd(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def newevent_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    title = " ".join(context.args).strip()
    if not title:
        await update.message.reply_text("Usage: /newevent <title>")
        return
    eid = create_event(title, user.id)
    await update.message.reply_text(
        f"Created event <b>{title}</b> with ID <code>{eid}</code>.\nUse /share {eid} to get the join link.",
        parse_mode=ParseMode.HTML,
    )


async def myevents_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    events = list_my_events(user.id)
    if not events:
        await update.message.reply_text("You have no events. Create one with /newevent <title>.")
        return
    lines = ["Your events:"]
    for e in events:
        status = "OPEN" if e.join_open else "CLOSED"
        lines.append(f"• ID {e.id} – {e.title} [{status}]")
    await update.message.reply_text("\n".join(lines))


async def share_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /share <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can share the event link.")
        return
    link = f"https://t.me/{BOT_USERNAME}?start=join_{eid}"
    kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton(text="Join the event", url=link)]]
    )
    await update.message.reply_text(
        "Share this link:\n"
    )
    await update.message.reply_text(
        f"{link}", reply_markup=kb
    )


async def list_cmd(update: Update, context: CallbackContext) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /list <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    plist = list_participants(eid)
    if not plist:
        await update.message.reply_text("No participants yet.")
        return
    lines = [f"Participants for <b>{ev.title}</b> (ID {ev.id}):"]
    for p in plist:
        lines.append(f"• {p.display_name} (ID {p.user_id})")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def leave_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /leave <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if not ev.join_open:
        await update.message.reply_text("Signups are closed: you cannot leave now.")
        return
    remove_participant(eid, user.id)
    await update.message.reply_text("You have left the event.")


async def close_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /close <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can close signups.")
        return
    set_join_open(eid, False)
    await update.message.reply_text(
        f"Signups closed for <b>{ev.title}</b>. You can now add constraints with /add_illegal and then run /draw.",
        parse_mode=ParseMode.HTML,
    )


async def reopen_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /reopen <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can reopen signups.")
        return
    if get_assignments(eid):
        await update.message.reply_text("Assignments have already been made: you cannot reopen.")
        return
    set_join_open(eid, True)
    await update.message.reply_text(f"Signups reopened for <b>{ev.title}</b>.", parse_mode=ParseMode.HTML)


# --- Helpers to parse user references (by @username or display name snippet or numeric ID)

def resolve_user_ref(event_id: int, token: str) -> Optional[Participant]:
    token = token.strip()
    plist = list_participants(event_id)
    # Try numeric ID
    try:
        uid = int(token)
        for p in plist:
            if p.user_id == uid:
                return p
    except ValueError:
        pass
    # Try @username match inside display_name
    if token.startswith("@"):
        token = token[1:]
    token_low = token.lower()
    # best-effort: substring case-insensitive
    matches = [p for p in plist if token_low in p.display_name.lower()]
    if len(matches) == 1:
        return matches[0]
    # If ambiguous, prefer exact (case-insensitive)
    exact = [p for p in plist if p.display_name.lower() == token_low]
    if len(exact) == 1:
        return exact[0]
    return None


async def debug_cycle_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /debug_cycle <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can use this command.")
        return
    pairs = get_assignments(eid)
    if not pairs:
        await update.message.reply_text("There are no saved assignments for this event.")
        return
    # Reconstruct the chain order
    nxt = {g: r for g, r in pairs}
    start = pairs[0][0]
    order = [start]
    while True:
        nxt_g = nxt.get(order[-1])
        if nxt_g is None or nxt_g == start:
            break
        order.append(nxt_g)
    plist = {p.user_id: p.display_name for p in list_participants(eid)}
    chain = " → ".join(plist.get(uid, str(uid)) for uid in order) + f" → {plist.get(start, str(start))}"
    await update.message.reply_text(
        f"<b>Debug chain</b> for <i>{ev.title}</i> (ID {eid}):\n{chain}",
        parse_mode=ParseMode.HTML,
    )


async def deleteevent_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /deleteevent <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found (or already deleted).")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can delete the event.")
        return
    delete_event(eid)
    await update.message.reply_text("Event deleted permanently.")


async def add_illegal_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    args = context.args
    if len(args) < 3:
        await update.message.reply_text("Usage: /add_illegal <event_id> <giver> <receiver>")
        return
    try:
        eid = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can add constraints.")
        return
    if ev.join_open:
        await update.message.reply_text("Close signups first with /close.")
        return
    giver_ref, recv_ref = args[1], args[2]
    giver = resolve_user_ref(eid, giver_ref)
    receiver = resolve_user_ref(eid, recv_ref)
    if not giver or not receiver:
        await update.message.reply_text("Unable to resolve references. Use numeric ID or part of the name as shown in /list.")
        return
    if giver.user_id == receiver.user_id:
        await update.message.reply_text("A participant cannot give to themselves.")
        return
    add_disallowed(eid, giver.user_id, receiver.user_id)
    await update.message.reply_text(
        f"Added constraint: <b>{giver.display_name}</b> <i>cannot give to</i> <b>{receiver.display_name}</b>.",
        parse_mode=ParseMode.HTML,
    )


async def view_illegal_cmd(update: Update, context: CallbackContext) -> None:
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Usage: /view_illegal <event_id>")
        return
    try:
        eid = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    plist = {p.user_id: p for p in list_participants(eid)}
    rules = list_disallowed(eid)
    if not rules:
        await update.message.reply_text("No constraints defined.")
        return
    lines = [f"Constraints for <b>{ev.title}</b>:"]
    for g, r in rules:
        gname = plist.get(g).display_name if g in plist else str(g)
        rname = plist.get(r).display_name if r in plist else str(r)
        lines.append(f"• {gname} → {rname} (forbidden)")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def clear_illegal_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    args = context.args
    if len(args) < 1:
        await update.message.reply_text("Usage: /clear_illegal <event_id>")
        return
    try:
        eid = int(args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can clear constraints.")
        return
    with db() as conn:
        conn.execute("DELETE FROM disallowed_pairs WHERE event_id = ?", (eid,))
    await update.message.reply_text("Constraints removed.")


async def draw_cmd(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    if not context.args:
        await update.message.reply_text("Usage: /draw <event_id>")
        return
    try:
        eid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Invalid event ID.")
        return
    ev = get_event(eid)
    if not ev:
        await update.message.reply_text("Event not found.")
        return
    if ev.creator_id != user.id:
        await update.message.reply_text("Only the creator can run the draw.")
        return
    if ev.join_open:
        await update.message.reply_text("Close signups first with /close.")
        return

    # Compute assignments
    try:
        pairs = compute_single_cycle_with_constraints(eid)
    except DrawError as e:
        await update.message.reply_text(str(e))
        return

    # Save assignments and DM participants
    clear_assignments(eid)
    save_assignments(eid, pairs)

    participants_map = {p.user_id: p.display_name for p in list_participants(eid)}

    app: Application = context.application
    sent = 0
    for giver, receiver in pairs:
        giver_name = str(participants_map.get(giver) or giver)
        receiver_name = str(participants_map.get(receiver) or receiver)
    
        try:
            await app.bot.send_message(
                chat_id=giver,
                text=(
                    f"Здравствуй, {giver_name}! 🎁\n"
                    f"\n"
                    f"Жеребьёвка игры <b>{ev.title}</b> завершена.\n"
                    f"Твой подопечный, для которого ты готовишь подарок - <b>{receiver_name}</b>.\n"
                    f"\n"
                    f"Твой бюджет на создание новогоднего чуда — от 500 рублей!\n"
                    f"\n"
                    f"Покажи, на что способен! 😉"
                ),
                parse_mode=ParseMode.HTML,
            )
            sent += 1
        except Exception as e:
            logger.warning("DM failed to %s: %s", giver, e)

    await update.message.reply_text(
        f"Draw completed ✅. Assignments saved. Messages sent: {sent}/{len(pairs)}. "
        f"To view the chain for debugging: /debug_cycle {eid} (creator only)."
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def build_app() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("Set the BOT_TOKEN environment variable.")

    init_db()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("newevent", newevent_cmd))
    app.add_handler(CommandHandler("myevents", myevents_cmd))
    app.add_handler(CommandHandler("share", share_cmd))
    app.add_handler(CommandHandler("list", list_cmd))
    app.add_handler(CommandHandler("leave", leave_cmd))
    app.add_handler(CommandHandler("close", close_cmd))
    app.add_handler(CommandHandler("reopen", reopen_cmd))
    app.add_handler(CommandHandler("add_illegal", add_illegal_cmd))
    app.add_handler(CommandHandler("view_illegal", view_illegal_cmd))
    app.add_handler(CommandHandler("clear_illegal", clear_illegal_cmd))
    app.add_handler(CommandHandler("draw", draw_cmd))
    app.add_handler(CommandHandler("debug_cycle", debug_cycle_cmd))
    app.add_handler(CommandHandler("deleteevent", deleteevent_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, collect_name_response))

    return app


async def amain() -> None:
    lock_acquired = False
    try:
        lock_acquired = acquire_single_instance_lock()
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    app = build_app()
    logger.info("Starting Secret Santa bot with polling…")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    try:
        await asyncio.Event().wait()
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        if lock_acquired:
            release_single_instance_lock()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except (KeyboardInterrupt, SystemExit):
        print("Bye!")
