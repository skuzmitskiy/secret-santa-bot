# Secret Santa Telegram Bot

A minimal Telegram bot to run Secret Santa.

## Features

* Create and manage multiple events in parallel
* Shareable deep link to join: `https://t.me/<BOT_USERNAME>?start=join_<EVENTID>`
* Automatic full-name confirmation DM after joining (persists across restarts)
* Open/close signups; reopen (before drawing)
* Draw guarantees **one single cycle** over all participants
* Creator-only illegal edges (giver → receiver not allowed)
* Stores data in SQLite (`secretsanta.db`)
* Optional host-level lock to prevent multiple polling instances with the same token

## Requirements

* Python **3.9+**
* [`python-telegram-bot` 21.x](https://docs.python-telegram-bot.org/)

## Environment

Set the following variables:

```bash
export BOT_TOKEN="<your_botfather_token>"
export BOT_USERNAME="<your_bot_username_without_@>"
# Optional overrides:
# export SS_DB_PATH="secretsanta.db"
# export SS_LOCK_FILE="/var/run/ssbot.lock"
```

## Installation

```bash
pip install python-telegram-bot==21.6
```

## Run (Polling)

```bash
python run.py
```

## Quick Start (BotFather)

1. Chat with **@BotFather** → `/newbot` → follow prompts.
2. Save the provided **token** and **username**.
3. (Optional) Set commands via `/setcommands` (see below).

## Commands

```
/start                                  – Start or accept invites
/help                                   – Show help
/newevent <title>                       – Create a new event
/myevents                               – List my events
/share <event_id>                       – Get a join link
/list <event_id>                        – List participants
/leave <event_id>                       – Leave an event
/close <event_id>                       – Close signups (creator only)
/reopen <event_id>                      – Reopen signups (creator only)
/add_illegal <id> <giver> <receiver>    – Forbid G→R (creator only)
/view_illegal <event_id>                – Show forbidden pairs (creator only)
/clear_illegal <event_id>               – Remove all forbidden pairs (creator only)
/draw <event_id>                        – Run draw & DM participants (creator only)
/debug_cycle <event_id>                 – Show the cycle (creator only)
/deleteevent <event_id>                 – Delete the event (creator only)
```

## Typical Workflow

1. `/newevent Secret Santa Lab`
2. `/share 1` and share the deep link with your group
3. Participants click the link → bot confirms they joined
4. `/close 1` when you’re ready
5. (Optional) Add constraints with `/add_illegal 1 Alice Bob`
6. `/draw 1` to compute the single cycle and DM everyone

> Note: Telegram can block DMs to users who never pressed **Start** on the bot. Ask participants to start the bot before the draw.

## Constraints (Illegal Pairs)

* Use `/add_illegal <event_id> <giver> <receiver>` after closing signups.
* References may be numeric user IDs, `@username`, or a substring of the display name (as shown by `/list`).
* Too many constraints may make the cycle impossible; remove some or `/clear_illegal`.


