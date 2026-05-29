# linc

> An IM Gateway daemon that lets LLM Agents talk to multiple IM platforms via a SQLite file.

`linc` decouples your agent code from IM platform plumbing. The gateway process owns the IM connections (Slack Socket Mode today; Lark / Wecom / Telegram / Dingtalk on the roadmap) and persists every message into a single SQLite file. Your agent reads/writes that file through a tiny SDK — no webhooks, no queues, no extra services.

```
+-------------+        SQLite          +-----------------+
|  agent.py   |  <------------------>  |  linc gateway   |  <==> Slack / Lark / ...
| (your LLM)  |   .linc/linc.db        |  (linc serve)   |
+-------------+                        +-----------------+
```

- One file (`.linc/linc.db`) is the contract. Crash the agent, restart it — no message loss.
- The gateway and the agent are independent processes guarded by two `flock` files (`linc.pid` + `agent.lock`).
- All adapters share one `Hub` (HTTP client pool, future shared web server) so adding a new IM platform never duplicates infra.

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/<you>/linc.git
cd linc
uv sync
```

(Python ≥ 3.11. Uses [`uv`](https://docs.astral.sh/uv/) for env + dep management.)

### 2. Configure a Slack app

You need a Slack App with **Socket Mode** enabled (no public URL required):

- `bot_token` — Bot User OAuth Token (`xoxb-...`), scopes: `chat:write`, `channels:history`, `im:history`, `app_mentions:read`.
- `app_token` — App-Level Token (`xapp-...`) with the `connections:write` scope.

Copy the example config and drop the tokens in:

```bash
cp examples/linc.yaml linc.yaml
chmod 600 linc.yaml          # gateway warns otherwise
$EDITOR linc.yaml
```

### 3. Run the gateway

```bash
linc serve -c linc.yaml
```

This starts the daemon, opens `.linc/linc.db`, acquires `.linc/linc.pid` flock, and keeps a Slack Socket Mode WebSocket open. Leave it running.

### 4. Run an agent (in another terminal)

```bash
python examples/echo_agent.py
```

Send a DM to your bot in Slack — the agent echoes it back. That's it.

---

## Agent SDK in 6 lines

```python
import asyncio
from linc import Linc

async def main():
    async with Linc(".linc") as linc:
        slack = linc.slack()
        for m in await slack.read_unread():
            await slack.send(f"echo: {m.content.text}", conv_id=m.conv_id)

asyncio.run(main())
```

`linc.<platform>()` returns a stateless `Client` handle (think `boto3.client('s3')`). Unknown platform names fail loudly with `AttributeError` instead of silently no-op'ing.

---

## CLI cheat sheet

| Command | What it does |
|---|---|
| `linc serve [-c linc.yaml]` | Start the gateway daemon. SIGINT/SIGTERM stops cleanly. |
| `linc unread [-p slack] [--json]` | Peek unread messages without consuming them. |
| `linc history -p slack [-C C123] [-n 50]` | Show inbound + outbound history for a conversation. |
| `linc send slack C123 "hi"` | Enqueue an outbound message via the agent SDK. |
| `linc tail [-p slack]` | Stream new messages as they land. |
| `linc status` | Probe whether a gateway is running for the given `data_dir`. |

All commands accept `--data-dir`, defaulting to `.linc`.

---

## Examples

- [`examples/echo_agent.py`](examples/echo_agent.py) — minimal `unread → echo` loop.
- [`examples/llm_agent.py`](examples/llm_agent.py) — same shape, with a stub `call_llm()` to plug in your own client (OpenAI / Anthropic / local Ollama).
- [`examples/multi_platform.py`](examples/multi_platform.py) — one agent serving every registered platform via `linc.read_unread_all()`.

---

## Project layout

```
src/linc/
├── gateway.py          # LincGateway daemon: lifecycle + outbox dispatcher
├── client.py           # Linc / Client — agent-side SDK
├── cli.py              # `linc` typer entry point
├── adapters/
│   ├── __init__.py     # adapter registry (REGISTRY + register/get/...)
│   └── slack.py        # Slack Socket Mode adapter
└── core/
    ├── adapter.py      # Adapter ABC + ParsedInbound
    ├── config.py       # LincConfig + from_yaml
    ├── errors.py       # AlreadyRunning / SendError / ConfigError / UnknownPlatform
    ├── http.py         # HttpClient ABC + HttpxClient (5xx exponential backoff)
    ├── hub.py          # Shared infra (HttpClient pool, future web server)
    ├── locks.py        # fcntl flock helpers (linc.pid + agent.lock)
    ├── models.py       # Pydantic v2: Content / InboundMessage / OutboundMessage / ...
    ├── schema.py       # SQLite DDL
    └── store.py        # SqliteStore (single-conn + WAL + asyncio.Lock)
```

Design rationale for each file: see [`design/prd.md`](design/prd.md).

---

## Development

```bash
uv sync --all-extras
uv run pytest -q          # 74 tests
uv run ruff check .
```

Tests cover store / adapter registry / gateway lifecycle / client SDK / CLI / Slack adapter (mocked) / end-to-end echo loop.

---

## Status

**v0.1** (this release)
- ✅ SQLite-backed inbox + outbox with WAL, single-writer asyncio.Lock
- ✅ Gateway daemon with outbox dispatcher and partial-teardown rollback
- ✅ Agent SDK with two-flock coordination
- ✅ Slack Socket Mode adapter
- ✅ `linc serve / unread / history / send / tail / status` CLI

**v0.2** (planned)
- Lark / Wecom / Dingtalk webhook adapters (sharing one `Hub.webserver`)
- Telegram long-polling adapter
- `PRAGMA data_version` cross-process notification (lower latency than 100ms tick)
- Sender resolution cache (so Slack `sender.name` is the actual display name, not the user ID)

---

## License

[MIT](LICENSE)
