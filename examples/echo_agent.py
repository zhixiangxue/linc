"""Example: an echo agent.

Reads unread Slack messages and replies "echo: <text>" to each. Run alongside
``linc serve`` (in a separate terminal) — the two processes coordinate via
``.linc/linc.pid`` (gateway) and ``.linc/agent.lock`` (this script).

    # terminal 1
    linc serve -c linc.yaml

    # terminal 2
    python examples/echo_agent.py

The agent polls every 1s. For latency-sensitive setups, tighten the sleep or
plug in PRAGMA data_version notification (see PRD §13).
"""

from __future__ import annotations

import asyncio
import logging

from linc import Linc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("echo_agent")


async def main() -> None:
    async with Linc(".linc") as linc:
        slack = linc.slack()
        log.info("echo agent started; waiting for unread messages...")
        while True:
            unread = await slack.read_unread()
            for m in unread:
                reply = f"{m.content.text or ''}"
                log.info("[%s] %s -> %s", m.conv_id, m.content.text, reply)
                await slack.send(reply, conv_id=m.conv_id)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
