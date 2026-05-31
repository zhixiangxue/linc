"""Example: a cross-platform echo agent.

Reads unread messages from every enabled LINC platform and replies the same text
back to the platform/conversation where each message arrived. Run alongside
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
        log.info("cross-platform echo agent started; waiting for unread messages...")
        while True:
            unread = await linc.read_unread_all()
            for m in unread:
                reply = f"{m.content.text or ''}"
                client = linc.get(m.platform)
                sender = m.sender.name or m.sender.id
                log.info("[%s/%s] %s (%s) -> %s", m.platform, m.conv_id, sender, m.sender.id, reply)
                await client.send(reply, conv_id=m.conv_id)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
