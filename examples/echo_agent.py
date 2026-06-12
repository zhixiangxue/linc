"""Example: a cross-platform echo agent.

Reads unread messages from every enabled LINC platform and replies the same
text back to the platform/conversation where each message arrived.

Single-command startup — ``launch()`` handles gateway + client in one line::

    python examples/echo_agent.py

The agent polls every 1s. For latency-sensitive setups, tighten the sleep.
"""

from __future__ import annotations

import asyncio
import logging

from linc import launch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("echo_agent")


async def main() -> None:
    client = await launch("linc.yaml")
    try:
        log.info("cross-platform echo agent started; waiting for unread messages...")
        while True:
            unread = await client.pull()
            for m in unread:
                reply = f"{m.content.text or ''}"
                messenger = client.messenger(m.platform)
                sender = m.sender.name or m.sender.id
                log.info(
                    "[%s/%s] %s -> %s",
                    m.platform, m.conv_id, sender, reply,
                )
                await messenger.send(reply, conv_id=m.conv_id)
            await asyncio.sleep(1.0)
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
