"""Example: routing one agent across multiple IM platforms.

Demonstrates the cross-platform API surface:
  - ``linc.read_unread_all()`` — atomically claim unread from every platform.
  - dispatch back to the correct platform via ``getattr(linc, m.platform)()``.

For v0.1 only the Slack adapter ships, so in practice you'll see only Slack
messages — but the loop is identical when more adapters land in v0.2 (Lark,
Wecom, Telegram, ...). Just list each in ``linc.yaml`` under ``adapters:``.

Run alongside ``linc serve`` exactly like the other examples.
"""

from __future__ import annotations

import asyncio
import logging

from linc import Linc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("multi_platform")


async def main() -> None:
    async with Linc(".linc") as linc:
        log.info("multi-platform agent started; ctrl-c to stop")
        while True:
            # One claim across all platforms — the gateway does NOT need to be
            # told which platforms to listen on; the registry already knows.
            unread = await linc.read_unread_all()
            for m in unread:
                # Dispatch back to the same platform the message arrived on.
                client = getattr(linc, m.platform)()
                reply = f"[{m.platform}] echo: {m.content.text or ''}"
                log.info("[%s/%s] %s", m.platform, m.conv_id, reply)
                await client.send(reply, conv_id=m.conv_id)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
