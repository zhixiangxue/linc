"""Hub: shared, long-lived infrastructure handed to every Adapter.

`Hub` is the single object every Adapter receives at construction time. It
holds the cross-cutting resources that multiple adapters legitimately need to
share rather than each spinning up its own copy.

Why this layer exists (the three "collisions" it resolves):

  1. Shared WebServer port.
     Webhook-style platforms (Lark, Wecom, Slack Events API, Dingtalk...) all
     need an inbound HTTP endpoint. Without a shared WebServer each adapter
     would have to open its own port — exploding the public surface and
     reverse-proxy config. One FastAPI instance + per-adapter route prefixes
     keeps everything on a single port.

  2. Shared HttpClient connection pool.
     Polling-style platforms (Telegram long-polling) and outbound REST calls
     (every adapter sends via HTTPS POST) benefit from a single
     `httpx.AsyncClient` that pools connections per host, applies one proxy /
     timeout policy, and amortizes TLS handshakes.

  3. Future global cross-cutting policies (v0.3+).
     Rate limiting, trace context injection, structured-log enrichment all
     need ONE injection point shared by every adapter. The Hub is that point.

v0.1 reality: only Slack is wired up, and Slack uses Socket Mode (slack-sdk's
own aiohttp connection), so the Hub is effectively an empty container today.
It exists so that adding Telegram / Lark / Wecom in v0.2 does NOT require
changing the Adapter constructor signature.
"""

from __future__ import annotations

from typing import Any

from .http import HttpClient


class Hub:
    """Container for cross-adapter shared infrastructure.

    Not a pydantic model: holds non-data objects (clients, servers) and never
    needs serialization.
    """

    def __init__(
        self,
        http: HttpClient,
        webserver: Any | None = None,  # WebServer in v0.2; typed Any to defer the import
    ) -> None:
        self.http = http
        self.webserver = webserver

    async def startup(self) -> None:
        await self.http.startup()
        if self.webserver is not None:
            await self.webserver.startup()

    async def shutdown(self) -> None:
        # Shut down in reverse order of startup. Best-effort: a failure in one
        # should not prevent the other from running.
        if self.webserver is not None:
            try:
                await self.webserver.shutdown()
            except Exception:
                pass
        try:
            await self.http.shutdown()
        except Exception:
            pass
