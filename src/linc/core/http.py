"""Async HTTP client abstraction. Default implementation backed by httpx.

The Slack adapter delegates to slack-sdk (which uses aiohttp internally), so it
does not consume HttpClient directly. HttpxClient exists for future webhook /
polling adapters (Telegram / Lark / Wecom in v0.2) that need a shared HTTP
client with proxy + retry policy.

Retry policy (PRD §6.6):
  - Connection errors: exponential backoff, up to N retries.
  - 5xx responses:    exponential backoff, up to N retries.
  - 4xx responses:    raise immediately (no retry).
"""

from __future__ import annotations

import abc
import asyncio
from typing import Any

import httpx


class HttpClient(abc.ABC):
    """Async HTTP client interface."""

    @abc.abstractmethod
    async def startup(self) -> None: ...

    @abc.abstractmethod
    async def shutdown(self) -> None: ...

    @abc.abstractmethod
    async def request(self, method: str, url: str, **kw: Any) -> dict[str, Any]: ...

    async def get(self, url: str, **kw: Any) -> dict[str, Any]:
        return await self.request("GET", url, **kw)

    async def post(self, url: str, **kw: Any) -> dict[str, Any]:
        return await self.request("POST", url, **kw)


class HttpxClient(HttpClient):
    """Default HttpClient backed by httpx.AsyncClient.

    `transport.retries` covers connect errors at the transport layer; 5xx retries
    are implemented in `request()` since httpx's transport does not retry on
    HTTP status alone.
    """

    def __init__(
        self,
        *,
        proxy: str | None = None,
        timeout: float = 30.0,
        retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        self._proxy = proxy
        self._timeout = timeout
        self._retries = retries
        self._backoff_base = backoff_base
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        if self._client is not None:
            return
        transport = httpx.AsyncHTTPTransport(retries=self._retries)
        # httpx 0.28+ uses `proxy=` (singular). We keep it None when unset.
        kwargs: dict[str, Any] = {
            "timeout": self._timeout,
            "transport": transport,
        }
        if self._proxy:
            kwargs["proxy"] = self._proxy
        self._client = httpx.AsyncClient(**kwargs)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HttpxClient not started; call startup() first")
        return self._client

    async def request(self, method: str, url: str, **kw: Any) -> dict[str, Any]:
        """Perform a request and return the JSON-decoded body.

        Retries 5xx responses with exponential backoff. Raises httpx.HTTPStatusError
        for 4xx and any persistent 5xx after retries.
        """
        attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                resp = await self.client.request(method, url, **kw)
            except httpx.ConnectError as e:
                # transport.retries already attempted; if we still get here,
                # apply our own backoff once more before giving up.
                last_exc = e
                if attempt >= self._retries:
                    raise
                await asyncio.sleep(self._backoff_base * (2**attempt))
                attempt += 1
                continue

            if 500 <= resp.status_code < 600 and attempt < self._retries:
                await asyncio.sleep(self._backoff_base * (2**attempt))
                attempt += 1
                continue

            resp.raise_for_status()  # 4xx -> immediate raise
            if not resp.content:
                return {}
            return resp.json()
