"""Unit tests for the Slack adapter — no real network involved.

We never call ``start()``; instead we hand the adapter a stub web client and
exercise ``send`` / ``parse_inbound`` directly. Real Socket Mode connectivity
must be smoke-tested manually with a Slack workspace.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from linc.adapters import is_supported, get
from linc.adapters.slack import SlackAdapter, SlackConfig
from linc.core.errors import SendError
from linc.core.models import Content


def _make_adapter() -> SlackAdapter:
    """Construct a SlackAdapter with stub config and no real connections."""
    cfg = SlackConfig(bot_token="xoxb-test", app_token="xapp-test")
    # hub/store are only used by the default on_event path; parse_inbound and
    # send don't touch them, so MagicMock is fine for these unit tests.
    return SlackAdapter(cfg, hub=MagicMock(), store=MagicMock())


# ============================================================================
# registry
# ============================================================================


def test_slack_is_auto_registered():
    assert is_supported("slack")
    assert get("slack") is SlackAdapter


# ============================================================================
# config
# ============================================================================


def test_slack_config_requires_both_tokens():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SlackConfig(bot_token="xoxb-only")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SlackConfig(app_token="xapp-only")  # type: ignore[call-arg]


# ============================================================================
# parse_inbound
# ============================================================================


def test_parse_inbound_normal_message():
    adapter = _make_adapter()
    parsed = adapter.parse_inbound(
        {
            "type": "message",
            "channel": "C123",
            "user": "U456",
            "text": "hello world",
            "ts": "1700000000.123456",
        }
    )
    assert parsed is not None
    assert parsed.conv_id == "C123"
    assert parsed.msg_id == "1700000000.123456"
    assert parsed.ts == 1700000000.123456
    assert parsed.sender.id == "U456"
    assert parsed.sender.name == "U456"  # v0.1: name = user_id
    assert parsed.content.text == "hello world"


def test_parse_inbound_drops_bot_messages():
    adapter = _make_adapter()
    # Bots echo their own messages back via the Events API. Two ways Slack
    # marks them — both must be filtered to avoid infinite echo loops.
    assert adapter.parse_inbound(
        {
            "type": "message",
            "subtype": "bot_message",
            "channel": "C1", "user": "U1", "text": "hi", "ts": "1.0",
        }
    ) is None
    assert adapter.parse_inbound(
        {
            "type": "message",
            "channel": "C1", "user": "U1", "text": "hi", "ts": "1.0",
            "bot_id": "B999",
        }
    ) is None


def test_parse_inbound_drops_non_message_events():
    adapter = _make_adapter()
    assert adapter.parse_inbound({"type": "reaction_added"}) is None
    assert adapter.parse_inbound({"type": "channel_join"}) is None
    assert adapter.parse_inbound({}) is None


def test_parse_inbound_drops_message_subtypes():
    adapter = _make_adapter()
    # Edits, deletes, channel notifications etc. all carry a `subtype`.
    for sub in ("message_changed", "message_deleted", "channel_topic"):
        raw = {
            "type": "message", "subtype": sub,
            "channel": "C1", "user": "U1", "text": "x", "ts": "1.0",
        }
        assert adapter.parse_inbound(raw) is None


def test_parse_inbound_drops_incomplete_payload():
    adapter = _make_adapter()
    base = {"type": "message", "channel": "C1", "user": "U1", "ts": "1.0"}
    # Each missing field invalidates the event.
    for missing in ("channel", "user", "ts"):
        raw = {k: v for k, v in base.items() if k != missing}
        assert adapter.parse_inbound(raw) is None


def test_parse_inbound_rejects_garbage_ts():
    adapter = _make_adapter()
    raw = {"type": "message", "channel": "C1", "user": "U1", "ts": "not-a-float"}
    assert adapter.parse_inbound(raw) is None


def test_parse_inbound_empty_text_is_ok():
    """Slack allows empty-text messages (e.g. file-only). Don't drop them."""
    adapter = _make_adapter()
    parsed = adapter.parse_inbound(
        {
            "type": "message",
            "channel": "C1", "user": "U1", "text": "", "ts": "1.0",
        }
    )
    assert parsed is not None
    assert parsed.content.text == ""


# ============================================================================
# send
# ============================================================================


def _ok_response(ts: str = "1700000000.111") -> MagicMock:
    """Build a stub SlackResponse-like object."""
    resp = MagicMock()
    resp.data = {"ok": True, "ts": ts, "channel": "C1"}
    return resp


async def test_send_calls_chat_postMessage_with_correct_args():
    adapter = _make_adapter()
    adapter._web = MagicMock()
    adapter._web.chat_postMessage = AsyncMock(return_value=_ok_response("1700000000.111"))

    msg_id, raw = await adapter.send("C1", Content(text="hi there"))

    adapter._web.chat_postMessage.assert_awaited_once_with(channel="C1", text="hi there", mrkdwn=True)
    assert msg_id == "1700000000.111"
    assert raw["ok"] is True
    assert raw["ts"] == "1700000000.111"


async def test_send_empty_text_falls_back_to_empty_string():
    adapter = _make_adapter()
    adapter._web = MagicMock()
    adapter._web.chat_postMessage = AsyncMock(return_value=_ok_response())

    await adapter.send("C1", Content(text=""))
    adapter._web.chat_postMessage.assert_awaited_once_with(channel="C1", text="", mrkdwn=True)


async def test_send_raises_send_error_when_not_started():
    adapter = _make_adapter()
    # _web is None until start()
    with pytest.raises(SendError, match="not started"):
        await adapter.send("C1", Content(text="hi"))


async def test_send_raises_send_error_when_slack_returns_not_ok():
    adapter = _make_adapter()
    bad = MagicMock()
    bad.data = {"ok": False, "error": "channel_not_found"}
    adapter._web = MagicMock()
    adapter._web.chat_postMessage = AsyncMock(return_value=bad)

    with pytest.raises(SendError, match="channel_not_found"):
        await adapter.send("C-missing", Content(text="hi"))


async def test_send_raises_send_error_when_chat_postMessage_throws():
    adapter = _make_adapter()
    adapter._web = MagicMock()
    adapter._web.chat_postMessage = AsyncMock(side_effect=RuntimeError("boom"))

    with pytest.raises(SendError, match="boom"):
        await adapter.send("C1", Content(text="hi"))


async def test_send_raises_send_error_when_response_has_no_ts():
    adapter = _make_adapter()
    weird = MagicMock()
    weird.data = {"ok": True}  # no ts
    adapter._web = MagicMock()
    adapter._web.chat_postMessage = AsyncMock(return_value=weird)

    with pytest.raises(SendError, match="no ts"):
        await adapter.send("C1", Content(text="hi"))
