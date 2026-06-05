"""Unit tests for the DingTalk adapter — no real network involved.

We never call ``start()``; instead we inject a mock stream_client and
exercise ``send`` / ``parse_inbound`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linc.adapters import is_supported, get
from linc.adapters.dingtalk import DingtalkAdapter, DingtalkConfig
from linc.core.errors import SendError
from linc.core.models import Attachment, Content


def _make_adapter() -> DingtalkAdapter:
    """Construct a DingtalkAdapter with stub config and no real connections."""
    cfg = DingtalkConfig(
        client_id="test_app_key",
        client_secret="test_app_secret",
        robot_code="test_robot",
    )
    adapter = DingtalkAdapter(cfg, hub=MagicMock(), store=MagicMock())
    return adapter


def _make_started_adapter() -> DingtalkAdapter:
    """Return an adapter with a mocked _stream_client ready for send tests."""
    adapter = _make_adapter()
    # Mock the stream client — the key dependency for send operations
    mock_client = MagicMock()
    mock_client.get_access_token.return_value = "fake_access_token_123"
    mock_client.upload_to_dingtalk.return_value = "@lADPfake_media_id_12345"
    adapter._stream_client = mock_client
    adapter._robot_code = "test_robot"
    return adapter


# ============================================================================
# registry
# ============================================================================


def test_dingtalk_is_auto_registered():
    assert is_supported("dingtalk")
    assert get("dingtalk") is DingtalkAdapter


# ============================================================================
# parse_inbound
# ============================================================================


def test_parse_inbound_text_message():
    adapter = _make_adapter()
    parsed = adapter.parse_inbound(
        {
            "message_type": "text",
            "text": "hello dingtalk",
            "sender_staff_id": "user123",
            "sender_nick": "Alice",
            "conversation_type": "1",
            "conversation_id": "conv001",
            "message_id": "msg001",
        }
    )
    assert parsed is not None
    assert parsed.conv_id == "user:user123"
    assert parsed.msg_id == "msg001"
    assert parsed.sender.id == "user123"
    assert parsed.sender.name == "Alice"
    assert parsed.content.text == "hello dingtalk"


def test_parse_inbound_group_message():
    adapter = _make_adapter()
    parsed = adapter.parse_inbound(
        {
            "message_type": "text",
            "text": "group hello",
            "sender_staff_id": "user456",
            "sender_nick": "Bob",
            "conversation_type": "2",
            "conversation_id": "cidXYZ",
            "message_id": "msg002",
        }
    )
    assert parsed is not None
    assert parsed.conv_id == "group:cidXYZ"


def test_parse_inbound_picture_message():
    adapter = _make_adapter()
    parsed = adapter.parse_inbound(
        {
            "message_type": "picture",
            "sender_staff_id": "user789",
            "sender_nick": "Charlie",
            "conversation_type": "1",
            "conversation_id": "conv003",
            "message_id": "msg003",
            "_image_codes": ["download_code_abc"],
        }
    )
    assert parsed is not None
    assert parsed.content.text == "[图片]"
    assert len(parsed.content.attachments) == 1
    assert parsed.content.attachments[0].kind == "image"
    assert parsed.content.attachments[0].meta["download_code"] == "download_code_abc"


def test_parse_inbound_drops_unsupported_type():
    adapter = _make_adapter()
    assert adapter.parse_inbound({"message_type": "video"}) is None


def test_parse_inbound_drops_empty_text():
    adapter = _make_adapter()
    assert adapter.parse_inbound(
        {
            "message_type": "text",
            "text": "",
            "sender_staff_id": "u1",
            "conversation_type": "1",
            "conversation_id": "c1",
            "message_id": "m1",
        }
    ) is None


def test_parse_inbound_drops_unknown_conversation_type():
    adapter = _make_adapter()
    assert adapter.parse_inbound(
        {
            "message_type": "text",
            "text": "hi",
            "sender_staff_id": "u1",
            "conversation_type": "99",
            "conversation_id": "c1",
            "message_id": "m1",
        }
    ) is None


# ============================================================================
# send — text only
# ============================================================================


async def test_send_text_only_to_user():
    adapter = _make_started_adapter()

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"processQueryKey": "pqk_001"}
        mock_post.return_value = mock_resp

        msg_id, raw = await adapter.send("user:staff001", Content(text="hello"))

    assert msg_id == "pqk_001"
    # Verify the API was called with markdown msg
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["msgKey"] == "sampleMarkdown"
    assert payload["userIds"] == ["staff001"]
    msg_param = json.loads(payload["msgParam"])
    assert msg_param["text"] == "hello"


async def test_send_text_only_to_group():
    adapter = _make_started_adapter()

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"processQueryKey": "pqk_002"}
        mock_post.return_value = mock_resp

        msg_id, raw = await adapter.send("group:conv123", Content(text="hi group"))

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["msgKey"] == "sampleMarkdown"
    assert payload["openConversationId"] == "conv123"


# ============================================================================
# send — image attachments
# ============================================================================


async def test_send_image_with_local_path(tmp_path: Path):
    adapter = _make_started_adapter()

    # Create a fake image file
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG fake image data")

    content = Content(
        attachments=[Attachment(kind="image", path=str(img_file))],
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"processQueryKey": "pqk_img_001"}
        mock_post.return_value = mock_resp

        msg_id, raw = await adapter.send("user:staff001", content)

    # upload_to_dingtalk should have been called with the file bytes
    adapter._stream_client.upload_to_dingtalk.assert_called_once_with(
        b"\x89PNG fake image data",
        filetype="image",
        filename="test.png",
        mimetype="image/png",
    )

    # The send API should have been called with sampleImageMsg
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert payload["msgKey"] == "sampleImageMsg"
    msg_param = json.loads(payload["msgParam"])
    assert msg_param["photoURL"] == "@lADPfake_media_id_12345"


async def test_send_text_and_image(tmp_path: Path):
    """Text + image should result in two API calls: markdown then image."""
    adapter = _make_started_adapter()

    img_file = tmp_path / "photo.jpg"
    img_file.write_bytes(b"\xff\xd8\xff fake jpeg")

    content = Content(
        text="Look at this!",
        attachments=[Attachment(kind="image", path=str(img_file))],
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"processQueryKey": "pqk_mixed"}
        mock_post.return_value = mock_resp

        msg_id, raw = await adapter.send("user:staff001", content)

    # Should have made 2 API calls: markdown + image
    assert mock_post.call_count == 2

    # First call: markdown
    first_payload = mock_post.call_args_list[0].kwargs.get("json") or mock_post.call_args_list[0][1].get("json")
    assert first_payload["msgKey"] == "sampleMarkdown"

    # Second call: image
    second_payload = mock_post.call_args_list[1].kwargs.get("json") or mock_post.call_args_list[1][1].get("json")
    assert second_payload["msgKey"] == "sampleImageMsg"


async def test_send_image_with_url():
    """Image with URL should download then upload."""
    adapter = _make_started_adapter()

    content = Content(
        attachments=[Attachment(kind="image", url="https://example.com/pic.png")],
    )

    with patch("httpx.AsyncClient.post") as mock_post, \
         patch("httpx.AsyncClient.get") as mock_get:
        # Mock the image download
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.raise_for_status = MagicMock()
        mock_get_resp.content = b"\x89PNG downloaded image"
        mock_get.return_value = mock_get_resp

        # Mock the send API
        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200
        mock_post_resp.raise_for_status = MagicMock()
        mock_post_resp.json.return_value = {"processQueryKey": "pqk_url_img"}
        mock_post.return_value = mock_post_resp

        msg_id, raw = await adapter.send("user:staff001", content)

    # upload_to_dingtalk should have been called (with temp file bytes)
    adapter._stream_client.upload_to_dingtalk.assert_called_once()
    assert msg_id == "pqk_url_img"


async def test_send_skips_image_without_path_or_url():
    """Image attachment with neither path nor url should be skipped gracefully."""
    adapter = _make_started_adapter()

    content = Content(
        text="some text",
        attachments=[Attachment(kind="image")],  # no path, no url
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"processQueryKey": "pqk_text_only"}
        mock_post.return_value = mock_resp

        msg_id, raw = await adapter.send("user:staff001", content)

    # Only 1 call (the markdown), image was skipped
    assert mock_post.call_count == 1
    payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
    assert payload["msgKey"] == "sampleMarkdown"


# ============================================================================
# send — error handling
# ============================================================================


async def test_send_raises_send_error_when_not_started():
    adapter = _make_adapter()
    # _stream_client is None until start()
    with pytest.raises(SendError, match="not started"):
        await adapter.send("user:u1", Content(text="hi"))


async def test_send_raises_send_error_on_invalid_conv_id():
    adapter = _make_started_adapter()

    with patch("httpx.AsyncClient.post"):
        with pytest.raises(SendError, match="invalid conv_id"):
            await adapter.send("bad_prefix:123", Content(text="hi"))


async def test_send_raises_send_error_on_upload_failure(tmp_path: Path):
    adapter = _make_started_adapter()
    adapter._stream_client.upload_to_dingtalk.return_value = None  # upload fails

    img_file = tmp_path / "fail.png"
    img_file.write_bytes(b"fake")

    content = Content(
        attachments=[Attachment(kind="image", path=str(img_file))],
    )

    with pytest.raises(SendError, match="upload_to_dingtalk returned no media_id"):
        await adapter.send("user:u1", content)
