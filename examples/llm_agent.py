"""Example: a cross-platform LLM chat agent powered by Chak + Bailian Qwen-VL.

Run this while ``linc serve -c linc.yaml`` is running. The agent reads unread
messages from all configured platforms, sends each user message (text + images)
to Qwen-VL via chak, and replies back to the same conversation.

Configuration:
  - Put ``BAILIAN_API_KEY=...`` in the project ``.env`` file, or export it in
    the shell environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import chak

from linc import Linc

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
log = logging.getLogger("llm_agent")

SYSTEM_PROMPT = """你是 LINC 示例里的聊天助手。
请用中文与用户自然对话，回答要简洁、友好、有帮助。
如果用户的问题信息不足，先问一个明确的澄清问题。
如果用户发送了图片，请仔细查看图片内容并根据用户的文字说明进行回复。
"""
MODEL_URI = "bailian/qwen3.6-plus"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    load_env_file(ENV_FILE)
    api_key = os.getenv("BAILIAN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "BAILIAN_API_KEY is not set. Put it in .env or export it before running."
        )
    return api_key


def new_conversation(api_key: str) -> chak.Conversation:
    return chak.Conversation(
        MODEL_URI,
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
    )


async def ask_llm(
    conversation: chak.Conversation,
    text: str,
    attachments: list | None = None,
) -> str:
    """Send a message (with optional image attachments) to the LLM."""
    kwargs: dict = {"timeout": 60}
    if attachments:
        kwargs["attachments"] = attachments
        log.debug("ask_llm: text=%r, attachments=%d, types=%s",
                   text[:100], len(attachments), [type(a).__name__ for a in attachments])
    else:
        log.debug("ask_llm: text=%r, no attachments", text[:100])
    try:
        response = await conversation.asend(text, **kwargs)
        result = str(getattr(response, "content", response)).strip()
        log.debug("ask_llm response: %r", result[:200] if result else "<empty>")
        return result
    except Exception as e:
        log.exception("ask_llm failed: %s", e)
        return ""


async def main() -> None:
    api_key = get_api_key()
    conversations: dict[tuple[str, str], chak.Conversation] = {}

    async with Linc(".linc") as linc:
        log.info("llm agent started with %s; waiting for unread messages...", MODEL_URI)
        while True:
            unread = await linc.read_unread_all()
            for message in unread:
                if message.sender.is_bot:
                    continue
                text = message.content.text or ""
                has_attachments = bool(message.content.attachments)

                # Skip messages that have neither text nor attachments.
                if not text.strip() and not has_attachments:
                    continue

                # Convert linc Attachment -> chak attachment by kind.
                # Supported: image → chak.Image, audio → chak.Audio,
                # video → chak.Video. Other types (file, etc.) are skipped
                # for now — most LLMs only accept media types.
                # If a local ``path`` is available (downloaded by the adapter),
                # read the file and encode as base64 data URI so the LLM's
                # server can access it without auth.
                _KIND_MAP = {
                    "image": chak.Image,
                    "audio": chak.Audio,
                    "video": chak.Video,
                }
                chak_attachments: list = []
                for att in message.content.attachments:
                    if att.kind not in _KIND_MAP:
                        continue
                    url = None
                    if att.path:
                        try:
                            import base64 as _b64
                            from pathlib import Path as _Path
                            data = _Path(att.path).read_bytes()
                            b64 = _b64.b64encode(data).decode("ascii")
                            mime = att.mime or "application/octet-stream"
                            url = f"data:{mime};base64,{b64}"
                            log.debug("loaded local file %s (%d bytes)", att.path, len(data))
                        except Exception:
                            log.exception("failed to read local file %s", att.path)
                    if not url:
                        url = att.url
                    if url:
                        chak_attachments.append(_KIND_MAP[att.kind](url))

                conversation_key = (message.platform, message.conv_id)
                conversation = conversations.get(conversation_key)
                if conversation is None:
                    conversation = new_conversation(api_key)
                    conversations[conversation_key] = conversation

                human_text = text.strip() or "请处理这个附件"
                reply = await ask_llm(conversation, human_text, chak_attachments or None)
                if not reply:
                    reply = "我暂时没有生成有效回复，请再试一次。"

                client = linc.get(message.platform)
                sender = message.sender.name or message.sender.id
                log.info(
                    "[%s/%s] %s (%s) -> %s",
                    message.platform,
                    message.conv_id,
                    sender,
                    message.sender.id,
                    reply,
                )
                await client.send(reply, conv_id=message.conv_id)
            await asyncio.sleep(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
