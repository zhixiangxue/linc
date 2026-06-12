"""Example: a cross-platform LLM chat agent powered by Chak + Bailian Qwen-VL.

Single-command startup — ``launch()`` handles gateway + client in one line::

    python examples/llm_agent.py

The agent reads unread messages from all configured platforms, sends each
user message (text + images) to Qwen-VL via chak, and replies back to the
same conversation.

Configuration:
  - Put ``BAILIAN_API_KEY=...`` in the project ``.env`` file, or export it in
    the shell environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import chak

from linc import launch
from linc.core.models import Attachment

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("aiosqlite").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
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


# ---------------------------------------------------------------------------
# Attachment conversion helpers
# ---------------------------------------------------------------------------

_KIND_MAP = {
    "image": chak.Image,
    "audio": chak.Audio,
    "video": chak.Video,
}


def convert_attachments(attachments: list[Attachment]) -> list:
    """Convert linc Attachments to chak attachment objects for LLM consumption.

    Uses Attachment convenience properties (is_image, is_audio, is_video, data_uri)
    for clean type resolution and base64 encoding.
    """
    result: list = []
    for att in attachments:
        if att.is_image:
            kind = "image"
        elif att.is_audio:
            kind = "audio"
        elif att.is_video:
            kind = "video"
        else:
            continue
        url = att.data_uri or att.url
        if url:
            result.append(_KIND_MAP[kind](url))
    return result


def _is_placeholder(text: str) -> bool:
    """Return True if text is a platform-generated placeholder like [图片] or [文件: xx]."""
    t = text.strip()
    return bool(t) and t.startswith("[") and t.endswith("]")


# ---------------------------------------------------------------------------
# Debounce buffer config
# ---------------------------------------------------------------------------

DEBOUNCE_SECONDS = 3.0
"""Wait this long after the last message in a conversation before processing."""


async def main() -> None:
    api_key = get_api_key()
    conversations: dict[tuple[str, str], chak.Conversation] = {}

    # Per-conversation debounce buffer.
    # key = (platform, conv_id)
    # value = {"messages": [...], "last_ts": float}
    pending: dict[tuple[str, str], dict] = {}

    client = await launch("linc.yaml")
    try:
        log.info("llm agent started with %s; debounce=%.1fs", MODEL_URI, DEBOUNCE_SECONDS)
        while True:
            unread = await client.pull()
            now = time.time()

            # 1) Incoming messages -> buffer
            for message in unread:
                if message.sender.is_bot:
                    continue
                has_text = bool((message.content.text or "").strip())
                has_attachments = bool(message.content.attachments)
                if not has_text and not has_attachments:
                    continue
                key = (message.platform, message.conv_id)
                if key not in pending:
                    pending[key] = {"messages": [], "last_ts": now}
                pending[key]["messages"].append(message)
                pending[key]["last_ts"] = now

            # 2) Process conversations whose buffer has been quiet >= DEBOUNCE_SECONDS
            ready_keys = [
                k for k, v in pending.items()
                if now - v["last_ts"] >= DEBOUNCE_SECONDS
            ]

            for key in ready_keys:
                batch = pending.pop(key)
                messages = batch["messages"]

                # Merge text and attachments from all messages in the batch
                texts: list[str] = []
                all_attachments: list[Attachment] = []
                for msg in messages:
                    t = (msg.content.text or "").strip()
                    if t and not _is_placeholder(t):
                        texts.append(t)
                    all_attachments.extend(msg.content.attachments)

                if not texts and not all_attachments:
                    continue

                # Convert attachments for LLM
                chak_attachments = convert_attachments(all_attachments)

                # Get or create conversation
                platform, conv_id = key
                conversation = conversations.get(key)
                if conversation is None:
                    conversation = new_conversation(api_key)
                    conversations[key] = conversation

                human_text = "\n".join(texts) if texts else "请处理这个附件"
                reply = await ask_llm(conversation, human_text, chak_attachments or None)
                if not reply:
                    reply = "我暂时没有生成有效回复，请再试一次。"

                messenger = client.messenger(platform)
                sender = messages[0].sender.name or messages[0].sender.id
                log.info(
                    "[%s/%s] %s -> %s (merged %d msgs)",
                    platform, conv_id, sender, reply[:100], len(messages),
                )
                await messenger.send(reply, conv_id=conv_id)

            await asyncio.sleep(1.0)
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
