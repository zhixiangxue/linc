"""Example: a cross-platform LLM chat agent powered by Chak + DeepSeek.

Run this while ``linc serve -c linc.yaml`` is running. The agent reads unread
messages from all configured platforms, sends each user message to DeepSeek via
chak, and replies back to the same conversation.

Configuration:
  - Put ``DEEPSEEK_API_KEY=...`` in the project ``.env`` file, or export it in
    the shell environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import chak

from linc import Linc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm_agent")

SYSTEM_PROMPT = """你是 LINC 示例里的聊天助手。
请用中文与用户自然对话，回答要简洁、友好、有帮助。
如果用户的问题信息不足，先问一个明确的澄清问题。
"""
MODEL_URI = "deepseek/deepseek-chat"
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


def get_deepseek_api_key() -> str:
    load_env_file(ENV_FILE)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Put it in .env or export it before running."
        )
    return api_key


def new_conversation(api_key: str) -> chak.Conversation:
    return chak.Conversation(
        MODEL_URI,
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
    )


async def ask_llm(conversation: chak.Conversation, text: str) -> str:
    response = await conversation.asend(text, timeout=60)
    return str(getattr(response, "content", response)).strip()


async def main() -> None:
    api_key = get_deepseek_api_key()
    conversations: dict[tuple[str, str], chak.Conversation] = {}

    async with Linc(".linc") as linc:
        log.info("llm agent started with %s; waiting for unread messages...", MODEL_URI)
        while True:
            unread = await linc.read_unread_all()
            for message in unread:
                if message.sender.is_bot:
                    continue
                text = message.content.text or ""
                if not text.strip():
                    continue

                conversation_key = (message.platform, message.conv_id)
                conversation = conversations.get(conversation_key)
                if conversation is None:
                    conversation = new_conversation(api_key)
                    conversations[conversation_key] = conversation
                reply = await ask_llm(conversation, text)
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
