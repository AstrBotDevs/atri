import typing as T
from astrbot.api.provider import Provider
from ..util.prompts import SUMMARIZE_PROMPT
from dataclasses import dataclass


@dataclass
class Summarization:
    pass


class Summarize:
    """对对话进行总结"""

    def __init__(self, provider: Provider):
        self.provider = provider

    async def summarize(self, context: T.List[T.Dict]) -> str:
        """Summarize the given text using the provider."""
        summarize_res = await self.provider.text_chat(
            system_prompt=SUMMARIZE_PROMPT,
            prompt=await self.assemble_context(context),
        )
        return summarize_res.completion_text

    async def assemble_context(self, context: T.List[T.Dict]) -> str:
        """Assemble the context into a single string."""
        # TODO: 适配群聊多用户的场景。
        ret = ""
        for item in context:
            if item["role"] == "user" and "content" in item:
                ret += f"User: {item['content']}\n"
            elif item["role"] == "assistant" and "content" in item:
                ret += f"Assistant: {item['content']}\n"
