from astrbot.api.provider import Provider
from astrbot.api import logger
from ..util.prompts import SUMMARIZE_PROMPT
from dataclasses import dataclass


@dataclass
class Summarization:
    pass


class Summarize:
    """对对话进行总结"""

    def __init__(self, provider: Provider):
        self.provider = provider

    async def summarize(self, context: str) -> str:
        """Summarize the given context using the provider."""
        logger.debug(f"Summarizing - {context}")
        summarize_res = await self.provider.text_chat(
            system_prompt=SUMMARIZE_PROMPT,
            prompt=context,
        )
        return summarize_res.completion_text
