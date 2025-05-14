# from astrbot.api.provider import Provider
import datetime
from ..provider.llm.openai_source import ProviderOpenAI
from ..util.prompts import SUMMARIZE_PROMPT
from dataclasses import dataclass


@dataclass
class Summarization:
    pass


class Summarize:
    """对对话进行总结"""

    def __init__(self, provider: ProviderOpenAI):
        self.provider = provider

    async def summarize(self, context: str) -> str:
        """Summarize the given context using the provider."""
        # logger.debug(f"Summarizing - {context}")
        sys_prompt = (
            SUMMARIZE_PROMPT
            + "\nCurrent time: "
            + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        summarize_res = await self.provider.text_chat(
            system_prompt=sys_prompt,
            prompt=context,
        )
        return summarize_res.completion_text
