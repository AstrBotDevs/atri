from openai import AsyncOpenAI
from dataclasses import dataclass

@dataclass
class LLMResponse:
    completion_text: str
    raw_response: dict

class ProviderOpenAI:
    def __init__(self, api_key: str, model: str, base_url: str = None):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    async def text_chat(
        self,
        prompt: str,
        context: list = None,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        model = model or self.model
        context = context or []
        query = [*context, {"role": "user", "content": prompt}]
        if system_prompt:
            query.insert(0, {"role": "system", "content": system_prompt})
        resp = await self.client.chat.completions.create(
            model=model,
            messages=query,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            completion_text=resp.choices[0].message.content.strip(),
            raw_response=resp,
        )
