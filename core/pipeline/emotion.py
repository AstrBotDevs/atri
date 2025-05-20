from enum import Enum
import json

from astrbot.api import logger
from ..provider.llm.openai_source import ProviderOpenAI
from ..util.prompts import EMOTION_ANALYSIS_PROMPT


class EmotionTendency(Enum):
    """情绪倾向枚举"""

    POSITIVE = "positive"
    """积极"""
    NEGATIVE = "negative"
    """消极"""
    NEUTRAL = "neutral"
    """中性"""


class Emotion(Enum):
    """情绪枚举"""

    JOY = "joy"
    """喜悦"""
    CONTENTMENT = "contentment"
    """满足"""
    SURPRISE = "surprise"
    """惊讶"""
    NEUTRAL = "neutral"
    """中性"""
    FEAR = "fear"
    """恐惧"""
    SADNESS = "sadness"
    """悲伤"""
    ANGER = "anger"
    """愤怒"""
    DISGUST = "disgust"
    """厌恶"""
    PANIC = "panic"
    """恐慌"""

    @property
    def tendency(self) -> EmotionTendency:
        """获取情绪倾向: POSITIVE(积极), NEGATIVE(消极), NEUTRAL(中性)"""
        positive = {Emotion.JOY, Emotion.CONTENTMENT}
        negative = {
            Emotion.FEAR,
            Emotion.SADNESS,
            Emotion.ANGER,
            Emotion.DISGUST,
            Emotion.PANIC,
        }

        if self in positive:
            return EmotionTendency.POSITIVE
        elif self in negative:
            return EmotionTendency.NEGATIVE
        return EmotionTendency.NEUTRAL

    @property
    def prompt(self) -> str:
        """返回情绪提示"""
        return f"Your current emotion is {self.value}. The output should reflect this emotion. If there is an emotional shift, ensure the transition is natural to simulate human emotional changes."


class EmotionAnalysis:
    """情绪分析类"""

    def __init__(self, provider: ProviderOpenAI):
        self.provider = provider

    async def analyze(self, text: dict) -> dict | None:
        """分析文本情绪"""
        sys_prompt = EMOTION_ANALYSIS_PROMPT.format(Emotion=Emotion)
        res = await self.provider.text_chat(
            system_prompt=sys_prompt,
            prompt=json.dumps(text),
        )
        try:
            cleaned_text = (
                res.completion_text.replace("```json", "")
                .replace("```", "")
                .replace("{{", "{")
                .replace("}}", "}")
                .strip()
            )
            resp = json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {cleaned_text}")
            return None

        return resp
