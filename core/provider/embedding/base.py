import abc
import numpy as np


class EmbeddingProvider:
    @abc.abstractmethod
    async def get_embedding(self, text) -> np.ndarray:
        """
        获取文本的嵌入
        """
        ...
