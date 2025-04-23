import numpy as np
from . import EmbeddingProvider
from nomic import embed


class NomicEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "nomic-embed-text-v1.5") -> None:
        self.model = model
        super().__init__()

    async def get_embedding(self, text):
        output = embed.text(
            texts=[text],
            model=self.model,
            task_type="search_document",
            inference_mode="local",
        )
        return np.array(output["embeddings"][0])

    async def get_dim(self):
        return 768
