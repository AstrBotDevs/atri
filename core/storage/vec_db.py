import uuid
import json
import numpy as np
from .documents.document_storage import DocumentStorage
from .embedding.embedding_storage import EmbeddingStorage
from ..provider.embedding import EmbeddingProvider


class VecDB:
    """
    A class to represent a vector database.
    """

    def __init__(
        self,
        document_storage: DocumentStorage,
        embedding_storage: EmbeddingStorage,
        embedding_provider: EmbeddingProvider = None,
    ):
        self.document_storage = document_storage
        self.embedding_storage = embedding_storage
        self.embedding_provider = embedding_provider

    async def insert(self, content: str, metadata: dict = None, id: str = None) -> int:
        """
        插入一条文本和其对应向量，自动生成 ID 并保持一致性。
        """
        metadata = metadata or {}
        str_id = id or str(uuid.uuid4()) # 使用 UUID 作为原始 ID

        # 获取向量
        vector = await self.embedding_provider.get_embedding(content)
        vector = np.array(vector, dtype=np.float32)

        # 插入 SQLite 获取自增 int_id
        async with self.document_storage.connection.cursor() as cursor:
            await cursor.execute(
                "INSERT INTO documents (id, text, meta) VALUES (?, ?, ?)",
                (str_id, content, json.dumps(metadata)),
            )
            await self.document_storage.connection.commit()
            int_id = cursor.lastrowid

        # 插入向量到 FAISS
        self.embedding_storage.insert(vector, int_id)
        return int_id

    async def retrieve(self, query: str, k: int = 5) -> list:
        """
        搜索最相似的文档。

        Returns:
            List[dict]: 查询结果，每个结果包含 id, content, metadata
        """
        embedding = await self.embedding_provider.get_embedding(query)
        _, indices = self.embedding_storage.search(embedding, k)
        result_docs = []

        for idx in indices[0]:
            if idx == -1:
                continue
            doc = await self.document_storage.get_document(idx)
            if doc:
                result_docs.append(doc)
        return result_docs

    async def delete(self, doc_id: int):
        """
        删除一条文档（同时从 SQLite 中删除）
        """
        await self.document_storage.connection.execute(
            "DELETE FROM documents WHERE id = ?", (doc_id,)
        )
        await self.document_storage.connection.commit()

    async def close(self):
        await self.document_storage.close()
