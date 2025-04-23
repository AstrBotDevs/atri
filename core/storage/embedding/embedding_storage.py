import faiss
import os
import numpy as np


class EmbeddingStorage:
    def __init__(self, dimention: int, path: str = None):
        self.dimention = dimention
        self.path = path
        self.index = None
        if path and os.path.exists(path):
            self.index = faiss.read_index(path)
        else:
            base_index = faiss.IndexFlatL2(dimention)
            self.index = faiss.IndexIDMap(base_index)
        self.storage = {}

    async def insert(self, vector: np.ndarray, id: int):
        """插入向量

        Args:
            vector (np.ndarray): 要插入的向量
            id (int): 向量的ID
        Raises:
            ValueError: 如果向量的维度与存储的维度不匹配
        """
        if vector.shape[0] != self.dimention:
            raise ValueError(
                f"向量维度不匹配, 期望: {self.dimention}, 实际: {vector.shape[0]}"
            )
        self.index.add_with_ids(vector.reshape(1, -1), np.array([id]))
        self.storage[id] = vector
        await self.save_index()

    async def search(self, vector: np.ndarray, k: int) -> tuple:
        """搜索最相似的向量

        Args:
            vector (np.ndarray): 查询向量
            k (int): 返回的最相似向量的数量
        Returns:
            tuple: (距离, 索引)
        """
        distances, indices = self.index.search(vector.reshape(1, -1), k)
        return distances, indices

    async def save_index(self):
        """保存索引

        Args:
            path (str): 保存索引的路径
        """
        faiss.write_index(self.index, self.path)
