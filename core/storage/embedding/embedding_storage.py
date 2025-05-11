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
            self.index = faiss.IndexFlatL2(dimention)
            self.id_map = faiss.IndexIDMap2(self.index)
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
        self.id_map.add_with_ids(vector.reshape(1, -1), np.array([id]))
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

    async def search_by_ids(self, ids: list[int], k: int) -> tuple:
        """根据ID搜索向量

        Args:
            ids (list[int]): 要搜索的ID列表
            k (int): 返回的最相似向量的数量
        Returns:
            tuple: (距离, 索引)
        """
        # restrict FAISS search to filtered ids
        subset_index = faiss.IndexIDMap2(self.index.__class__(self.index.d))
        mask = np.array(ids, dtype=np.int64)
        vectors = self.id_map.reconstruct_n(0, self.id_map.ntotal)
        

    async def save_index(self):
        """保存索引

        Args:
            path (str): 保存索引的路径
        """
        faiss.write_index(self.index, self.path)
