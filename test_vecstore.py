import pytest
import os


class TestVecStore:
    @classmethod
    def setup_class(cls):
        cls.fact_db_path = "test_vecstore.db"
        cls.embedding_db_path = "test_vecstore_embeddings.db"
        cls.embedding_model = None
        cls.vec_dim = None
        cls.fact_docs_store = None
        cls.fact_vec_store = None
        cls.fact_vec_db = None

    @pytest.mark.asyncio
    async def test_initialize(self):
        from core.storage.embedding.embedding_storage import EmbeddingStorage
        from core.storage.documents.document_storage import DocumentStorage
        from core.provider.embedding.nomic_embed import NomicEmbeddingProvider
        from core.storage.vec_db import VecDB

        # Initialize the embedding model and vector dimension
        self.embedding_model = NomicEmbeddingProvider()
        self.vec_dim = await self.embedding_model.get_dim()

        # FACT VEC DB
        self.fact_docs_store = DocumentStorage(self.fact_db_path)
        self.fact_vec_store = EmbeddingStorage(self.vec_dim, self.embedding_db_path)
        await self.fact_docs_store.initialize()
        self.fact_vec_db = VecDB(
            document_storage=self.fact_docs_store,
            embedding_storage=self.fact_vec_store,
            embedding_provider=self.embedding_model,
        )

    @pytest.mark.asyncio
    async def test_vecstore(self):
        if not self.fact_vec_db:
            await self.test_initialize()
        cases = [
            {
                "content": "北海道真好玩",
                "metadata": {"author": "test_user"},
            },
            {
                "content": "苹果是我最喜欢的水果",
                "metadata": {"user_id": "test_user"},
            },
            {
                "content": "Python 是一种流行的编程语言",
                "metadata": {"user_id": "test_user"},
            },
            {
                "content": "我喜欢在周末去爬山",
                "metadata": {"user_id": "test_user"},
            },
            {
                "content": "今天的天气真好",
                "metadata": {"user_id": "test_user"},
            },
            {
                "content": "我喜欢喝咖啡",
                "metadata": {"user_id": "test_user"},
            },
            {
                "content": "日本街道好整洁",
                "metadata": {"user_id": "test_user"},
            },
            {
                "content": "我今天去了北海道",
                "metadata": {"user_id": "test_user"},
            },
        ]
        for case in cases:
            content = case["content"]
            metadata = case["metadata"]
            doc_id = await self.fact_vec_db.insert(content, metadata)
            assert doc_id is not None

        # retrieval
        query = "北海道真好看"
        metadata_filters = {"user_id": "test_user"}
        k = 3
        results = await self.fact_vec_db.retrieve(
            query=query,
            k=k,
            metadata_filters=metadata_filters,
        )
        print(f"Results: {results}")
        assert doc_id is not None

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.fact_db_path):
            os.remove(cls.fact_db_path)
        if os.path.exists(cls.embedding_db_path):
            os.remove(cls.embedding_db_path)
