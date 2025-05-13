import os
import logging
from astrbot.api.provider import Provider
from .provider.embedding.nomic_embed import NomicEmbeddingProvider
from .storage.vec_db import VecDB
from .storage.documents.document_storage import DocumentStorage
from .storage.embedding.embedding_storage import EmbeddingStorage
from .storage.graph.kuzu_impl import KuzuGraphStore
from .pipeline.graph_mem import GraphMemory
from .pipeline.summarize import Summarize

logger = logging.getLogger("astrbot")


class ATRIMemoryStarter:
    def __init__(self, data_dir_path: str, llm_provider: Provider):
        self.data_dir_path = data_dir_path
        self.llm_provider = llm_provider

        if not os.path.exists(self.data_dir_path):
            os.makedirs(self.data_dir_path)

    async def initialize(self):
        """可选择实现异步的插件初始化方法，当实例化该插件类之后会自动调用该方法。"""
        self.summarizer = Summarize(provider=self.llm_provider)

        self.fact_db_path = os.path.join(self.data_dir_path, "mem_fact.db")
        self.embedding_db_path = os.path.join(self.data_dir_path, "mem_fact.faiss")

        self.summary_db_path = os.path.join(self.data_dir_path, "mem_sum.db")
        self.summary_embedding_db_path = os.path.join(
            self.data_dir_path, "mem_sum.faiss"
        )

        self.mem_graph_path = os.path.join(self.data_dir_path, "mem_graph")

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

        # SUMMARY VEC DB
        self.summary_docs_store = DocumentStorage(self.summary_db_path)
        self.summary_vec_store = EmbeddingStorage(
            self.vec_dim, self.summary_embedding_db_path
        )
        await self.summary_docs_store.initialize()
        self.summary_vec_db = VecDB(
            document_storage=self.summary_docs_store,
            embedding_storage=self.summary_vec_store,
            embedding_provider=self.embedding_model,
        )

        # graph store
        kuzu_graph_store = KuzuGraphStore(
            db_path=self.mem_graph_path,
        )

        # TODO: 支持当用户更换提供商时，这里也同步更换
        self.provider = self.llm_provider
        self.graph_memory = GraphMemory(
            provider=self.provider,
            file_path=self.mem_graph_path,
            embedding_provider=self.embedding_model,
            vec_db=self.fact_vec_db,
            vec_db_summary=self.summary_vec_db,
            graph_store=kuzu_graph_store,
        )
        logger.info("Graph memory initialized successfully.")
