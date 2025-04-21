import networkx as nx
import uuid
import time
from astrbot.api.provider import Provider
from ..util.prompts import EXTRACT_ENTITES_PROMPT, BUILD_RELATIONS_PROMPT
from ..util.misc import parse_json
from ..storage.vec_db import VecDB
from ..provider.embedding import EmbeddingProvider
from dataclasses import dataclass


@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str


class GraphMemory:
    def __init__(
        self,
        provider: Provider,
        file_path: str = None,
        embedding_provider: EmbeddingProvider = None,
        vec_db: VecDB = None,
    ) -> None:
        self.provider = provider
        self.file_path = file_path
        self.G = None
        self.embedding_provider = embedding_provider
        self.vec_db = vec_db
        if file_path:
            self.load_graph(file_path)
        else:
            self.G = nx.Graph()

    async def load_graph(self, file_path: str) -> None:
        """从文件加载图"""
        self.G = nx.read_gpickle(file_path)
        if not isinstance(self.G, nx.Graph):
            raise ValueError(f"File {file_path} is not a valid graph file.")

    async def add_to_graph(self, text: str) -> None:
        """将文本添加到图中

        1. Extract entities from the text.
        2. Build relations between entities.
        3. Store the graph in memory.
        """
        entities = await self.get_entities(text)
        relations = await self.build_relations(entities, text)
        # Store the graph in memory
        timestamp = int(time.time())
        for entity in entities:
            self.G.add_node(entity.name, type=entity.type, ts=timestamp)
        for relation in relations:
            fact_id = str(uuid.uuid4())
            self.G.add_edge(
                relation.source,
                relation.target,
                relation_type=relation.relation_type,
                ts=timestamp,
                fact_id=fact_id,  # 将 fact ID 放在边上
            )
            fact = f"{relation.source} {relation.relation_type} {relation.target}"
            _ = self.vec_db.insert(
                content=fact,
                id=fact_id,
            )
        await self.save_graph(self.file_path)

    async def search_graph(self, query: str) -> list[dict]:
        results = self.vec_db.retrieve(
            query=query,
            k=5,
        )
        print(f"Search Docs results: {results}")
        ids = []
        for result in results:
            ids.append(result["id"])
        # 通过 ID 获取边，进而得到所有实体
        related_entities = set()
        for id in ids:
            edges = self.G.edges(data=True)
            for edge in edges:
                if edge[2]["fact_id"] == id:
                    related_entities.add(edge[0])
                    related_entities.add(edge[1])
        print(f"Related entities: {related_entities}")

    async def run_ppo(self, entities: set):
        """运行 PPO 算法。

        References:
            1. https://arxiv.org/pdf/2502.14802
        """
        pass

    async def save_graph(self, file_path: str) -> None:
        """将图保存到文件"""
        nx.write_gpickle(self.G, file_path)

    async def get_entities(self, text: str) -> list[Entity]:
        """从文本中获取实体"""
        llm_response = await self.provider.text_chat(
            prompt=text,
            system_prompt=EXTRACT_ENTITES_PROMPT,
            # func_tool=create_astrbot_func_mgr([EXTRACT_ENTITIES_TOOL]),
        )
        cleaned_data = parse_json(llm_response.completion_text)
        entites_data = cleaned_data.get("entities", [])
        entites = []
        for entity in entites_data:
            entites.append(
                Entity(
                    name=entity.get("name"),
                    type=entity.get("type"),
                )
            )
        return entites

    async def get_relations(self, entities: dict) -> list[Relation]:
        pass

    async def build_relations(self, entities: dict, text: str) -> list[Relation]:
        """构建实体之间的关系"""
        prompt = f"""
# Extracted entities:
```
{entities}
```
# Original text:
`{text}`
"""
        llm_response = await self.provider.text_chat(
            prompt=prompt,
            system_prompt=BUILD_RELATIONS_PROMPT,
            # func_tool=[BUILD_RELATIONS_TOOL],
        )
        cleaned_data = parse_json(llm_response.completion_text)
        relations_data = cleaned_data.get("relations", [])
        relations = []
        for relation in relations_data:
            relations.append(
                Relation(
                    source=relation.get("source"),
                    target=relation.get("target"),
                    relation_type=relation.get("relation_type"),
                )
            )
        return relations
