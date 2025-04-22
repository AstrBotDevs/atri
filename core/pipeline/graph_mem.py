import networkx as nx
import numpy as np
import uuid
import time
from collections import defaultdict
from astrbot.api.provider import Provider
from ..util.prompts import EXTRACT_ENTITES_PROMPT, BUILD_RELATIONS_PROMPT
from ..util.misc import parse_json
from ..storage.vec_db import VecDB
from ..provider.embedding import EmbeddingProvider
from dataclasses import dataclass

PASSAGE_NODE_TYPE = "passage"
PHASE_NODE_TYPE = "phase"
PASSAGE_PHASE_RELATION_TYPE = "_include_"


@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str


# @dataclass
# class Node:
#     id: str
#     node_type: str
#     ts: int


# @dataclass
# class PhaseNode(Node):
#     name: str
#     type: str


# @dataclass
# class PassageNode(Node):
#     summary: str


class GraphMemory:
    def __init__(
        self,
        provider: Provider,
        file_path: str = None,
        embedding_provider: EmbeddingProvider = None,
        vec_db: VecDB = None,
        vec_db_summary: VecDB = None,
    ) -> None:
        self.provider = provider
        self.file_path = file_path
        self.G = None
        self.embedding_provider = embedding_provider
        self.vec_db = vec_db  # 用于存储 Fact 的 VecDB
        self.vec_db_summary = vec_db_summary  # 用于存储摘要的 VecDB
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
        timestamp = int(time.time())
        # Add the passage node
        summary_id = str(uuid.uuid4())
        _ = await self.vec_db_summary.insert(
            text,
            id=summary_id,
        )
        self.G.add_node(
            summary_id, node_type=PASSAGE_NODE_TYPE, summary=text, ts=timestamp
        )
        # Add the phase nodes
        _node_id = {}
        for entity in entities:
            _node_id[entity.name] = str(uuid.uuid4())
            self.G.add_node(
                _node_id[entity.name],
                node_type=PHASE_NODE_TYPE,
                name=entity.name,
                type=entity.type,
                ts=timestamp,
            )
            # phase node - passage node
            self.G.add_edge(
                summary_id,
                _node_id[entity.name],
                relation_type=PASSAGE_PHASE_RELATION_TYPE,
                ts=timestamp,
            )
        for relation in relations:
            fact_id = str(uuid.uuid4())
            if relation.source not in _node_id or relation.target not in _node_id:
                continue
            self.G.add_edge(
                _node_id[relation.source],
                _node_id[relation.target],
                relation_type=relation.relation_type,
                ts=timestamp,
                fact_id=fact_id,  # 将 fact ID 放在边上
            )
            fact = f"{relation.source} {relation.relation_type} {relation.target}"
            _ = await self.vec_db.insert(
                content=fact,
                id=fact_id,
            )
        await self.save_graph(self.file_path)

    async def search_graph(self, query: str, num_to_retrieval: int = 5) -> list[dict]:
        # --- FACT RESULTS
        results = await self.vec_db.retrieve(
            query=query,
            k=5,
        )
        print(f"Search Fact results: {results}")
        # 通过 ID 获取边，进而得到所有实体
        final_related_node_score: dict[str, float] = {}
        related_node_scores = defaultdict(list[float])
        edges = self.G.edges(data=True)
        for result in results:
            for edge in edges:
                if edge[2].get("fact_id") == result.data["id"]:
                    related_node_scores[edge[0]].append(result.similarity)
                    related_node_scores[edge[1]].append(result.similarity)
        print(f"Related phase entities: {related_node_scores}")
        for node, scores in related_node_scores.items():
            final_related_node_score[node] = np.mean(scores)

        # --- SUMMARY RESULTS
        summary_results = await self.vec_db_summary.retrieve(
            query=query,
            k=3,
        )
        # related_passage_nodes: set[tuple[str, float]] = set()
        related_passage_node_scores: dict[str, float] = {}
        for result in summary_results:
            related_passage_node_scores[result.data["id"]] = result.similarity

        # 执行 PPR 算法，得到最终的文档
        ranked_docs = await self.run_ppr(
            seed_phase_nodes=related_node_scores,
            seed_passage_nodes=related_passage_node_scores,
        )
        ret = {}
        i = 0
        for id, score in ranked_docs.items():
            ret[id] = self.G.nodes[id].get("summary", None)
            i += 1
            if i >= num_to_retrieval:
                break
        print(f"Ranked passage nodes: {ret}")
        return ret

    async def run_ppr(
        self,
        seed_phase_nodes: dict[str, float],
        seed_passage_nodes: dict[str, float],
        damping_factor: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        passage_node_reset_factor: float = 0.05,
    ) -> dict[str, float]:
        """运行 Personalize PageRank 算法。

        Args:
            reset_prob (list[float]): 重置概率，表示每个节点的重置概率。
            damping_factor (float): 阻尼因子，通常设置为 0.5。
            max_iter (int): 最大迭代次数。
            tol (float): 收敛容忍度，表示当 PageRank 分数的变化小于该值时停止迭代。
            passage_node_reset_factor (float): 段落节点的重置权重，论文中默认设置为 0.05。

        References:
            1. https://arxiv.org/pdf/2502.14802
        """

        for seed_passage_node, score in seed_passage_nodes.items():
            # 将 passage node 的重置概率设置为 passage_node_reset_factor
            seed_phase_nodes[seed_passage_node] = passage_node_reset_factor * score

        ranked_scores = nx.pagerank(
            self.G,
            alpha=damping_factor,
            personalization=seed_phase_nodes + seed_passage_nodes,
            max_iter=max_iter,
            tol=tol,
        )

        passage_node_ids = await self._get_passage_node_ids()

        doc_scores = np.array([[id, ranked_scores[id]] for id in passage_node_ids])
        doc_scores = doc_scores[doc_scores[:, 1].argsort()[::-1]]
        ranked_docs = {id: score for id, score in doc_scores}  # noqa

        print(f"PageRank scores: {ranked_docs}")
        print(f"Passage node IDs: {passage_node_ids}")

        return ranked_docs

    async def _get_passage_node_ids(self) -> list[str]:
        """获取所有 passage node 的 ID"""
        passage_node_ids = []
        for node, data in self.G.nodes(data=True):
            if data.get("node_type") == PASSAGE_NODE_TYPE:
                passage_node_ids.append(node)
        return passage_node_ids

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
