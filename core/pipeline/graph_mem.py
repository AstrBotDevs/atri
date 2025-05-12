import networkx as nx
import numpy as np
import uuid
import time
import os
import logging
import pickle
import json
from collections import defaultdict
from astrbot.api.provider import Provider
from ..util.prompts import EXTRACT_ENTITES_PROMPT, BUILD_RELATIONS_PROMPT
from ..util.misc import parse_json
from ..storage.vec_db import VecDB
from ..provider.embedding import EmbeddingProvider
from dataclasses import dataclass
from typing import Dict, Any, Tuple

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


class GraphMemory:
    def __init__(
        self,
        provider: Provider,
        file_path: str = None,
        embedding_provider: EmbeddingProvider = None,
        vec_db: VecDB = None,
        vec_db_summary: VecDB = None,
        logger: logging.Logger = None,
    ) -> None:
        self.provider = provider
        self.file_path = file_path
        self.G = None
        self.embedding_provider = embedding_provider
        self.vec_db = vec_db  # 用于存储 Fact 的 VecDB
        self.vec_db_summary = vec_db_summary  # 用于存储摘要的 VecDB
        if file_path and os.path.exists(file_path):
            self.load_graph(file_path)
        else:
            self.G = nx.Graph()

        self.logger = logger or logging.getLogger("astrbot")

    def load_graph(self, file_path: str) -> None:
        """从文件加载图"""
        with open(file_path, "rb") as f:
            self.G = pickle.load(f)
        if not isinstance(self.G, nx.Graph):
            raise ValueError(f"File {file_path} is not a valid graph file.")

    async def get_phase_node(self, entity_name: str) -> str | None:
        """查找是否有对应的 Phase 节点

        Returns:
            节点的 id, 如果没找到, 返回 None
        """
        for node, data in self.G.nodes(data=True):
            if (
                data.get("node_type") == PHASE_NODE_TYPE
                and data.get("name") == entity_name
            ):
                return node
        return None

    async def add_to_graph(self,
        text: str,
        user_id: str,
        group_id: str = None,
        username: str = None
    ) -> None:
        """将文本添加到图中

        1. Extract entities from the text.
        2. Build relations between entities.
        3. Store the graph in memory.
        """
        if not username:
            username = user_id

        entities = await self.get_entities(text)

        if not entities:
            self.logger.info(f"对于`{text}`，没有检出任何 entities ")
            return

        relations = await self.build_relations(entities, text)

        if not relations:
            self.logger.info(f"对于`{text}` `{entities}`，没有检出任何 relations ")
            return

        self.logger.info(f"Entities: {entities}")
        self.logger.info(f"Relations: {relations}")
        timestamp = int(time.time())
        # Add the passage node
        summary_id = str(uuid.uuid4())
        self.logger.info(f"Summary insert -> {text} ID: {summary_id}")
        metadata = {
            "user_id": user_id,
        }
        if group_id:
            metadata["group_id"] = group_id
        _ = await self.vec_db_summary.insert(
            text,
            metadata=metadata,
            id=summary_id, # doc_id
        )
        self.G.add_node(
            summary_id, node_type=PASSAGE_NODE_TYPE, summary=text, ts=timestamp
        )

        # Add the phase nodes
        _node_id = {}
        for entity in entities:
            entity_name = entity.name
            entity_real_name = entity.name.replace("USER_ID", user_id)
            if node := await self.get_phase_node(entity_real_name):
                self.logger.info(f"Phase node already exists: {node}")
                _node_id[entity_name] = node
            else:
                _node_id[entity_name] = str(uuid.uuid4())
            self.G.add_node(
                _node_id[entity_name],
                node_type=PHASE_NODE_TYPE,
                name=entity_real_name,
                user_id=user_id,
                username=username,
                type=entity.type,
                ts=timestamp,
            )
            # phase node - passage node
            self.G.add_edge(
                summary_id,
                _node_id[entity_name],
                relation_type=PASSAGE_PHASE_RELATION_TYPE,
                ts=timestamp,
            )
        for relation in relations:
            fact_id = str(uuid.uuid4())
            if relation.source not in _node_id or relation.target not in _node_id:
                continue
            self.G.add_edge(
                _node_id[relation.source],  # entity_uuid
                _node_id[relation.target],  # entity_uuid
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

    async def search_graph(self, query: str, num_to_retrieval: int = 5, filters: dict = None) -> list[dict]:
        # --- FACT RESULTS
        results = await self.vec_db.retrieve(
            query=query,
            k=5,
            metadata_filters=filters,
        )
        self.logger.info(f"Search Fact results: {results}")
        # 通过 ID 获取边，进而得到所有实体
        final_related_node_score: dict[str, float] = {}
        related_node_scores = defaultdict(list[float])
        edges = self.G.edges(data=True)
        for result in results:
            for edge in edges:
                if edge[2].get("fact_id") == result.data["doc_id"]:
                    related_node_scores[edge[0]].append(result.similarity)
                    related_node_scores[edge[1]].append(result.similarity)
        self.logger.info(f"Related phase entities: {str(related_node_scores)}")
        for node, scores in related_node_scores.items():
            final_related_node_score[node] = np.mean(scores)

        # --- SUMMARY RESULTS
        summary_results = await self.vec_db_summary.retrieve(
            query=query,
            k=3,
            metadata_filters=filters,
        )
        # related_passage_nodes: set[tuple[str, float]] = set()
        related_passage_node_scores: dict[str, float] = {}
        for result in summary_results:
            related_passage_node_scores[result.data["doc_id"]] = result.similarity

        # 执行 PPR 算法，得到最终的文档
        ranked_docs = await self.run_ppr(
            seed_phase_nodes=final_related_node_score,
            seed_passage_nodes=related_passage_node_scores,
        )
        ret = {}
        i = 0
        for id, score in ranked_docs.items():
            ret[id] = self.G.nodes[id].get("summary", None)
            i += 1
            if i >= num_to_retrieval:
                break
        self.logger.info(f"Ranked passage nodes: {ret}")
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
        ranked_docs = {}

        for seed_passage_node, score in seed_passage_nodes.items():
            # 将 passage node 的重置概率设置为 passage_node_reset_factor
            seed_passage_nodes[seed_passage_node] = passage_node_reset_factor * score

        personalization = seed_passage_nodes.copy()
        for node, score in seed_phase_nodes.items():
            personalization[node] = score

        self.logger.info(
            f"personalization: {personalization}, max_iter: {max_iter}, tol: {tol}"
        )

        if not personalization:
            personalization = None

        ranked_scores = nx.pagerank(
            self.G,
            alpha=damping_factor,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
        )

        passage_node_ids = await self._get_passage_node_ids()

        doc_scores = np.array([[id, ranked_scores[id]] for id in passage_node_ids])
        self.logger.info(f"Doc scores: {doc_scores}")
        if len(doc_scores) > 0:
            doc_scores = doc_scores[doc_scores[:, 1].argsort()[::-1]]
            ranked_docs = {id: score for id, score in doc_scores}  # noqa

        self.logger.info(f"PageRank scores: {ranked_docs}")
        self.logger.info(f"Passage node IDs: {passage_node_ids}")

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
        # nx.write_gpickle(self.G, file_path)
        with open(file_path, "wb") as f:
            pickle.dump(self.G, f)

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

    async def visualize(
        self,
        output_path: str = "graph_visualization.png",
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 500,
        show_labels: bool = True,
    ) -> str:
        """可视化图结构并保存为图片

        Args:
            output_path: 图片保存路径
            figsize: 图像大小
            node_size: 节点大小
            show_labels: 是否显示标签

        Returns:
            保存的图片路径
        """
        import matplotlib.pyplot as plt

        if not self.G:
            self.logger.warning("无法可视化：图为空")
            return None

        plt.figure(figsize=figsize)

        # 为不同类型的节点设置不同的颜色
        node_colors = []
        for node in self.G.nodes():
            if self.G.nodes[node].get("node_type") == PASSAGE_NODE_TYPE:
                node_colors.append("skyblue")
            else:
                node_colors.append("lightgreen")

        # 创建节点标签
        node_labels = {}
        for node in self.G.nodes():
            if self.G.nodes[node].get("node_type") == PASSAGE_NODE_TYPE:
                summary = self.G.nodes[node].get("summary", "")
                node_labels[node] = (
                    summary[:20] + "..." if len(summary) > 20 else summary
                )
            else:
                node_labels[node] = self.G.nodes[node].get("name", str(node))

        # 为不同类型的边设置不同的颜色
        edge_colors = []
        for edge in self.G.edges():
            relation_type = self.G.edges[edge].get("relation_type")
            if relation_type == PASSAGE_PHASE_RELATION_TYPE:
                edge_colors.append("gray")
            else:
                edge_colors.append("red")

        # 使用spring布局算法
        pos = nx.spring_layout(self.G)

        # 绘制图
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=node_size)
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=1.0)

        if show_labels:
            nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=8)

        plt.title("图存储可视化")
        plt.axis("off")  # 隐藏坐标轴
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        self.logger.info(f"图可视化已保存至: {output_path}")
        return output_path

    async def export_to_json(self, output_path: str = "graph_data.json") -> str:
        """导出图到JSON格式，便于使用其他可视化工具

        Args:
            output_path: JSON文件保存路径

        Returns:
            保存的文件路径
        """
        if not self.G:
            self.logger.warning("无法导出：图为空")
            return None

        # 准备导出数据
        graph_data = {"nodes": [], "links": []}

        # 添加节点
        for node_id, node_data in self.G.nodes(data=True):
            node_info = {"id": node_id}
            node_info.update(node_data)
            # 将datetime等不可JSON序列化的对象转为字符串
            for k, v in node_info.items():
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    node_info[k] = str(v)
            graph_data["nodes"].append(node_info)

        # 添加边
        for source, target, edge_data in self.G.edges(data=True):
            edge_info = {"source": source, "target": target}
            edge_info.update(edge_data)
            # 将datetime等不可JSON序列化的对象转为字符串
            for k, v in edge_info.items():
                if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    edge_info[k] = str(v)
            graph_data["links"].append(edge_info)

        # 保存为JSON文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"图数据已导出至: {output_path}")
        return output_path

    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图的统计信息

        Returns:
            包含图统计信息的字典
        """
        if not self.G:
            return {"error": "图为空"}

        stats = {
            "节点总数": self.G.number_of_nodes(),
            "边总数": self.G.number_of_edges(),
            "段落节点数": 0,
            "实体节点数": 0,
            "关系类型统计": defaultdict(int),
            "实体类型统计": defaultdict(int),
        }

        # 计算节点类型
        for _, node_data in self.G.nodes(data=True):
            if node_data.get("node_type") == PASSAGE_NODE_TYPE:
                stats["段落节点数"] += 1
            elif node_data.get("node_type") == PHASE_NODE_TYPE:
                stats["实体节点数"] += 1
                entity_type = node_data.get("type")
                if entity_type:
                    stats["实体类型统计"][entity_type] += 1

        # 计算边关系类型
        for _, _, edge_data in self.G.edges(data=True):
            relation_type = edge_data.get("relation_type")
            if relation_type:
                stats["关系类型统计"][relation_type] += 1

        # 将defaultdict转为dict以便JSON序列化
        stats["关系类型统计"] = dict(stats["关系类型统计"])
        stats["实体类型统计"] = dict(stats["实体类型统计"])

        return stats
