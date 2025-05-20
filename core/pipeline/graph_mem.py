import numpy as np
import uuid
import time
import json
import logging
from collections import defaultdict
from ..provider.llm.openai_source import ProviderOpenAI
from ..util.prompts import (
    EXTRACT_ENTITES_PROMPT,
    BUILD_RELATIONS_PROMPT,
    REL_CHECK_PROMPT,
    RESUM_PROMPT,
)
from ..util.misc import parse_json
from ..storage.vec_db import VecDB, Result
from ..provider.embedding import EmbeddingProvider
from ..storage.graph.base import *  # noqa
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
    fact: str


class GraphMemory:
    def __init__(
        self,
        provider: ProviderOpenAI,
        file_path: str = None,
        embedding_provider: EmbeddingProvider = None,
        vec_db: VecDB = None,
        vec_db_summary: VecDB = None,
        graph_store: GraphStore = None,
        logger: logging.Logger = None,
    ) -> None:
        self.provider = provider
        self.file_path = file_path
        # self.G = None
        self.embedding_provider = embedding_provider
        self.vec_db = vec_db  # 用于存储 Fact 的 VecDB
        self.vec_db_summary = vec_db_summary  # 用于存储摘要的 VecDB
        self.graph_store = graph_store

        self.logger = logger or logging.getLogger("astrbot")

    async def get_phase_node(self, entity_name: str) -> str | None:
        """查找是否有对应的 Phase 节点

        Returns:
            节点的 id, 如果没找到, 返回 None
        """
        # for node, data in self.G.nodes(data=True):
        #     if (
        #         data.get("node_type") == PHASE_NODE_TYPE
        #         and data.get("name") == entity_name
        #     ):
        #         return node
        # return None
        return self.graph_store.find_phase_node_by_name(entity_name)

    async def add_to_graph(
        self, text: str, user_id: str, group_id: str = None, username: str = None
    ) -> None:
        """将文本添加到图中

        1. Extract entities from the text.
        2. Build relations between entities.
        3. Store.
        """
        if not username:
            username = user_id

        entities = await self.get_entities(text)
        print(f"Entities: {entities}")

        if not entities:
            self.logger.info(f"对于`{text}`，没有检出任何 entities ")
            return

        relations = await self.build_relations(entities, text)
        print(f"Relations: {relations}")

        if not relations:
            self.logger.info(f"对于`{text}` `{entities}`，没有检出任何 relations ")
            return

        self.logger.info(f"Entities: {entities}")
        self.logger.info(f"Relations: {relations}")
        timestamp = int(time.time())

        # Check Duplicate / Conflict
        # This step may set relation.fact to None
        await self.check_relations(relations, user_id)
        relations = [r for r in relations if r.fact is not None]
        r_entities_name_map = {}
        for entity in relations:
            r_entities_name_map[entity.source] = None
            r_entities_name_map[entity.target] = None
        # make sure the entities are in the relations
        entities = [entity for entity in entities if entity.name in r_entities_name_map]

        # Add the passage node
        summary_id = str(uuid.uuid4())
        self.logger.info(f"Summary insert -> {text} ID: {summary_id}")
        metadata = {
            "user_id": user_id,
        }
        if group_id:
            metadata["group_id"] = group_id
        await self.vec_db_summary.insert(
            text,
            metadata=metadata,
            id=summary_id,  # doc_id
        )
        self.graph_store.add_passage_node(
            PassageNode(id=summary_id, ts=timestamp, user_id=user_id)
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
                self.graph_store.add_phase_node(
                    PhaseNode(
                        id=_node_id[entity_name],
                        ts=timestamp,
                        name=entity_real_name,
                        type=entity.type,
                    )
                )
            self.graph_store.add_passage_edge(
                PassageEdge(
                    source=_node_id[entity_name],
                    target=summary_id,
                    ts=timestamp,
                    relation_type=PASSAGE_PHASE_RELATION_TYPE,
                    summary_id=summary_id,
                    user_id=user_id,
                )
            )
        # Add Edges
        for relation in relations:
            fact_id = str(uuid.uuid4())
            if relation.source not in _node_id or relation.target not in _node_id:
                continue
            self.graph_store.add_phase_edge(
                PhaseEdge(
                    source=_node_id[relation.source],
                    target=_node_id[relation.target],
                    ts=timestamp,
                    relation_type=relation.relation_type,
                    fact_id=fact_id,
                    user_id=user_id,
                )
            )
            # fact = f"{relation.source} {relation.relation_type} {relation.target}"
            if relation.fact:
                fact = relation.fact
            else:
                fact = f"{relation.source} {relation.relation_type} {relation.target}"
            _ = await self.vec_db.insert(
                content=fact,
                id=fact_id,
                metadata={
                    "summary_id": summary_id,
                    "user_id": user_id,
                    "username": username,
                },
            )
        self.graph_store.save(self.file_path)

    async def check_relations(self, relations: list[Relation], user_id: str):
        """检查关系是否重复或冲突，并且做出相关更新"""
        to_be_check = [r.fact for r in relations if r.fact]
        # all_facts: list[Result] = []
        all_facts: dict[str, Result] = {}
        for relation in relations:
            result_facts = await self.vec_db.retrieve(
                query=relation.fact,
                k=3,
                metadata_filters={
                    "user_id": user_id,
                },
            )
            for result in result_facts:
                all_facts[result.data["doc_id"]] = result
        all_facts: list[Result] = list(all_facts.values())
        if not all_facts:
            return
        self.logger.info(f"Check relations: {all_facts}")
        to_be_check_str = ""
        for idx, fact in enumerate(to_be_check):
            to_be_check_str += f"{idx}: {fact}\n"
        all_facts_str = ""
        for idx, fact in enumerate(all_facts):
            all_facts_str += f"{idx}: {fact.data['text']}\n"
        print(f"to_be_check_str: {to_be_check_str}")
        print(f"all_facts_str: {all_facts_str}")
        prompt = REL_CHECK_PROMPT.format(
            new_facts=to_be_check_str,
            existing_facts=all_facts_str,
        )
        # LLM 1
        llm_response = await self.provider.text_chat(
            system_prompt="You are a fact conflict detection expert.",
            prompt=prompt,
        )
        cleaned_data = parse_json(llm_response.completion_text)
        print(f"fact conflict detection LLM response: {cleaned_data}")
        for idx, result in cleaned_data.items():
            assert isinstance(idx, str)
            assert isinstance(result, dict)
            if not idx.isdigit():
                continue
            if "existing_fact_idx" not in result:
                continue
            idx = int(idx)
            existing_fact_idx = int(result["existing_fact_idx"])
            if (
                idx < 0
                or idx >= len(to_be_check)
                or existing_fact_idx < 0
                or existing_fact_idx >= len(all_facts)
            ):
                continue

            if result["result"] == 1:
                # 1 = confict
                self.logger.info(
                    f"Conflict detected: {relations[int(idx)]} with {all_facts[int(result['existing_fact_idx'])]}"
                )
                # re sum
                metadata = json.loads(all_facts[existing_fact_idx].data["metadata"])
                old_summary_id = metadata.get("summary_id", None)
                old_summary = (
                    await self.vec_db_summary.document_storage.get_document_by_doc_id(
                        old_summary_id
                    )
                )
                llm_response_resum = await self.provider.text_chat(
                    system_prompt="You are an intelligent assistant that helps update personal memory summaries.",
                    prompt=RESUM_PROMPT.format(
                        old_summary=old_summary["text"],
                        conflicting_fact=all_facts[existing_fact_idx].data["text"],
                        new_fact=relations[idx].fact,
                    ),
                )
                print(f"llm_response_resum: {llm_response_resum.completion_text}")
                await self.vec_db_summary.document_storage.update_document_by_doc_id(
                    doc_id=old_summary_id,
                    new_text=llm_response_resum.completion_text,
                )
                # delete edge and fact from store
                self.graph_store.delete_phase_edge_by_fact_id(
                    fact_id=all_facts[existing_fact_idx].data["doc_id"]
                )
                await self.vec_db.delete(
                    doc_id=all_facts[existing_fact_idx].data["doc_id"]
                )

            elif result["result"] == 2:
                # 2 = duplicate
                self.logger.info(
                    f"Duplicate detected: {relations[int(idx)]} with {all_facts[int(result['existing_fact_idx'])]}"
                )
                relations[int(idx)].fact = None

    async def search_graph(
        self, query: str, num_to_retrieval: int = 5, filters: dict = None
    ) -> list[dict]:
        # --- FACT RESULTS
        results = await self.vec_db.retrieve(
            query=query,
            k=5,
            metadata_filters=filters,
        )
        print(f"Search Fact results: {results}")
        # 通过 ID 获取边，进而得到所有实体
        final_related_node_score: dict[str, float] = {}
        related_node_scores = defaultdict(list[float])

        for result in results:
            if result.data["doc_id"] == "-1":
                continue
            for n1, n2 in self.graph_store.get_phase_nodes_by_fact_id(
                fact_id=result.data["doc_id"]
            ):
                related_node_scores[n1.id].append(result.similarity)
                related_node_scores[n2.id].append(result.similarity)

        self.logger.info(f"Related phase entities: {str(related_node_scores)}")
        for node, scores in related_node_scores.items():
            final_related_node_score[node] = np.mean(scores)

        # --- SUMMARY RESULTS
        summary_results = await self.vec_db_summary.retrieve(
            query=query,
            k=3,
            metadata_filters=filters,
        )
        related_passage_node_scores: dict[str, float] = {}
        for result in summary_results:
            if result.data["doc_id"] == "-1":
                continue
            related_passage_node_scores[result.data["doc_id"]] = result.similarity

        # 执行 PPR 算法，得到最终的文档
        # Reference: https://arxiv.org/pdf/2502.14802
        for seed_passage_node, score in related_passage_node_scores.items():
            # 将 passage node 的重置概率设置为 passage_node_reset_factor
            related_passage_node_scores[seed_passage_node] = 0.05 * score
        personalization = related_passage_node_scores.copy()
        for node, score in final_related_node_score.items():
            personalization[node] = score

        ranked_docs = await self.run_ppr(
            personalization=personalization,
            user_id=filters.get("user_id", None),  # TODO
        )
        ret = {}
        i = 0
        for doc_id, score in ranked_docs.items():
            # ret[id] = self.G.nodes[id].get("summary", None)
            doc_data = (
                await self.vec_db_summary.document_storage.get_document_by_doc_id(
                    doc_id
                )
            )
            ret[doc_id] = doc_data.get("text", None)
            i += 1
            if i >= num_to_retrieval:
                break
        self.logger.info(f"Ranked passage nodes: {ret}")
        return ret

    async def run_ppr(
        self,
        personalization: dict[str, float],
        user_id: str,
        damping_factor: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        ranked_docs = {}

        self.logger.info(
            f"personalization: {personalization}, max_iter: {max_iter}, tol: {tol}"
        )

        if not personalization:
            personalization = None

        ranked_scores = self.graph_store.run_ppr(
            personalization=personalization,
            user_id=user_id,
            damping_factor=damping_factor,
            max_iter=max_iter,
            tol=tol,
        )

        passage_nodes = await self._get_passage_nodes()

        print("AFTER PPR: ranked_scores", ranked_scores)
        ranked_docs = {}
        for node_id, score in ranked_scores.items():
            if node_id in passage_nodes:
                ranked_docs[node_id] = score

        self.logger.info(f"Ranked doc nodes: {ranked_docs}")
        # {id: (passage_node, score)}
        return ranked_docs

    async def _get_passage_nodes(self, user_id: str = None) -> dict[str, PassageNode]:
        """获取所有 passage node 的 ID"""
        ret = {}
        filter = {}
        if user_id:
            filter["user_id"] = user_id
        for node in self.graph_store.get_passage_nodes(filter=filter):
            ret[node.id] = node
        return ret

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
                    fact=relation.get("fact", None),
                )
            )
        return relations

    async def get_graph(self, filter: dict = None) -> GraphResult:
        """获取图谱"""
        # return self.G
        return self.graph_store.get_graph_networkx(filter)

    async def get_user_ids(self) -> list[str]:
        """获取所有用户 ID"""
        return await self.vec_db.document_storage.get_user_ids()
