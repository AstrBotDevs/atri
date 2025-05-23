from typing import Iterable
from .base import *  # noqa
import kuzu
import networkx as nx


class KuzuGraphStore(GraphStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        """初始化数据库模式"""
        self.conn.execute(
            (
                "CREATE NODE TABLE IF NOT EXISTS PhaseNode(id STRING, ts TIMESTAMP, name STRING, type STRING, PRIMARY KEY(id));"
                "CREATE NODE TABLE IF NOT EXISTS PassageNode(id STRING, ts TIMESTAMP, user_id STRING, PRIMARY KEY(id));"
                "CREATE REL TABLE IF NOT EXISTS PassageEdge(FROM PhaseNode TO PassageNode, ts TIMESTAMP, relation_type STRING, summary_id STRING, user_id STRING);"
                "CREATE REL TABLE IF NOT EXISTS PhaseEdge(FROM PhaseNode TO PhaseNode, ts TIMESTAMP, relation_type STRING, fact_id STRING, user_id STRING);"
            )
        )

    def add_passage_node(self, node: PassageNode) -> None:
        """添加记忆节点

        Args:
            node (PassageNode): 记忆节点对象
        """
        query = (
            "MERGE (:PassageNode {id: $id, user_id: $user_id, ts: to_timestamp($ts)});"
        )
        params = {"id": node.id, "user_id": node.user_id, "ts": node.ts}
        self.conn.execute(query, params)

    def add_phase_node(self, node: PhaseNode) -> None:
        """添加概念节点

        Args:
            node (PhaseNode): 概念节点对象
        """
        query = "MERGE (:PhaseNode {id: $id, ts: to_timestamp($ts), name: $name, type: $type});"
        params = {"id": node.id, "ts": node.ts, "name": node.name, "type": node.type}
        self.conn.execute(query, params)

    def add_passage_edge(self, edge: PassageEdge) -> None:
        """添加记忆关联边(段落关联边)

        Args:
            edge (PassageEdge): 记忆关联边(段落关联边)对象
        """
        query = """
            MATCH (a:PhaseNode), (b:PassageNode)
            WHERE a.id = $source AND b.id = $target
            MERGE (a)-[:PassageEdge {ts: to_timestamp($ts), relation_type: $relation_type, summary_id: $summary_id, user_id: $user_id}]->(b)
        """
        params = {
            "source": edge.source,
            "target": edge.target,
            "ts": edge.ts,
            "relation_type": edge.relation_type,
            "summary_id": edge.summary_id,
            "user_id": edge.user_id,
        }
        self.conn.execute(query, params)

    def add_phase_edge(self, edge: PhaseEdge) -> None:
        """添加概念关系边(实体关系边)

        Args:
            edge (PhaseEdge): 概念关系边(实体关系边)对象
        """
        query = """
            MATCH (a:PhaseNode), (b:PhaseNode)
            WHERE a.id = $source AND b.id = $target
            MERGE (a)-[:PhaseEdge {ts: to_timestamp($ts), relation_type: $relation_type, fact_id: $fact_id, user_id: $user_id}]->(b)
        """
        params = {
            "source": edge.source,
            "target": edge.target,
            "ts": edge.ts,
            "relation_type": edge.relation_type,
            "fact_id": edge.fact_id,
            "user_id": edge.user_id,
        }
        self.conn.execute(query, params)

    def find_phase_node_by_name(self, name: str) -> str | None:
        """根据名称查找概念节点(实体节点)

        Args:
            name (str): 概念节点(实体节点)名称
        """
        query = "MATCH (n:PhaseNode) WHERE n.name = $name RETURN n.id;"
        params = {"name": name}
        result = self.conn.execute(query, params)
        if result.has_next():
            return result.get_next()[0]
        return None

    def _build_where_clause(
        self, filter: dict, node_alias: str = "n"
    ) -> tuple[str, dict]:
        """构建 WHERE 子句和参数字典

        Args:
            filter (dict): 过滤器字典，键为属性名，值为属性值
            node_alias (str): 节点别名，默认为 "n"

        Returns:
            tuple[str, dict]: WHERE 子句和参数字典
        """
        where_clause = ""
        params = {}
        if filter:
            clauses = []
            for k, v in filter.items():
                param_name = f"param_{k}"
                clauses.append(f"{node_alias}.{k} = ${param_name}")
                params[param_name] = v
            if clauses:
                where_clause = "WHERE " + " AND ".join(clauses)
        return where_clause, params

    def get_passage_nodes(self, filter: dict = {}) -> Iterable[PassageNode]:
        """根据过滤器获取记忆节点(段落节点)

        Args:
            filter (dict): 过滤器字典，键为属性名，值为属性值
        Returns:
            Iterable[PassageNode]: 记忆节点(段落节点)的迭代器
        """
        where_clause, params = self._build_where_clause(filter)
        query = f"MATCH (n:PassageNode) {where_clause} RETURN n.id, n.ts, n.user_id;"
        result = self.conn.execute(query, params)
        while result.has_next():
            id_val, ts, user_id = result.get_next()
            yield PassageNode(id=id_val, ts=ts, user_id=user_id)

    def get_phase_nodes(self, filter: dict = {}) -> Iterable[PhaseNode]:
        """根据过滤器获取概念节点(实体节点)

        Args:
            filter (dict): 过滤器字典，键为属性名，值为属性值
        Returns:
            Iterable[PhaseNode]: 概念节点(实体节点)的迭代器
        """
        where_clause, params = self._build_where_clause(filter)
        query = f"MATCH (n:PhaseNode) {where_clause} RETURN n.id, n.ts, n.name, n.type;"
        result = self.conn.execute(query, params)
        while result.has_next():
            id_val, ts, name, type_val = result.get_next()
            yield PhaseNode(id=id_val, ts=ts, name=name, type=type_val)

    def get_passage_edges(self, filter: dict = {}) -> Iterable[PassageEdge]:
        """根据过滤器获取记忆关联边(段落关联边)

        Args:
            filter (dict): 过滤器字典，键为属性名，值为属性值
        Returns:
            Iterable[PassageEdge]: 记忆关联边(段落关联边)的迭代器
        """
        where_clause, params = self._build_where_clause(filter, node_alias="e")
        query = f"""
            MATCH (a:PhaseNode)-[e:PassageEdge]->(b:PassageNode)
            {where_clause}
            RETURN a.id, b.id, e.ts, e.relation_type, e.summary_id, e.user_id;
        """
        result = self.conn.execute(query, params)
        while result.has_next():
            src, tgt, ts, rel_type, summary_id, user_id = result.get_next()
            yield PassageEdge(
                source=src,
                target=tgt,
                ts=ts,
                relation_type=rel_type,
                summary_id=summary_id,
                user_id=user_id,
            )

    def get_phase_edges(self, filter: dict = {}) -> Iterable[PhaseEdge]:
        """根据过滤器获取概念关系边(实体关系边)

        Args:
            filter (dict): 过滤器字典，键为属性名，值为属性值
        Returns:
            Iterable[PhaseEdge]: 概念关系边(实体关系边)的迭代器
        """
        where_clause, params = self._build_where_clause(filter, node_alias="e")
        query = f"""
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            {where_clause}
            RETURN a.id, b.id, e.ts, e.relation_type, e.fact_id, e.user_id;
        """
        result = self.conn.execute(query, params)
        while result.has_next():
            src, tgt, ts, rel_type, fact_id, user_id = result.get_next()
            yield PhaseEdge(
                source=src,
                target=tgt,
                ts=ts,
                relation_type=rel_type,
                fact_id=fact_id,
                user_id=user_id,
            )

    def get_phase_nodes_by_fact_id(
        self, fact_id: str
    ) -> Iterable[tuple[PhaseNode, PhaseNode]]:
        """根据概念 id 获取两个由该概念关联的概念节点(实体节点)

        Args:
            fact_id (str): 概念 id
        Returns:
            Iterable[tuple[PhaseNode, PhaseNode]]: 概念节点(实体节点)元组的迭代器, 其中每对概念节点都由这个概念 id 对应的概念关联起来
        """
        query = """
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            WHERE e.fact_id = $fact_id
            RETURN a, b;
        """
        params = {"fact_id": fact_id}
        result = self.conn.execute(query, params)
        while result.has_next():
            a, b = result.get_next()
            a.pop("_id")
            a.pop("_label")
            b.pop("_id")
            b.pop("_label")
            yield (PhaseNode(**a), PhaseNode(**b))

    def delete_phase_edge_by_fact_id(self, fact_id: str):
        """根据概念 id 删除概念关系边(实体关系边)

        Args:
            fact_id (str): 概念 id
        """
        query = """
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            WHERE e.fact_id = $fact_id
            DELETE e;
        """
        params = {"fact_id": fact_id}
        self.conn.execute(query, params)

    def cnt_phase_node_edges(self, node_id: str) -> int:
        """统计概念节点(实体节点)的边数

        Args:
            node_id (str): 概念节点(实体节点) id
        Returns:
            int: 概念节点(实体节点)的边数
        """
        query = """
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            WHERE a.id = $node_id OR b.id = $node_id
            RETURN COUNT(e);
        """
        params = {"node_id": node_id}
        result = self.conn.execute(query, params)
        if result.has_next():
            return result.get_next()[0]
        return 0

    def save(self, path: str) -> None:
        # Kuzu automatically persists to disk; left for interface compatibility
        pass

    def load(self, path: str) -> None:
        # Already loaded via __init__; left for interface compatibility
        pass

    def run_ppr(
        self,
        personalization,
        user_id,
        damping_factor=0.5,
        max_iter=100,
        tol=0.000001,
    ):
        """运行 PPR 算法

        Args:
            personalization (dict): 节点的个性化分数，键为节点 ID，值为分数
            user_id (str): 用户 ID
            damping_factor (float): 阻尼因子，通常设置为 0.5
            max_iter (int): 最大迭代次数
            tol (float): 收敛容忍度，表示当 PageRank 分数的变化小于该值时停止迭代
        """
        query = """
        MATCH (u) -[e]-> (v)
        WHERE e.user_id = $user_id
        RETURN u, e, v;
        """
        params = {"user_id": user_id}
        result = self.conn.execute(query, params)
        G = result.get_as_networkx()
        nodes = list(G.nodes(data=True))
        new_personalization = {}
        for node_id, data in nodes:
            id = data.get("id")
            new_personalization[node_id] = personalization.get(id, 0.0)
        # print("Graph Metadata:", G.nodes(data=True), G.edges(data=True))
        # print("Personalization:", new_personalization)
        ranked_scores: dict[str, float] = nx.pagerank(
            G,
            alpha=damping_factor,
            personalization=new_personalization,
            max_iter=max_iter,
            tol=tol,
        )
        # ranked_scores = dict
        m = {}
        for k, v in ranked_scores.items():
            m[G.nodes[k]["id"]] = v
        m = dict(sorted(m.items(), key=lambda item: item[1], reverse=True))
        return m

    def get_graph_networkx(self, filter: dict = {}) -> GraphResult:
        """获取图结构, 可以根据过滤器获取

        Args:
            filter (dict): 过滤器，键为属性名，值为属性值
        Returns:
            GraphResult: 图结构结果，包含节点和边的列表
        """
        where_clause, params = self._build_where_clause(filter, node_alias="e")
        query = f"""
            MATCH (a)-[e: PhaseEdge]->(b)
            {where_clause}
            RETURN a, e, b;
        """
        result = self.conn.execute(query, params)
        G = result.get_as_networkx()
        nodes = list(G.nodes(data=True))
        edges = list(G.edges(data=True))
        return GraphResult(
            nodes=nodes,
            edges=edges,
        )
