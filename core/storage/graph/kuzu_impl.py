from typing import Iterable
from .base import GraphNode, GraphEdge, GraphStore
import kuzu
import json
import networkx as nx


class KuzuGraphStore(GraphStore):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute(
            (
                "INSTALL json;"
                "LOAD json;"
                "CREATE NODE TABLE IF NOT EXISTS Node(id STRING, properties JSON, PRIMARY KEY(id));"
                "CREATE REL TABLE IF NOT EXISTS Edge(FROM Node TO Node, properties JSON);"
            )
        )

    def add_node(self, node: GraphNode) -> None:
        prop_str = json.dumps(node.properties, ensure_ascii=False)
        query = f"""
            MERGE INTO Node(id, properties)
            VALUES ('{node.id}', to_json('{prop_str.replace("'", "''")}'))
        """
        self.conn.execute(query)

    def add_edge(self, edge: GraphEdge) -> None:
        prop_str = json.dumps(edge.properties, ensure_ascii=False)
        query = f"""
            MATCH (a:Node), (b:Node)
            WHERE a.id = '{edge.source}' AND b.id = '{edge.target}'
            MERGE (a)-[:Edge {{properties: to_json('{prop_str.replace("'", "''")}')}}]->(b)
        """
        self.conn.execute(query)

    def find_node(self, filter: dict) -> str | None:
        if "id" in filter:
            result = self.conn.execute(
                f"MATCH (n:Node) WHERE n.id = '{filter['id']}' RETURN n.id;"
            )
            return result.get_next()[0] if result.has_next() else None
        return None

    def get_node(self, node_id: str) -> GraphNode:
        result = self.conn.execute(
            f"MATCH (n:Node) WHERE n.id = '{node_id}' RETURN n.id, n.properties;"
        )
        if result.has_next():
            id_val, prop_str = result.get_next()
            return GraphNode(id=id_val, properties=json.loads(prop_str))
        raise ValueError(f"Node '{node_id}' not found")

    def get_edges(self, filter: dict = {}) -> Iterable[GraphEdge]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                val = json.dumps(v, ensure_ascii=False).replace("'", "''")
                clauses.append(f"JSON_EXTRACT(e.properties, '$.{k}') = '{val}'")
            where_clause = "WHERE " + " AND ".join(clauses)

        query = f"""
            MATCH (a:Node)-[e:Edge]->(b:Node)
            {where_clause}
            RETURN a.id, b.id, e.properties;
        """
        result = self.conn.execute(query)
        while result.has_next():
            src, tgt, prop_str = result.get_next()
            yield GraphEdge(source=src, target=tgt, properties=json.loads(prop_str))

    def get_nodes(self, filter: dict = {}) -> Iterable[GraphNode]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                val = json.dumps(v, ensure_ascii=False).replace("'", "''")
                clauses.append(f"JSON_EXTRACT(n.properties, '$.{k}') = '{val}'")
            where_clause = "WHERE " + " AND ".join(clauses)

        query = f"MATCH (n:Node) {where_clause} RETURN n.id, n.properties;"
        result = self.conn.execute(query)
        while result.has_next():
            id_val, prop_str = result.get_next()
            yield GraphNode(id=id_val, properties=json.loads(prop_str))

    def get_nodes_by_edge_filter(
        self, filter: dict = {}
    ) -> Iterable[tuple[GraphNode, GraphNode]]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                val = json.dumps(v, ensure_ascii=False).replace("'", "''")
                clauses.append(f"JSON_EXTRACT(e.properties, '$.{k}') = '{val}'")
            where_clause = "WHERE " + " AND ".join(clauses)

        query = f"""
            MATCH (a:Node)-[e:Edge]->(b:Node)
            {where_clause}
            RETURN a.id, a.properties, b.id, b.properties;
        """
        result = self.conn.execute(query)
        while result.has_next():
            a_id, a_props, b_id, b_props = result.get_next()
            yield (
                GraphNode(id=a_id, properties=json.loads(a_props)),
                GraphNode(id=b_id, properties=json.loads(b_props)),
            )

    def save(self, path: str) -> None:
        # Kuzu automatically persists to disk; left for interface compatibility
        pass

    def load(self, path: str) -> None:
        # Already loaded via __init__; left for interface compatibility
        pass

    def run_ppr(
        self,
        personalization,
        filter=None,
        damping_factor=0.5,
        max_iter=100,
        tol=0.000001,
    ):
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                val = json.dumps(v, ensure_ascii=False).replace("'", "''")
                clauses.append(f"JSON_EXTRACT(e.properties, '$.{k}') = '{val}'")
            where_clause = "WHERE " + " AND ".join(clauses)

        query = f"""
            MATCH (u:Node)-[r:Edge]->(m:Node)
            {where_clause}
            RETURN u, r, m;
        """
        result = self.conn.execute(query)
        G = result.get_as_networkx()
        ranked_scores: dict[str, float] = nx.pagerank(
            G,
            alpha=damping_factor,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
        )
        return ranked_scores
