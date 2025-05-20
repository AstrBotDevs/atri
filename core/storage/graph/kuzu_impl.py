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
        self.conn.execute(
            (
                "CREATE NODE TABLE IF NOT EXISTS PhaseNode(id STRING, ts TIMESTAMP, name STRING, type STRING, PRIMARY KEY(id));"
                "CREATE NODE TABLE IF NOT EXISTS PassageNode(id STRING, ts TIMESTAMP, user_id STRING, PRIMARY KEY(id));"
                "CREATE REL TABLE IF NOT EXISTS PassageEdge(FROM PhaseNode TO PassageNode, ts TIMESTAMP, relation_type STRING, summary_id STRING, user_id STRING);"
                "CREATE REL TABLE IF NOT EXISTS PhaseEdge(FROM PhaseNode TO PhaseNode, ts TIMESTAMP, relation_type STRING, fact_id STRING, user_id STRING);"
            )
        )

    def add_passage_node(self, node: PassageNode) -> None:
        query = f"MERGE (:PassageNode {{id: '{node.id}', user_id: '{node.user_id}', ts: to_timestamp({node.ts})}});"
        self.conn.execute(query)

    def add_phase_node(self, node: PhaseNode) -> None:
        query = f"MERGE (:PhaseNode {{id: '{node.id}', ts: to_timestamp({node.ts}), name: '{node.name}', type: '{node.type}'}});"
        self.conn.execute(query)

    def add_passage_edge(self, edge: PassageEdge) -> None:
        query = f"""
            MATCH (a:PhaseNode), (b:PassageNode)
            WHERE a.id = '{edge.source}' AND b.id = '{edge.target}'
            MERGE (a)-[:PassageEdge {{ts: to_timestamp({edge.ts}), relation_type: '{edge.relation_type}', summary_id: '{edge.summary_id}', user_id: '{edge.user_id}'}}]->(b)
        """
        self.conn.execute(query)

    def add_phase_edge(self, edge: PhaseEdge) -> None:
        query = f"""
            MATCH (a:PhaseNode), (b:PhaseNode)
            WHERE a.id = '{edge.source}' AND b.id = '{edge.target}'
            MERGE (a)-[:PhaseEdge {{ts: to_timestamp({edge.ts}), relation_type: '{edge.relation_type}', fact_id: '{edge.fact_id}', user_id: '{edge.user_id}'}}]->(b)
        """
        self.conn.execute(query)

    def find_phase_node_by_name(self, name: str) -> str | None:
        result = self.conn.execute(
            f"MATCH (n:PhaseNode) WHERE n.name = '{name}' RETURN n.id;"
        )
        if result.has_next():
            return result.get_next()[0]
        return None

    def get_passage_nodes(self, filter=None) -> Iterable[PassageNode]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                clauses.append(f"n.{k} = '{v}'")
            if clauses:
                where_clause = "WHERE " + " AND ".join(clauses)

        query = f"MATCH (n:PassageNode) {where_clause} RETURN n.id, n.ts, n.user_id;"
        result = self.conn.execute(query)
        while result.has_next():
            id_val, ts, user_id = result.get_next()
            yield PassageNode(id=id_val, ts=ts, user_id=user_id)

    def get_phase_nodes(self, filter=None) -> Iterable[PhaseNode]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                clauses.append(f"n.{k} = '{v}'")
            if clauses:
                where_clause = "WHERE " + " AND ".join(clauses)

        query = f"MATCH (n:PhaseNode) {where_clause} RETURN n.id, n.ts, n.name, n.type;"
        result = self.conn.execute(query)
        while result.has_next():
            id_val, ts, name, type_val = result.get_next()
            yield PhaseNode(id=id_val, ts=ts, name=name, type=type_val)

    def get_passage_edges(self, filter=None) -> Iterable[PassageEdge]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                clauses.append(f"e.{k} = '{v}'")
            if clauses:
                where_clause = "WHERE " + " AND ".join(clauses)

        query = f"""
            MATCH (a:PhaseNode)-[e:PassageEdge]->(b:PassageNode)
            {where_clause}
            RETURN a.id, b.id, e.ts, e.relation_type, e.summary_id, e.user_id;
        """
        result = self.conn.execute(query)
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

    def get_phase_edges(self, filter=None) -> Iterable[PhaseEdge]:
        where_clause = ""
        if filter:
            clauses = []
            for k, v in filter.items():
                clauses.append(f"e.{k} = '{v}'")
            if clauses:
                where_clause = "WHERE " + " AND ".join(clauses)

        query = f"""
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            {where_clause}
            RETURN a.id, b.id, e.ts, e.relation_type, e.fact_id, e.user_id;
        """
        result = self.conn.execute(query)
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
        query = f"""
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            WHERE e.fact_id = '{fact_id}'
            RETURN a, b;
        """
        result = self.conn.execute(query)
        while result.has_next():
            a, b = result.get_next()
            a.pop("_id")
            a.pop("_label")
            b.pop("_id")
            b.pop("_label")
            yield (PhaseNode(**a), PhaseNode(**b))

    def delete_phase_edge_by_fact_id(self, fact_id):
        query = f"""
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            WHERE e.fact_id = '{fact_id}'
            DELETE e;
        """
        self.conn.execute(query)

    def cnt_phase_node_edges(self, node_id: str) -> int:
        query = f"""
            MATCH (a:PhaseNode)-[e:PhaseEdge]->(b:PhaseNode)
            WHERE a.id = '{node_id}' OR b.id = '{node_id}'
            RETURN COUNT(e);
        """
        result = self.conn.execute(query)
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
        query = f"""
        MATCH (u) -[e]-> (v)
        WHERE e.user_id = '{user_id}'
        RETURN u, e, v;
        """
        result = self.conn.execute(query)
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

    def get_graph_networkx(self, filter=None):
        if filter:
            where_clause = "WHERE "
            clauses = []
            for k, v in filter.items():
                clauses.append(f"e.{k} = '{v}'")
            if clauses:
                where_clause += " AND ".join(clauses)
        else:
            where_clause = ""
        query = f"""
            MATCH (a)-[e: PhaseEdge]->(b)
            {where_clause}
            RETURN a, e, b;
        """
        result = self.conn.execute(query)
        G = result.get_as_networkx()
        nodes = list(G.nodes(data=True))
        edges = list(G.edges(data=True))
        return GraphResult(
            nodes=nodes,
            edges=edges,
        )
