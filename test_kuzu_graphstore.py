import pytest
import os
import time
from core.storage.graph.kuzu_impl import * # noqa


class TestVecStore:
    @classmethod
    def setup_class(cls):
        cls.kuzu_path = "test_kuzu_graphstore"
        cls.graph_store = None

    @pytest.mark.asyncio
    async def test_initialize(self):
        self.graph_store = KuzuGraphStore(
            db_path=self.kuzu_path,
        )

    @pytest.mark.asyncio
    async def test_graphstore(self):
        if not self.graph_store:
            await self.test_initialize()
        nodes_sample = [
            PhaseNode(
                id="1",
                ts=int(time.time()),
                name="Alice",
                type="user",
            ),
            PhaseNode(
                id="2",
                ts=int(time.time()),
                name="Bob",
                type="user",
            ),
            PhaseNode(
                id="3",
                ts=int(time.time()),
                name="Charlie",
                type="user",
            ),
        ]
        edges_sample = [
            PhaseEdge(
                source="1",
                target="2",
                ts=int(time.time()),
                relation_type="friend",
                user_id="user_1",
                fact_id="fact_1",
            ),
            PhaseEdge(
                source="2",
                target="3",
                ts=int(time.time()),
                relation_type="colleague",
                user_id="user_2",
                fact_id="fact_2",
            ),
        ]
        # Create nodes
        for node in nodes_sample:
            self.graph_store.add_phase_node(node)
        # Create edges
        for edge in edges_sample:
            self.graph_store.add_phase_edge(edge)
        # Query nodes
        nodes = list(self.graph_store.get_phase_nodes())
        print("Nodes in the graph store:", nodes)
        assert len(nodes) == len(nodes_sample)
        # Query edges
        edges = list(self.graph_store.get_phase_edges())
        print("Edges in the graph store:", edges)
        assert len(edges) == len(edges_sample)
        # Query specific node
        node = list(self.graph_store.get_phase_nodes(filter={"name": "Alice"}))
        print("Query alice node:", node)
        assert len(node) == 1
        assert node[0].name == "Alice"
        # Query specific edge
        edges = list(self.graph_store.get_phase_edges(filter={"relation_type": "friend"}))
        print("Query friend edge:", edges)
        assert len(edges) == 1
        assert edges[0].relation_type == "friend"
        # Test find_phase_node_by_name()
        node_id = self.graph_store.find_phase_node_by_name(name="Charlie")
        print("Query node id:", node_id)
        assert node_id == "3"
        # Test get_nodes_by_edge_filter()
        nodes = list(
            self.graph_store.get_phase_nodes_by_fact_id(fact_id="fact_1")
        )
        print("Query friend edge nodes:", nodes)
        assert len(nodes) == 1
        assert nodes[0][0].id == "1"
        assert nodes[0][1].id == "2"
        # Test run_ppr()
        ppr_result = self.graph_store.run_ppr(
            personalization={"1": 1.0},
            user_id="user_1",
        )
        print("PPR result:", ppr_result)
        assert ppr_result is not None

    @classmethod
    def teardown_class(cls):
        import shutil
        if os.path.exists(cls.kuzu_path):
            shutil.rmtree(cls.kuzu_path)
