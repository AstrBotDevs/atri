import pytest
import os
from core.storage.graph.kuzu_impl import KuzuGraphStore, GraphNode, GraphEdge


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
            GraphNode(
                id="1",
                properties={"name": "Alice", "age": 30},
            ),
            GraphNode(
                id="2",
                properties={"name": "Bob", "age": 25},
            ),
            GraphNode(
                id="3",
                properties={"name": "Charlie", "age": 35},
            ),
        ]
        edges_sample = [
            GraphEdge(
                source="1",
                target="2",
                properties={"relationship": "friend"},
            ),
            GraphEdge(
                source="2",
                target="3",
                properties={"relationship": "colleague"},
            ),
        ]
        # Create nodes
        for node in nodes_sample:
            self.graph_store.add_node(node)
        # Create edges
        for edge in edges_sample:
            self.graph_store.add_edge(edge)
        # Query nodes
        nodes = list(self.graph_store.get_nodes())
        print("Nodes in the graph store:", nodes)
        assert len(nodes) == len(nodes_sample)
        # Query edges
        edges = list(self.graph_store.get_edges())
        print("Edges in the graph store:", edges)
        assert len(edges) == len(edges_sample)
        # Query specific node
        node = list(self.graph_store.get_nodes(filter={"name": "Alice"}))
        print("Query alice node:", node)
        assert len(node) == 1
        assert node[0].properties["name"] == "Alice"
        # Query specific edge
        edges = list(self.graph_store.get_edges(filter={"relationship": "friend"}))
        print("Query friend edge:", edges)
        assert len(edges) == 1
        assert edges[0].properties["relationship"] == "friend"
        # Test find_node()
        node_id = self.graph_store.find_node(filter={"id": "1"})
        print("Query node id:", node_id)
        assert node_id == "1"
        # Test get_nodes_by_edge_filter()
        nodes = list(
            self.graph_store.get_nodes_by_edge_filter(filter={"relationship": "friend"})
        )
        print("Query friend edge nodes:", nodes)
        assert len(nodes) == 2
        assert nodes[0].id == "1"
        assert nodes[1].id == "2"
        # Test run_ppr()
        ppr_result = self.graph_store.run_ppr(
            personalization={"1": 1.0},
            filter={"name": "Alice"},
        )
        print("PPR result:", ppr_result)
        assert ppr_result is not None

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.kuzu_path):
            os.remove(cls.kuzu_path)
