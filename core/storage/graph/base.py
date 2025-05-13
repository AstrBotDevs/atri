from typing import Protocol, Iterable
from dataclasses import dataclass


@dataclass
class GraphNode:
    id: str
    properties: dict


@dataclass
class GraphEdge:
    source: str
    """Source node ID"""
    target: str
    """Target node ID"""
    properties: dict


class GraphStore(Protocol):
    def add_node(self, node: GraphNode) -> None: ...
    def add_edge(self, edge: GraphEdge) -> None: ...
    def find_node(self, filter: dict) -> str | None: ...
    def get_node(self, node_id: str) -> GraphNode: ...
    def get_edges(self, filter: dict = {}) -> Iterable[GraphEdge]: ...
    def get_nodes(self, filter: dict = {}) -> Iterable[GraphNode]: ...
    def get_nodes_by_edge_filter(
        self, filter: dict = {}
    ) -> Iterable[tuple[GraphNode, GraphNode]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def run_ppr(
        self,
        personalization: dict[str, float],
        filter: dict = None,
        damping_factor: float = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """运行 Personalize PageRank 算法。

        Args:
            reset_prob (list[float]): 重置概率，表示每个节点的重置概率。
            damping_factor (float): 阻尼因子，通常设置为 0.5。
            max_iter (int): 最大迭代次数。
            tol (float): 收敛容忍度，表示当 PageRank 分数的变化小于该值时停止迭代。
        """
        ...
