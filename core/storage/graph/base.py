from typing import Protocol, Iterable
from dataclasses import dataclass


@dataclass
class BaseNode:
    id: str
    ts: int
    """Seconds"""


@dataclass
class PassageNode(BaseNode):
    user_id: str


@dataclass
class PhaseNode(BaseNode):
    name: str
    type: str


@dataclass
class BaseEdge:
    source: str
    """Source node ID"""
    target: str
    """Target node ID"""
    ts: int
    """Seconds"""
    relation_type: str
    user_id: str


@dataclass
class PassageEdge(BaseEdge):
    pass

@dataclass
class PhaseEdge(BaseEdge):
    fact_id: str
    """Fact ID"""


class GraphStore(Protocol):
    def add_passage_node(self, node: PassageNode) -> None: ...
    def add_phase_node(self, node: PhaseNode) -> None: ...
    def add_passage_edge(self, edge: PassageEdge) -> None: ...
    def add_phase_edge(self, edge: PhaseEdge) -> None: ...
    def find_phase_node_by_name(self, name: str) -> str | None: ...
    def get_passage_nodes(self, filter: dict = {}) -> Iterable[PassageNode]: ...
    def get_phase_nodes(self, filter: dict = {}) -> Iterable[PhaseNode]: ...
    def get_passage_edges(self, filter: dict = {}) -> Iterable[PassageEdge]: ...
    def get_phase_edges(self, filter: dict = {}) -> Iterable[PhaseEdge]: ...
    def get_phase_nodes_by_fact_id(
        self, fact_id: str
    ) -> Iterable[tuple[PhaseNode, PhaseNode]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def run_ppr(
        self,
        personalization: dict[str, float],
        user_id: str,
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
