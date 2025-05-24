from typing import Protocol, Iterable, TypedDict
from dataclasses import dataclass


@dataclass
class BaseNode:
    id: str
    ts: int
    """Seconds"""


@dataclass
class PassageNode(BaseNode):
    """
    记忆节点/段落节点
    从文本中提取出的段落或摘要，用于存储 LLM 的记忆内容
    """

    user_id: str


@dataclass
class PhaseNode(BaseNode):
    """
    概念节点/实体节点
    从文本中提取出的实体，例如人、地点、组织或概念, 代表了 LLM 记忆中的关键概念或知识点
    """

    name: str
    type: str


@dataclass
class BaseEdge:
    """
    基础边类
    图中边的基本属性，包括源节点、目标节点、时间戳、关系(边)类型和用户ID
    """

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
    """
    记忆关联边/段落关联边
    实体和段落之间的关系，用于连接 PhaseNode(概念节点) 和 PassageNode(记忆节点)
    在图数据库中, 表现为边的形式
    """

    summary_id: str
    """Summary ID"""


@dataclass
class PhaseEdge(BaseEdge):
    """
    概念关系边/实体关系边
    表示实体之间的关系，用于连接两个 PhaseNode(概念节点)
    在图数据库中, 表现为边的形式
    """

    fact_id: str
    """Fact ID"""


class GraphResult(TypedDict):
    nodes: list
    edges: list


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
    def delete_phase_edge_by_fact_id(self, fact_id: str) -> None: ...
    def cnt_phase_node_edges(self, node_id: str) -> int: ...
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

    def get_graph_networkx(self, filter: dict = None) -> GraphResult:
        """获取图的 NetworkX 表示"""
        ...
