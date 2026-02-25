"""
生成随机有向图序列，并将其转换为带时间戳的时序图（temporal graph）。
主要用于生成训练数据。
"""

import os
import random
import networkx as nx
from tqdm import tqdm
from typing import List

from utils import compute_edge_probability, graphs_to_temporal_graph


def generate_single_graph(n: int, directed: bool = True) -> nx.DiGraph:
    """
    生成单个随机有向图（目前使用 Erdős–Rényi 模型）
    """
    g = nx.erdos_renyi_graph(n, p=0.002, directed=directed)
        
    return g


def generate_graph_sequence(
    num_snapshots: int,
    n_nodes: int = None,
    directed: bool = True
) -> List[nx.DiGraph]:
    """
    生成一系列独立的快照图（目前每个快照独立生成，没有演化关系）
    
    Args:
        num_snapshots: 快照数量（时间步数）
        n_nodes: 节点数量（若为None则随机）
        directed: 是否生成有向图
    
    Returns:
        List[nx.DiGraph]: 时间序列上的图列表
    """
    if n_nodes is None:
        n_nodes = random.randint(50, 60)
        
    graph_sequence = []
    
    for _ in range(num_snapshots):
        g = generate_single_graph(n_nodes, directed=directed)
        graph_sequence.append(g)
        
    return graph_sequence


def compute_edge_probability(
    v_in_times: List[int],
    uv_times: List[int],
    t_current: int,
    zero_x: int = 10000
) -> float:
    """
    计算某条边在时间 t 的传播概率（使用二次函数衰减）
    
    当前实现：在最后一个时间点统一计算一个概率，然后应用到该边所有出现时刻
    """
    if not v_in_times:
        return 0.0
    
    # v 的所有入边在 t 时刻的权重总和
    total_v_weight = sum(
        (-1 / zero_x**2) * min(t_current - vt, zero_x - 0.001)**2 + 1
        for vt in v_in_times
    )
    
    # 该条边 (u,v) 的出现时间点的权重总和
    total_uv_weight = sum(
        (-1 / zero_x**2) * min(t_current - vt, zero_x - 0.001)**2 + 1
        for vt in uv_times
    )
    
    if total_v_weight <= 0:
        return 0.0
        
    prob = total_uv_weight / total_v_weight
    return max(0.0, min(prob, 1.0))  # 可根据需要调整上下界


def graphs_to_temporal_graph(
    graph_sequence: List[nx.DiGraph],
    T: int = None
) -> nx.DiGraph:
    """
    将一系列快照图合并为一个带时间和权重的时序图
    
    Args:
        graph_sequence: 图快照列表
        T: 时间范围（若为None则使用快照数量）
    
    Returns:
        nx.DiGraph: 时序图，每条边带有 'times' 和 'weights' 属性
    """
    if T is None:
        T = len(graph_sequence)
    
    temporal_g = nx.DiGraph()
    
    # 第一步：收集所有边的出现时间
    for t, g in enumerate(tqdm(graph_sequence, desc="收集边的时间戳")):
        for u, v in g.edges():
            if not temporal_g.has_edge(u, v):
                temporal_g.add_edge(u, v, times=[t], weights=[])
            else:
                temporal_g.edges[u, v]["times"].append(t)
    
    # 第二步：为每条边的每个出现时刻计算权重（当前实现：统一用最后时刻概率）
    for u, v in tqdm(temporal_g.edges(), desc="计算边权重"):
        # 收集所有指向 v 的边的出现时间
        v_in_times = [
            ts for pred in temporal_g.predecessors(v)
            for ts in temporal_g.edges[pred, v]["times"]
        ]
        
        uv_times = temporal_g.edges[u, v]["times"]
        
        if uv_times:
            # 使用最后一个时间点（T-1）来计算统一的概率
            p = compute_edge_probability(v_in_times, uv_times, T - 1, T)
            temporal_g.edges[u, v]["weights"] = [p] * len(uv_times)
    
    # 确保节点编号连续从 0 开始
    temporal_g = nx.convert_node_labels_to_integers(temporal_g, first_label=0)
    
    return temporal_g


def main():
    num_temporal_graphs = 10
    num_snapshots_range = (100, 200)          # 时间步数,快照数量
    n_nodes_range = (50, 100)     # 节点数随机范围
    
    for i in range(num_temporal_graphs):  
        print(f"正在生成第 {i+1} 个图序列...")
        graph_seq = generate_graph_sequence(
            num_snapshots=random.randint(*num_snapshots_range),
            n_nodes=random.randint(*n_nodes_range),
            directed=True
        )
        
        print("正在转换为时序图...")
        temporal_graph = graphs_to_temporal_graph(graph_seq, T=len(graph_seq))
        
        print(f"完成！节点数: {temporal_graph.number_of_nodes()}，"
              f"边数: {temporal_graph.number_of_edges()}")
        
        os.makedirs('train_graphs', exist_ok=True)
        
        # 保存为边列表（带时间和权重）
        nx.write_edgelist(temporal_graph, f"train_graphs/graph{i}.txt")


if __name__ == "__main__":
    main()