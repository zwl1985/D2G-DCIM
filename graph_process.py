"""
读取真实时序图数据文件（u, v, t 三列格式），
将其转换为带时间戳和权重的时序图（temporal graph），
并保存为训练数据。
"""

import os
import argparse
import networkx as nx
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────
# 数据读取
# ──────────────────────────────────────────────

SUPPORTED_SEPARATORS = [' ', ',', '\t', ';', '|']


def detect_separator(filepath: str, candidate_seps: List[str] = SUPPORTED_SEPARATORS) -> str:
    """
    自动探测文件分隔符（读取前几行非注释行进行判断）
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        sample_lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sample_lines.append(line)
            if len(sample_lines) >= 5:
                break

    if not sample_lines:
        raise ValueError(f"文件 {filepath} 没有可读内容。")

    best_sep = ' '
    best_count = 0
    for sep in candidate_seps:
        counts = [len(line.split(sep)) for line in sample_lines]
        # 期望每行恰好分出 ≥3 列，且各行列数一致
        if min(counts) >= 3 and max(counts) == min(counts):
            if counts[0] > best_count:
                best_count = counts[0]
                best_sep = sep

    return best_sep


def load_temporal_edgelist(
    filepath: str,
    col_order: Tuple[str, str, str] = ('source', 'target', 't'),
    sep: Optional[str] = None,
    directed: bool = True,
    comment: str = '#'
) -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
    """
    读取真实时序图文件（u v t 三列），返回 MultiDiGraph 和整理后的 DataFrame。

    Args:
        filepath:  数据文件路径
        col_order: 列名顺序，对应文件中第 1、2、3 列的语义
        sep:       分隔符，若为 None 则自动探测
        directed:  是否作为有向图读取
        comment:   注释行前缀（跳过）

    Returns:
        (G, data)
        G    : nx.MultiDiGraph，节点编号已重映射为从 0 开始的整数
        data : 清洗并排序后的 DataFrame，列为 ['source', 'target', 't']
    """
    if sep is None:
        sep = detect_separator(filepath)
        print(f"  自动探测分隔符: {repr(sep)}")

    data = pd.read_csv(
        filepath,
        sep=sep,
        header=None,
        comment=comment,
        engine='python',
        on_bad_lines='skip'
    )

    # 只取前三列，重命名为 source / target / t
    data = data.iloc[:, :3].copy()
    data.columns = list(col_order)
    data = data[['source', 'target', 't']]

    # 类型转换
    data['source'] = pd.to_numeric(data['source'], errors='coerce')
    data['target'] = pd.to_numeric(data['target'], errors='coerce')
    data['t']      = pd.to_numeric(data['t'],      errors='coerce')
    data.dropna(inplace=True)
    data[['source', 'target']] = data[['source', 'target']].astype(int)

    # 构建 MultiGraph，去除自环，重映射节点编号
    graph_cls = nx.MultiDiGraph if directed else nx.MultiGraph
    G = nx.from_pandas_edgelist(data, edge_attr='t', create_using=graph_cls)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G, first_label=0)

    # 将整理后的边表与重映射后的 G 同步
    data = nx.to_pandas_edgelist(G)[['source', 'target', 't']]
    data = data.sort_values('t').reset_index(drop=True)

    print(f"  读取完成：{G.number_of_nodes()} 节点，{G.number_of_edges()} 条边，"
          f"时间跨度 [{data['t'].min()}, {data['t'].max()}]")

    return G, data


# ──────────────────────────────────────────────
# 快照图序列
# ──────────────────────────────────────────────

def build_snapshot_sequence(
    data: pd.DataFrame,
    directed: bool = True
) -> List[nx.DiGraph]:
    """
    按时间戳分组，为每个时间步构建一个快照图。

    Args:
        data:     包含 ['source', 'target', 't'] 的 DataFrame
        directed: 是否构建有向图

    Returns:
        List[nx.DiGraph]：按时间顺序排列的快照图列表
    """
    graph_cls = nx.DiGraph if directed else nx.Graph
    graph_list: List[nx.DiGraph] = []

    groups = list(data.groupby('t'))
    with tqdm(groups, desc="构建快照图序列") as bar:
        for _gid, group in bar:
            g = nx.from_pandas_edgelist(group, create_using=graph_cls)
            if not directed:
                g = g.to_directed()
            graph_list.append(g)

    return graph_list


# ──────────────────────────────────────────────
# 权重计算（与 generate_random_graphs.py 对齐）
# ──────────────────────────────────────────────

def compute_edge_probability(
    v_in_times: List[int],
    uv_times: List[int],
    t_current: int,
    zero_x: int = 10000
) -> float:
    """
    计算某条边在时间 t 的传播概率（二次函数衰减）。

    Args:
        v_in_times:  所有指向节点 v 的边的出现时间列表
        uv_times:    边 (u, v) 的出现时间列表
        t_current:   当前参考时间点
        zero_x:      衰减函数的零点参数

    Returns:
        float: 归一化后的传播概率，范围 [0, 1]
    """
    if not v_in_times:
        return 0.0

    def weight(dt: int) -> float:
        return (-1 / zero_x ** 2) * min(dt, zero_x - 0.001) ** 2 + 1

    total_v_weight  = sum(weight(t_current - vt) for vt in v_in_times)
    total_uv_weight = sum(weight(t_current - ut) for ut in uv_times)

    if total_v_weight <= 0:
        return 0.0

    prob = total_uv_weight / total_v_weight
    return max(0.0, min(prob, 1.0))


# ──────────────────────────────────────────────
# 时序图构建（与 generate_random_graphs.py 对齐）
# ──────────────────────────────────────────────

def graphs_to_temporal_graph(
    graph_sequence: List[nx.DiGraph],
    T: int = None
) -> nx.DiGraph:
    """
    将一系列快照图合并为一个带时间戳和权重的时序图。

    Args:
        graph_sequence: 图快照列表
        T:              时间范围（若为 None 则使用快照数量）

    Returns:
        nx.DiGraph：时序图，每条边带有 'times' 和 'weights' 属性
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
                temporal_g.edges[u, v]['times'].append(t)

    # 第二步：为每条边计算权重（使用最后时刻的归一化概率）
    for u, v in tqdm(temporal_g.edges(), desc="计算边权重"):
        v_in_times = [
            ts
            for pred in temporal_g.predecessors(v)
            for ts in temporal_g.edges[pred, v]['times']
        ]
        uv_times = temporal_g.edges[u, v]['times']

        if uv_times:
            p = compute_edge_probability(v_in_times, uv_times, T - 1, T)
            temporal_g.edges[u, v]['weights'] = [p] * len(uv_times)

    # 确保节点编号连续从 0 开始
    temporal_g = nx.convert_node_labels_to_integers(temporal_g, first_label=0)

    return temporal_g


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='将真实时序图文件转换为训练用时序图并保存。'
    )
    parser.add_argument('input',          type=str,            help='输入文件路径（u v t 三列）')
    parser.add_argument('--name',         type=str,  default=None, help='数据集名称（默认取文件名去后缀）')
    parser.add_argument('--sep',          type=str,  default=None, help='列分隔符（默认自动探测）')
    parser.add_argument('--directed',     action='store_true', default=True,  help='是否有向图（默认 True）')
    parser.add_argument('--undirected',   dest='directed', action='store_false', help='指定无向图')
    parser.add_argument('--out_dir',      type=str,  default='train_graphs', help='输出目录')

    parser.add_argument('--col_order',    type=str,  default='source,target,t',
                        help='列语义顺序，逗号分隔，默认 source,target,t')
    args = parser.parse_args()

    # 数据集名称
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    col_order = tuple(args.col_order.split(','))
    assert len(col_order) == 3, "--col_order 必须恰好包含三个逗号分隔的列名"

    print(f"\n{'='*50}")
    print(f"数据集: {name}  |  有向: {args.directed}")
    print(f"{'='*50}")

    # 1. 读取数据
    print("\n[1/4] 读取文件...")
    _G, data = load_temporal_edgelist(
        filepath=args.input,
        col_order=col_order,
        sep=args.sep,
        directed=args.directed
    )

    # 2. 构建快照序列
    print("\n[2/4] 构建快照图序列...")
    graph_list = build_snapshot_sequence(data, directed=args.directed)
    print(f"  共 {len(graph_list)} 个时间步快照")

    # 3. 转换为时序图
    print("\n[3/4] 合并为时序图并计算权重...")
    temporal_graph = graphs_to_temporal_graph(graph_list, T=len(graph_list))
    print(f"  时序图：{temporal_graph.number_of_nodes()} 节点，"
          f"{temporal_graph.number_of_edges()} 条边")

    # 4. 保存
    print("\n[4/4] 保存结果...")
    os.makedirs(args.out_dir, exist_ok=True)

    # 保存时序图边列表
    out_edge = os.path.join(args.out_dir, f'{name}.txt')
    nx.write_edgelist(temporal_graph, out_edge)
    print(f"  时序图边列表 -> {out_edge}")

print("\n完成！")


if __name__ == '__main__':
    main()
