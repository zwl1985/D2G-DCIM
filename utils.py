import random
import collections
from typing import List

import networkx as nx
import torch
import numpy as np
from torch_geometric.utils import coalesce

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
    
    # 第二步：为每条边的每个出现时刻计算权重
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


def DCIC(G, s_a, s_b, R=1000):  
    """  
    模拟时序图上的竞争的独立级联模型。  
      
    参数:  
    G (networkx.Graph): 包含节点和边的图，边包含时间 't' 和权重 'w'。  
    s_a (set): 初始激活节点集合 A。  
    s_b (set): 初始激活节点集合 B。  
    R (int): 独立模拟的次数。  
      
    返回:  
    float: 平均激活节点数。  
    """  
    ds = 0  # 总的激活节点数  
  
    for _ in range(R):  # 重复R次独立模拟  
        a_activated_nodes = set(s_a)  # 集合A的已激活节点  
        a_current_activations = set(s_a)  # 集合A当前正在尝试激活的节点  
        b_activated_nodes = set(s_b)  # 集合B的已激活节点  
        b_current_activations = set(s_b)  # 集合B当前正在尝试激活的节点  
        at = [0 if node in (a_activated_nodes | b_activated_nodes) else -1 for node in range(G.number_of_nodes())]
  
        while True:  
            a_new_activations = set()  
            b_new_activations = set()  
  
            # 集合A尝试激活新节点  
            for u in a_current_activations:  
                for v in G.neighbors(u):  
                    if v in a_activated_nodes or v in b_activated_nodes or at[u] > max(G.edges[u, v]['t']):  
                        continue  
                    index, min_at = min([(i, t) for i, t in enumerate(G.edges[u, v]['t']) if at[u] <= t], key=lambda x: x[1])
                    p = G.edges[u, v]['w'][index]
                    if random.random() <= p:  
                        a_new_activations.add(v)  
                        at[v] = min_at if at[v] == -1 else min(at[v], min_at)
  
            # 集合B尝试激活新节点  
            for u in b_current_activations:  
                for v in G.neighbors(u):  
                    if v in a_activated_nodes or v in b_activated_nodes or at[u] > max(G.edges[u, v]['t']):  
                        continue  
                    
                    index, min_at = min([(i, t) for i, t in enumerate(G.edges[u, v]['t']) if at[u] <= t], key=lambda x: x[1])
                    p = G.edges[u, v]['w'][index]
                    b_able_try_activate_v = (v in a_new_activations and min_at == at[v])
                        
                    if random.random() <= p:
                        if b_able_try_activate_v:
                            if random.random() <= 0.5:
                                a_new_activations.remove(v)
                                b_new_activations.add(v)
                        else:
                            b_new_activations.add(v)
                            at[v] = min_at if at[v] == -1 else min(at[v], min_at)                     
  
            # 更新集合  
            a_activated_nodes.update(a_new_activations)  
            a_current_activations = a_new_activations  
            b_activated_nodes.update(b_new_activations)  
            b_current_activations = b_new_activations  
  
            # 如果没有新的节点被激活，结束循环  
            if not a_new_activations and not b_new_activations:  
                break  
  
        ds += len(a_activated_nodes)  # 累加每次模拟的总激活节点数  
  
    return ds / R  # 返回平均激活节点数  

def edge_process(edge_index, w, t, at, aggr_direction):
    if aggr_direction == 'out':
        mask = t.view(-1) <= at[edge_index[1]]
        reduce='min'
    elif aggr_direction == 'in':
        mask = t.view(-1) >= at[edge_index[1]]
        reduce='min'
    else:
        raise ValueError
    edge_index_selected = edge_index[:, mask]
    edge_index_coalesce, attr = coalesce(edge_index_selected, torch.cat([w, t], dim=1), at.shape[0], reduce=reduce)
    w_coalesce = attr[:, 0:1]
    t_coalesce = attr[:, 1]
    return edge_index_coalesce, w_coalesce, t_coalesce


def remove_nodes(edge_index, edge_attr, nodes_to_remove):
    """
    从图中移除指定的节点及其相关的边。

    参数:
    - edge_index (torch.Tensor): 形状为 [2, E] 的张量，表示边的索引。
                                 第一行包含源节点索引，第二行包含目标节点索引。
    - edge_attr (torch.Tensor): 形状为 [E, F] 的张量，表示边的属性。
    - nodes_to_remove (torch.Tensor): 形状为 [N_remove] 的张量，包含要移除的节点索引。

    返回:
    - filtered_edge_index (torch.Tensor): 移除指定节点后的边索引。
    - filtered_edge_attr (torch.Tensor): 移除指定节点后的边属性。
    """
    
    # 检查输入类型是否为tensor
    if not isinstance(nodes_to_remove, torch.Tensor):
        raise TypeError(f'nodes_to_remove的类型必须是tensor。')
    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(f'edge_index的类型必须是tensor。')
    if not isinstance(edge_attr, torch.Tensor):
        raise TypeError(f'edge_index的类型必须是tensor。')
    
    # 如果要移除的节点集合为空，则直接返回原始边索引和边属性
    if nodes_to_remove.size(0) == 0:
        return edge_index, edge_attr
      
    # 计算哪些边不在要移除的节点集合中
    # 对边的第一个端点（源节点）进行过滤，得到不在移除列表中的边的索引掩码
    mask_i = ~torch.isin(edge_index[0], nodes_to_remove)
    # 对边的第二个端点（目标节点）进行过滤，得到不在移除列表中的边的索引掩码
    mask_j = ~torch.isin(edge_index[1], nodes_to_remove)
      
    # 使用逻辑与操作组合两个掩码，得到同时满足两个条件的边的索引
    # 即边的两个端点都不在移除节点集合中的边
    combined_mask = mask_i * mask_j
      
    # 根据组合后的掩码选择保留的边索引和边属性
    filtered_edge_index = edge_index[:, combined_mask]
    filtered_edge_attr = edge_attr[combined_mask]
      
    # 返回移除指定节点后的边索引和边属性
    return filtered_edge_index, filtered_edge_attr

def get_time_graph_edge_info(graph):
    """
        获取图的edge_index、edge_weight和t
    """
    
    return [(u, v, graph.edges[u, v]['t'][i] + 1, graph.edges[u, v]['w'][i]) for u, v, a in graph.edges(data=True) for i in range(len(graph.edges[u, v]['t']))]


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, graph):
        self.buffer.append((state, action, reward, next_state, done, graph))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, graph = zip(*transitions)
        return state, action, reward, next_state, done, graph

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)
  
# 设置随机种子  
def set_seed(seed=42):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False   


if __name__ == '__main__':
    pass
