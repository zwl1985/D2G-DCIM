import statistics
from joblib import Parallel, delayed

import networkx as nx
import torch

import utils
from models import QNet
from agent import get_input_data

# ====================== 实验配置 ======================

GRAPH_NAME = 'Hypertext'
SEED_BUDGET = 50        # 我方种子数量 k
S_B_K = 10              # 竞争方种子数量
STEP = 10
NUM_WORKERS = 6
R = 10000
DEVICE = torch.device('cuda')

MODEL_PATH = 'qnet.pth'


# ====================== 图加载 ======================

def load_graph(name):
    path = f'test_graphs/{name}.txt'
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph)
    print("图信息：", G)
    return G


# ====================== 竞争方种子 ======================

def get_competitor_seeds(G, k):
    """按出度排序选择竞争方种子"""
    seeds = [
        node for node, _ in
        sorted(dict(G.out_degree()).items(),
               key=lambda x: x[1],
               reverse=True)[:k]
    ]
    return seeds


# ====================== QNet 选种 ======================

def select_seeds(qnet, G, s_b, k):

    data = get_input_data([G], 64, DEVICE)

    states = torch.zeros(G.number_of_nodes(),
                         dtype=torch.long,
                         device=DEVICE)

    states[s_b] = 2   # 竞争方状态标记

    S = []

    # 分批选择策略
    if GRAPH_NAME in ['Hypertext', 'High-School', 'mammalia', 'netscience']:
        strategy = [10] * (k // 10)
    else:
        strategy = [k]

    print("选种策略：", strategy)

    for num in strategy:
        scores = qnet(
            data.x,
            data.edge_index,
            data.t,
            data.w,
            states,
            data.batch
        )

        # 已选节点打极小值，防止重复选
        scores = scores + states * -1e8

        selected = scores.topk(num).indices.tolist()
        S.extend(selected)

        states[selected] = 1

    return S


# ====================== 传播评估 ======================

def evaluate_spread(G, seeds, dm, s_b,
                    step=1,
                    num_workers=8,
                    R=10000):

    results = []

    for i in range(step, len(seeds) + 1, step):
        partial = seeds[:i]

        spread = statistics.mean(
            Parallel(n_jobs=num_workers)(
                delayed(dm)(
                    G,
                    partial,
                    s_b,
                    R // num_workers
                )
                for _ in range(num_workers)
            )
        )

        results.append(spread)

    return results


# ====================== 主函数 ======================

def main():

    # 1. 加载图
    G = load_graph(GRAPH_NAME)

    # 2. 生成竞争方种子
    s_b = get_competitor_seeds(G, S_B_K)
    print("竞争方种子：", s_b)

    # 3. 加载模型
    qnet = QNet(64, 3).to(DEVICE)
    qnet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    qnet.eval()

    # 4. 选种
    seeds = select_seeds(qnet, G, s_b, SEED_BUDGET)
    print("我方种子：", seeds)

    # 5. 传播评估
    spread_list = evaluate_spread(
        G,
        seeds,
        utils.DCIC,
        s_b,
        step=STEP,
        num_workers=NUM_WORKERS,
        R=R
    )

    print("传播结果：", spread_list)


if __name__ == "__main__":
    main()