import argparse
import os
import sys
from datetime import datetime
import time

import networkx as nx
import torch
from tqdm import tqdm

import utils
from agent import Agent
from environment import GraphEnvironment

utils.set_seed(42)  # 设置随机种子


def explore(env, agent, eps, replay_buffer, num_episodes, train=True, show_bar=True):
    """
    在给定环境中使用给定的代理进行探索。

    参数:
    env (object): 环境对象，用于与代理交互。
    agent (object): 代理对象，包含策略网络、epsilon值等。
    eps (float): 当前轮次的epsilon值，用于平衡探索和利用。
    replay_buffer (object): 经验回放缓冲区，用于存储探索过程中产生的数据。
    num_episodes (int): 要探索的序列数量。
    train (bool, optional): 是否处于训练模式，默认为True。在训练模式下，数据将被添加到回放缓冲区。
    show_bar (bool, optional): 是否显示进度条，默认为True。

    返回:
    如果train为False，则返回float类型的episode_return，即测试过程中的累积奖励；
    否则不返回任何值（None）。
    """

    agent.epsilon = eps  # 设置代理的epsilon值

    if train:
        if show_bar:
            bar = tqdm(total=num_episodes, desc=f'epsilon={eps}时探索{num_episodes}条序列')  # 初始化进度条

        for _ in range(num_episodes):
            state = env.reset()  # 重置环境，获取初始状态
            done = False
            episode_return = 0

            while not done:
                action = agent.take_action(state, env, 0)  # 根据当前状态和epsilon值选择动作
                reward, next_state, done = env.step(action)  # 执行动作，获取奖励和下一个状态
                episode_return += reward  # 累积奖励
                state = next_state  # 更新状态

            # 探索序列结束后，将n步奖励添加到经验回放缓冲区中
            env.n_step_add_buffer(replay_buffer)

            if show_bar:
                bar.update(1)  # 更新进度条

        if show_bar:
            bar.close()  # 关闭进度条

    else:
        for _ in range(num_episodes):
            state = env.reset()  # 重置环境，获取初始状态
            done = False
            episode_return = 0

            while not done:
                action = agent.take_action(state, env, 0)  # 根据当前状态选择动作
                reward, next_state, done = env.step(action)  # 执行动作，获取奖励和下一个状态
                episode_return += reward  # 累积奖励
                state = next_state  # 更新状态

        return episode_return  # 返回测试过程中的累积奖励


def train(agent, num_epochs, train_graphs, env, test_env, replay_buffer, batch_size, folder_path):
    """
    训练强化学习代理（agent）在给定的图集上进行。

    参数:
    agent (Agent): 代理对象，包含Q网络等训练所需组件。
    num_epochs (int): 训练的总轮数。
    train_graphs (list): 用于训练的子图列表。
    env (object): 训练环境对象。
    test_env (GraphEnvironment): 测试环境对象。
    replay_buffer (ReplayBuffer): 经验回放缓冲区对象，用于存储训练数据。
    batch_size (int): 每次从回放缓冲区中采样的批量大小。
    folder_path (str): 用于保存Q网络参数的文件夹路径。

    返回:
    无返回值，但会更新代理（agent）的Q网络参数，并保存部分中间结果。
    """

    # 初始探索阶段，填充经验回放缓冲区
    explore(env, agent, 1, replay_buffer, 1)

    # 设置探索率的起始和结束值
    eps_start = 1
    eps_end = 0.05

    # 使用tqdm显示进度条
    tbar = tqdm(total=num_epochs, desc=f'使用{len(train_graphs)}个子图训练')

    for i in range(num_epochs):
        # 计算当前轮次的探索率 eps
        eps = eps_end + max(0.0, (eps_start - eps_end) * (num_epochs // 2 - i) / (num_epochs // 2))
        # eps = 0.05

        if i % 10 == 0:
            # 每10轮保存一次Q网络的参数
            
            # 使用当前探索率 eps 进行探索，并填充经验回放缓冲区
            explore(env, agent, eps, replay_buffer, 1, True, False)
            
            # 在测试环境上进行测试，并获取奖励
            rewards = explore(test_env, agent, 0, None, 1, False, False)
            if rewards >= 1:
                torch.save(agent.q_net.state_dict(), f'{folder_path}/q_net_{i // 10}.pth')

            # 在进度条上更新当前轮次的信息
            tbar.write(
                f'{i}/{num_epochs}：种子集：{test_env.seeds}，奖励：{rewards:.3f}，回放缓冲区长度：{len(replay_buffer)}',
                file=sys.stdout)

            # 从经验回放缓冲区中采样数据，并更新Q网络
        states, actions, rewards, next_states, dones, graphs = replay_buffer.sample(batch_size)
        agent.update(states, actions, rewards, next_states, graphs, dones, 0)

        # 更新进度条
        tbar.update(1)


def main(args):
    t_start = time.time()
    lr = args.lr
    gamma = args.gamma
    epsilon = 1
    target_update = args.target_update
    buffer_size = args.buffer_size
    batch_size = 64
    num_epochs = args.num_epochs
    num_features = args.num_features
    k = args.k
    n_steps = args.n_steps
    tau = args.tau
    test_graph_name = args.test_graph_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replay_buffer = utils.ReplayBuffer(buffer_size)

    agent = Agent(num_features, gamma, epsilon, lr, device, target_update, n_steps, tau, ntype='DDQN')
    print(agent.q_net)

    # 创建保存模型的文件夹
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")  
    # 设定文件夹路径
    pre_file_path = os.getcwd()
    folder_name = f"{time_str}"
    folder_path = os.path.join(pre_file_path, 'my_model', f'{args.dmodel}', folder_name)  
    # 如果文件夹不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 训练
    g_folder_path = f'{pre_file_path}/train_graphs'
    filenames = os.listdir(g_folder_path)
    # 打印文件的完整路径
    train_graphs = []
    test_graphs = []
    for filename in filenames:
        if filename == '.ipynb_checkpoints':
            continue
        file_path = os.path.join(g_folder_path, filename)
        g = nx.read_edgelist(file_path, nodetype=int, create_using=nx.DiGraph, data=True)
        train_graphs.append(g)
        print(g)
        if len(train_graphs) == 10:
            break
    
    g = nx.read_edgelist(f'{pre_file_path}/test_graphs/{test_graph_name}.txt', nodetype=int, create_using=nx.DiGraph, data=True)
    print(g)
    test_graphs.append(g)
    env = GraphEnvironment(train_graphs, k, gamma, n_steps, R=10000, num_workers=6)
    # 测试环境
    test_env = GraphEnvironment(test_graphs, 5, gamma, n_steps, R=10000, num_workers=6)

    # 训练
    train(agent, num_epochs, train_graphs, env, test_env, replay_buffer, batch_size, folder_path)
    
    t_end = time.time()
    
    print(f'{t_start}, {t_end}, 总训练时间：{t_end-t_start} s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练所需参数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率。")
    parser.add_argument("--k", type=int, default=5, help="选择的种子节点的个数。")
    parser.add_argument("--n_steps", type=int, default=1, help="步长。")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子。")
    parser.add_argument("--target_update", type=int, default=50, help="目标网络更新的频率。")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络软更新参数。")
    parser.add_argument("--buffer_size", type=int, default=30000, help="经验回放池的大小。")
    parser.add_argument("--num_features", type=int, default=64, help="特征维度。")
    parser.add_argument("--num_epochs", type=int, default=2000, help="训练的轮数。")
    parser.add_argument("--R", type=int, default=1000, help="传播的次数。")
    parser.add_argument("--dmodel", type=str, default='DCIC', help="传播模型。")
    parser.add_argument("--lambd", type=float, default=0.05, help="SI模型的感染率。")
    parser.add_argument("--test_graph_name", type=str, default='Hypertext', help="测试图名称。")
    args = parser.parse_args()

    main(args)
