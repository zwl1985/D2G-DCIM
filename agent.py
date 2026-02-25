import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.data import Data, Batch

import utils
from models import QNet


class Agent:
    def __init__(self, num_features, gamma, epsilon, lr, device, target_update=100, n_steps=1, tau=0.005, 
                 ntype='DQN', training=True):
        self.num_features = num_features
        # q网络
        self.q_net = QNet(num_features, R=3).to(device)
        self.q_net.apply(self.init_weights)

        # 目标q网络
        self.target_q_net = QNet(num_features, R=3).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.count = 0
        self.target_update = target_update
        self.n_steps = n_steps
        self.tau = tau
        self.ntype = ntype
        self.training = training

    @torch.no_grad()
    def take_action(self, state, env, max_len):
        selectable_nodes = list(set(env.graph.nodes()) - set(env.seeds) - set(env.s_b))

        if random.random() < self.epsilon:
            # 随机选择一个节点
            node = random.choice(selectable_nodes)
        else:
            # print('t')
            states = torch.tensor(state, dtype=torch.long, device=self.device)
            data = get_input_data([env.graph], self.num_features, self.device)

            # 设置网络为评估模式
            self.q_net.eval()

            # 计算所有节点的Q值
            q_values = self.q_net(data.x, data.edge_index, data.t, data.w, states, data.batch)

            # 只保留可选节点的Q值
            q_values_selectable = q_values[selectable_nodes]

            # 找到最大的Q值
            max_q_value, _ = q_values_selectable.max(0)

            # 找到所有等于最大Q值的节点索引
            max_indices = (q_values_selectable == max_q_value).nonzero(as_tuple=False).squeeze()

            # 如果有多个最大值，从中随机选择一个
            if max_indices.numel() == 1:  # 如果只有一个最大值
                max_index = max_indices.item()
            else:
                # 从多个最大值中随机选择一个
                max_index = max_indices[random.randint(0, len(max_indices) - 1)].item()

            # 转换索引回原始节点列表中的位置
            node = selectable_nodes[max_index]

        return node
    
    def soft_update(self, net, target_net):
        for param_traget,  param in zip(target_net.parameters(), net.parameters()):
            param_traget.data.copy_(param_traget.data * (1 - self.tau) + param.data * self.tau)

    def update(self, states, actions, rewards, next_states, graphs, dones, max_len):
        states = torch.tensor([s for state in states for s in state], dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        # print(rewards)
        next_states = torch.tensor([s for state in next_states for s in state], dtype=torch.long,
                                   device=self.device)
        dones = torch.tensor(dones, dtype=torch.int, device=self.device)

        data = get_input_data(graphs, self.num_features, self.device)

        # 首先找到每个batch的唯一值和对应的计数
        bidx, counts = torch.unique(data.batch, return_counts=True)

        self.q_net.train()
        actions_q_values = self.q_net(data.x, data.edge_index, data.t, data.w, states, data.batch)
        # 计算累积和以得到每个batch的偏移量
        offsets = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(counts[:-1], dim=0)])
        # 直接更新actions中的对应位置
        actions[bidx] += offsets

        q_values = actions_q_values.gather(dim=0, index=actions)
        
        self.target_q_net.eval()
        self.q_net.eval()
        with torch.no_grad():
            if self.ntype == 'DQN':
                max_q_values = torch_scatter.scatter_max(
                    self.target_q_net(data.x, data.edge_index, data.t, data.w, next_states,
                                      data.batch) + next_states * -1e8, data.batch)[0].clamp(min=0)
            elif self.ntype == 'DDQN':
                max_actions = torch_scatter.scatter_max(self.q_net(data.x, data.edge_index, data.t, data.w, next_states, data.batch) + next_states * -1e8, data.batch)[1]
                max_q_values = self.target_q_net(data.x, data.edge_index, data.t, data.w, next_states, data.batch).gather(dim=0, index=max_actions).clamp(min=0)
            else:
                raise ValueError(f'')
        
        q_targets = rewards + self.gamma ** self.n_steps * max_q_values 
        self.optim.zero_grad()
        loss = F.mse_loss(q_values, q_targets.detach()) 
        loss.backward()
        self.optim.step()

        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # self.soft_update(self.q_net, self.target_q_net)
        # self.target_q_net.load_state_dict(self.q_net.state_dict())
            
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01) 

def get_input_data(graphs, num_features, device):
    data_list = []
    for i, graph in enumerate(graphs):
        # 每个图中的节点的初始嵌入表示
        ew = torch.tensor(utils.get_time_graph_edge_info(graph))
        edge_index = ew[:, :2].long().T
        t = ew[:, 2:3].float()
        w = ew[:, 3:].float()
        x = torch.zeros(graph.number_of_nodes(), num_features, dtype=torch.float)
        data = Data(x, edge_index, w=w, t=t)
        data_list.append(data)

    batched_data = Batch.from_data_list(data_list)
    return batched_data.to(device)