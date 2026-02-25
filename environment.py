import multiprocessing
import random
import statistics
import math

from joblib import Parallel, delayed

import utils


class GraphEnvironment:
    def __init__(self, graphs, k, gamma=0.99, n_steps=1, R=10000, num_workers=5, s_b_k=5):
        """
        G: networkx的图，Graph或DiGraph；
        k: 种子集大小；
        n_steps: 计算奖励时的步长；
        method: 计算奖励的方法；
        R: 使用蒙特卡洛估计奖励的轮数；
        num_workers: 使用多少个核心计算传播范围
        """
        self.graphs = graphs  # 子图列表
        self.k = k
        self.gamma = gamma
        self.n_steps = n_steps
        self.R = R
        self.num_workers = num_workers
        self.s_b_k = s_b_k
        self.graph = None  # 当前使用的子图
        # 当前状态，每个位置表示一个节点是否被选择，1是已选，0是未选
        self.state = None
        # 前一状态的奖励
        self.preview_reward = 0
        # 记录每次探索的状态、动作、奖励、下一步状态，以便计算n步奖励（为了学习得更好，n步可以更好反应真实的情况）
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

        self.seeds = []
        self.state_records = {}  # 记录种子集的奖励

    def reset(self):
        """
        重置环境。
        """
        self.graph = random.choice(self.graphs)  # 随机选一个子图
        self.s_b = [n for n, _ in sorted([(node, self.graph.out_degree(node)) for node in self.graph.nodes], key=lambda x: x[1], reverse=True)[: self.k]]
        self.seeds = []
        self.state = [2 if node in self.s_b else 0 for node in range(self.graph.number_of_nodes())] 
        self.preview_reward = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        return self.state

    def step(self, action):
        """
        根据所给的action，转移到新状态。
        """
        self.states.append(self.state.copy())
        self.state[action] = 1  # 更新状态
        self.seeds.append(action)
        # 计算奖励
        reward = self.compute_reward()

        done = False
        if len(self.seeds) == self.k:
            done = True

        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(self.state)
        return reward, self.state, done

    def compute_reward(self):
        """
        计算奖励，可以使用MC的方法或其他方法。
        """
        str_seeds = str(id(self.graph)) + str(sorted(self.seeds))
        if str_seeds in self.state_records:
            current_reward = self.state_records[str_seeds]
        else:
            results = Parallel(n_jobs=self.num_workers)(delayed(utils.DCIC)(self.graph, self.seeds, self.s_b, int(self.R / self.num_workers)) for _ in range(self.num_workers))
            current_reward = statistics.mean(results)
            
        r = max(0.0, current_reward - self.preview_reward)
        self.preview_reward = current_reward
        self.state_records[str_seeds] = current_reward
        return r


    def n_step_add_buffer(self, buffer):
        states = self.states
        rewards = self.rewards
        n = self.n_steps
        gamma = self.gamma
        
        for i in range(len(states) - n):
            done = (i + n) == (len(states) - 1)
            next_state = states[i + n]
            
            n_reward = sum(rewards[i + j] * (gamma ** j) for j in range(n))
            
            buffer.add(states[i], self.actions[i], n_reward, next_state, done, self.graph)  