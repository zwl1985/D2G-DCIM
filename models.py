import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_scatter
from torch_scatter import scatter

import utils



class AdditiveAttention(nn.Module):
    def __init__(self, num_features, dropout=0.1):
        super(AdditiveAttention, self).__init__()
        self.ws = nn.Linear(num_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        scores = self.ws(torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1)))
        attention_weights = F.softmax(scores, dim=-1).squeeze(-1)
        new_values = attention_weights.bmm(values)
        return new_values


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_heads, num_hiddens, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.wq = nn.Linear(query_size, num_hiddens, bias=False)
        self.wk = nn.Linear(key_size, num_hiddens, bias=False)
        self.wv = nn.Linear(value_size, num_hiddens, bias=False)
        self.attention = AdditiveAttention(self.num_hiddens // self.num_heads, dropout)
        self.wo = nn.Linear(num_hiddens, value_size, bias=True)

    def forward(self, queries, keys, values):
        queries, keys, values = self.wq(queries), self.wk(keys), self.wv(values)

        queries = queries.view(queries.shape[0], queries.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        queries = queries.reshape(-1, queries.shape[2], queries.shape[3])

        keys = keys.view(keys.shape[0], keys.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        keys = keys.reshape(-1, keys.shape[2], keys.shape[3])

        values = values.view(values.shape[0], values.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        values = values.reshape(-1, values.shape[2], values.shape[3])
        # 恢复形状
        out = self.attention(queries, keys, values)
        out = out.view(-1, self.num_heads, out.shape[1], out.shape[2]).permute(0, 2, 1, 3)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.wo(out)
        return out


activate_function = nn.LeakyReLU()

class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, affine=True):
        """
        图归一化层。

        参数:
        - num_features (int): 输入特征的维度。
        - affine (bool): 是否应用可学习的仿射变换（即是否使用 gamma 和 beta）。
        - eps (float): 用于避免除以零的小常数。
        """
        super(GraphNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, batch, batch_num):
        # 图级归一化 
        mu = torch_scatter.scatter_mean(x, batch, dim=0).repeat_interleave(batch_num, dim=0)
        sigma = torch_scatter.scatter_std(x, batch, dim=0).repeat_interleave(batch_num, dim=0)
        if self.affine:
            x = (x - mu) / (sigma + self.eps) * self.gamma + self.beta
        else:
            x = (x - mu) / (sigma + self.eps)
        return x
    
    def __repr__(self):
        return f'{self.__class__.__name__}(num_features={self.num_features}, eps={self.eps}, affine={self.affine})'

class InfGNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, add_bias=False, mlp_bias=False, norm=True):
        super(InfGNNConv, self).__init__()
        self.add_bias = add_bias
        self.norm = norm
        
        self.w_s = nn.Linear(in_channels, in_channels, bias=False)
        
        self.mlp_x = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=mlp_bias),
            activate_function,
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels, bias=mlp_bias),
        )

        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        
        if self.norm:
            self.graph_norm = GraphNorm(out_channels, affine=True)
        
    def forward(self, x, edge_index, w, states, batch, batch_num):
        x = self.mlp_x(activate_function(x + self.w_s(states)))

        x_j = x[edge_index[1]]
        x_j = w * x_j
        
        x = torch_scatter.scatter_sum(x_j, edge_index[0], dim=0, dim_size=x.shape[0])
        
        if self.add_bias:
            x = x + self.bias
            
        if self.norm:
            x = self.graph_norm(x, batch, batch_num)
        return x
    
class StateGNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, add_bias=False, mlp_bias=False, norm=True):
        super(StateGNNConv, self).__init__()
        self.add_bias = add_bias
        self.norm = norm
        
        self.mlp_x = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=mlp_bias),
            activate_function,
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels, bias=mlp_bias),
        )
        
        self.w_o = nn.Linear(in_channels, out_channels, bias=False)
        self.w_n = nn.Linear(out_channels, out_channels, bias=False)
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        
        if self.norm:
            self.graph_norm = GraphNorm(out_channels, affine=True)
        
    def forward(self, x, edge_index, w, batch, batch_num):
        x = self.mlp_x(x)
        x_j = x[edge_index[1]]
        x_j = w * x_j

        x_n = activate_function(torch_scatter.scatter_sum(x_j, edge_index[0], dim=0, dim_size=x.shape[0]))
        
        x = self.w_o(x) + self.w_n(x_n)
        if self.add_bias:
            x = x + self.bias
            
        if self.norm:
            x = self.graph_norm(x, batch, batch_num)
        return x
        
        
class GNNBlock(nn.Module):
    def __init__(self, num_features, R=3, dropout=0.1, add_bias=False, mlp_bias=False):
        super(GNNBlock, self).__init__()
        self.dropout = dropout
        self.R = R
        
        self.w_a = nn.Linear(1, num_features, bias=False)
        self.w_b = nn.Linear(1, num_features, bias=False)
        self.state_convs = nn.ModuleList()
        self.inf_convs = nn.ModuleList()

        for _ in range(R):
            self.state_convs.append(StateGNNConv(num_features, num_features, dropout=dropout, add_bias=True, mlp_bias=True, norm=True))
            self.inf_convs.append(InfGNNConv(num_features, num_features, dropout=dropout, add_bias=True, mlp_bias=True, norm=True))
        
        self.attention = MultiHeadAttention(num_features, num_features, num_features, 4, num_features, 0.0)

    def forward(self, x, edge_index, t, w, s_a, s_b, batch, batch_num):
        x_list = []

        at_in = torch.ones(x.shape[0], device=x.device)
        at_out = float('inf') * torch.ones(x.shape[0], device=x.device)
        
        nodes_to_remove = (s_a + s_b).view(-1).nonzero().view(-1)
        edge_index_remain, edge_attr_remain = utils.remove_nodes(edge_index, torch.cat([w, t], dim=1), nodes_to_remove)
        w_remain = edge_attr_remain[:, 0:1]
        t_remain = edge_attr_remain[:, 1:]
        
        states = activate_function(self.w_a(s_a) + self.w_b(s_b))
        
        convs = list(zip(self.state_convs, self.inf_convs)) 
        
        for i in range(self.R):
            state_conv, inf_conv = convs[i][0], convs[i][1]
            
            edge_index_coalesce_in, w_coalesce_in, t_coalesce_in = utils.edge_process(edge_index[[1, 0]], w, t, at_in, 'in')
            edge_index_coalesce_out, w_coalesce_out, t_coalesce_out = utils.edge_process(edge_index_remain, w_remain, t_remain, at_out, 'out')
            
            states = state_conv(states, edge_index_coalesce_in, w_coalesce_in, batch, batch_num)
            states = activate_function(states)
            
            x = inf_conv(x, edge_index_coalesce_out, w_coalesce_out, states, batch, batch_num) 
            x = activate_function(x)
            x_list.append(x)
            
            if i < self.R - 1:
                # 更新at
                at_in = torch_scatter.scatter_min(t_coalesce_in, edge_index_coalesce_in[0], dim=0, dim_size=x.shape[0])[0]
                at_out = torch_scatter.scatter_max(t_coalesce_out, edge_index_coalesce_out[0], dim=0, dim_size=x.shape[0])[0]
        
        x = torch.cat(x_list, dim=1).view(x.shape[0], -1, x.shape[1])
        x = self.attention(x, x, x)
        x = x.sum(1)

        return x 
    

class QNet(nn.Module):
    def __init__(self, num_features, R=3, aggr_high_order=False, dropout=0.0):
        super(QNet, self).__init__()        
        self.conv_block = GNNBlock(num_features, R, dropout, add_bias=True, mlp_bias=True)
        self.lin_conv = nn.Linear(num_features, num_features, bias=True)

        self.w_x = nn.Linear(num_features, num_features, bias=False)
        self.ln_g = nn.LayerNorm([num_features])
        self.w_g = nn.Linear(num_features, num_features, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_features, 1 * num_features),
            activate_function,
            nn.Dropout(dropout),
            nn.Linear(1 * num_features, 1)
        )


    def forward(self, x, edge_index, t, w, states, batch):
        batch_num = torch_scatter.scatter_sum(torch.ones(batch.shape[0], dtype=torch.long, device=batch.device), batch)
        
        s_a = (states == 1).unsqueeze(1).float()
        s_b = (states == 2).unsqueeze(1).float()

        x = self.lin_conv(activate_function(self.conv_block(x, edge_index, t, w, s_a, s_b, batch, batch_num)))
        x = activate_function(x)
        
        # 图级表示
        x_g = torch_scatter.scatter_sum(x, batch, dim=0, dim_size=batch[-1].item() + 1)
        x_g = self.ln_g(x_g)
        x_g = self.w_g(x_g).repeat_interleave(batch_num, dim=0)
        
        x = self.w_x(x)
        x = torch.cat([x, x_g], dim=-1)
        
        q = self.mlp(x)
        return q.view(-1)
