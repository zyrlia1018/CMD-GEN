import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GINEConv


class GINEEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, out_dim, n_layers, dropout, batch_norm):
        super(GINEEncoderBlock, self).__init__()
        # 输入层
        self.linear_edge = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 隐藏层
        for l in range(n_layers):
            self.linear_edge.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
            self.layers.append(GINEConv(apply_func=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU())))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(out_dim))


        # Dropout层
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers

    def forward(self, graph, node_feat, edge_feat):
        #h = graph.ndata['feat']
        init = node_feat.clone()
        for i in range(self.n_layers):
            node_feat = self.layers[i](graph, node_feat, edge_feat)
            edge_feat = self.linear_edge[i](edge_feat)
            # 在隐藏层之间加入 dropout 和 batchnorm1d
            node_feat = self.dropout(node_feat)
            node_feat = self.bns[i](node_feat)
        # 最后一层不需要激活函数
        node_feat = self.layers[-1](graph, node_feat, edge_feat)
        # 残差连接
        node_feat = node_feat + init

        return node_feat


