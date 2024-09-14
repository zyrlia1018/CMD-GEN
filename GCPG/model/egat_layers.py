import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import EGATConv

# 定义EGAT模型
class EGATEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, out_dim, n_layers, num_heads, dropout, batch_norm):
        super(EGATEncoderBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # 输入层
        # 隐藏层
        for l in range(n_layers):
            self.layers.append(EGATConv(in_node_feats=hidden_dim,
                                        in_edge_feats=hidden_dim,
                                        out_node_feats=out_dim,
                                        out_edge_feats=out_dim,
                                        num_heads=num_heads))
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Dropout层
        self.dropout = nn.Dropout(p=dropout)

        self.n_layers = n_layers

    def forward(self, graph, node_feat, edge_feat):
        #h = graph.ndata['feat']
        init = node_feat.clone()
        for i in range(self.n_layers):
            node_feat, edge_feat = self.layers[i](graph, node_feat, edge_feat)    #如歌此处不输出边的特征，那我们在前面边的特征那就可以不变化。由于已知dp和bn边的特征对拟合的效果用处不大
            node_feat = torch.mean(node_feat, dim=1, keepdim=False)
            edge_feat = torch.mean(edge_feat, dim=1, keepdim=False)
            # 在隐藏层之间加入dropout和batchnorm1d
            node_feat = self.dropout(node_feat)
            # edge_feat = self.dropout(edge_feat)   #边的特征不要进行dropout和batchnorm1d会让结果更好
            node_feat = self.bns[i](node_feat)
            # edge_feat = self.bns[i](e dge_feat)
        # 最后一层不需要激活函数
        node_feat,_ = self.layers[-1](graph, node_feat,edge_feat)
        node_feat = torch.mean(node_feat, dim=1, keepdim=False)
        #残差连接
        node_feat = node_feat + init
        # 读出层
        return node_feat

