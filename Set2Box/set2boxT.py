import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_scatter
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange

import utils

EPS = 1e-10
MAX = 1e6
MIN = -1e6

class model(nn.Module):
    def __init__(self, num_item, hidden_dim, beta, attention='scp'):
        super(model, self).__init__()
        self.beta = beta
        self.dim = hidden_dim
        self.num_item = num_item
        
        if attention == 'scp':
            self.pool = self.attention_scp
        
        self.center_attention = nn.Parameter(torch.empty(self.dim))
        self.radius_attention = nn.Parameter(torch.empty(self.dim))
        
        self.center_embedding = nn.Embedding(self.num_item, self.dim)
        self.radius_embedding = nn.Embedding(self.num_item, self.dim)
        
        self.clip_max = torch.FloatTensor([1.0])
        self.init_weights()
        self.radius_embedding.weight.data = self.radius_embedding.weight.data.clamp(min=EPS)
        self.loss = nn.MSELoss(reduction='sum')

    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.radius_embedding.weight, mean=0.0, std=0.1)
        self.center_embedding.weight.data.div_(torch.max(torch.norm(self.center_embedding.weight.data, 2, 1, True),
                                                         self.clip_max).expand_as(self.center_embedding.weight.data))
        self.radius_embedding.weight.data.div_(torch.max(torch.norm(self.radius_embedding.weight.data, 2, 1, True),
                                                         self.clip_max).expand_as(self.radius_embedding.weight.data))
        
        stdv = 1. / math.sqrt(self.dim)
        nn.init.uniform_(self.center_attention, -stdv, stdv)
        nn.init.uniform_(self.radius_attention, -stdv, stdv)
    
    def embed_set(self, S, M, instances=None):
        if instances is not None:
            batch_sets, _instances = torch.unique(instances, return_inverse=True)
            S_batch = S[batch_sets]
            M_batch = M[batch_sets]
            emb_center = self.pool(S_batch, M_batch, self.center_embedding.weight, self.center_attention, False)
            emb_radius = self.pool(S_batch, M_batch, self.radius_embedding.weight, self.radius_attention, True)
            emb_center = emb_center[_instances]
            emb_radius = emb_radius[_instances]
        else:
            emb_center = self.pool(S, M, self.center_embedding.weight, self.center_attention, False)
            emb_radius = self.pool(S, M, self.radius_embedding.weight, self.radius_attention, True)
        return emb_center, emb_radius  
    
    def forward(self, S, M, instances, overlaps):
        emb_center, emb_radius = self.embed_set(S, M, instances)
        
        c_i, c_j = emb_center[:, 0, :], emb_center[:, 1, :]
        r_i, r_j = emb_radius[:, 0, :], emb_radius[:, 1, :]

        # 2 Box instances: Box_i, Box_j
        # m_i, m_j = c_i - r_i, c_j - r_j
        # M_i, M_j = c_i + r_i, c_j + r_j

        # 2 boxes: box_i, box_j;
        # Box Edge length: M_i > m_i, M_j > m_j
        m_i = f.softplus(c_i, self.beta)
        box_edge_i = f.softplus(r_i, self.beta)
        M_i = m_i + box_edge_i

        m_j = f.softplus(c_j, self.beta)
        box_edge_j = f.softplus(r_j, self.beta)
        M_j = m_j + box_edge_j

        box_edge_delta = torch.min(M_i, M_j) - torch.max(m_i, m_j)

        # Box Volume
        box_volume_i = torch.sum(torch.log(box_edge_i + EPS), 1)
        box_volume_j = torch.sum(torch.log(box_edge_j + EPS), 1)
        # Intersection and Union
        box_inter = torch.sum(torch.log(box_edge_delta + EPS), 1)
        box_union = torch.sum(torch.log(torch.max(M_i, M_j) - torch.min(m_i, m_j) + EPS), 1)
        # Costs
        c_overlap = box_inter  # c_overlap= box_inter/torch.max(box_volume_i, box_volume_j)
        c_jaccard = box_inter/box_union
        c_cosine = box_inter/torch.pow(torch.sum(torch.log(torch.stack((box_volume_i, box_volume_j), 1), EPS), 1), 1/self.dim)
        c_dice = 2*box_inter/(box_volume_i + box_volume_j)

        lose_1 = torch.exp(c_overlap)
        lose_2 = torch.exp(c_jaccard)
        lose_3 = torch.exp(c_cosine)
        lose_4 = torch.exp(c_dice)

        lose_1 = self.loss(lose_1, overlaps[0])
        lose_2 = self.loss(lose_2, overlaps[1])
        lose_3 = self.loss(lose_3, overlaps[2])
        lose_4 = self.loss(lose_4, overlaps[3])
        return lose_1, lose_2, lose_3, lose_4
    
    def attention_scp(self, S, M, X, A, size_reg=False):
        edges = torch.nonzero(M).t()
        edges[1, :] = S[edges[0], edges[1]]
        
        att = torch.matmul(X, A)
        weight = torch_scatter.scatter_softmax(att[edges[1]], edges[0])
        a = torch_scatter.scatter_sum(X[edges[1]] * weight.unsqueeze(1), edges[0], dim=0)
        
        att2 = torch.sum(X[edges[1]] * a[edges[0]], 1)
        weight2 = torch_scatter.scatter_softmax(att2, edges[0])
        emb = torch_scatter.scatter_sum(X[edges[1]] * weight2.unsqueeze(1), edges[0], dim=0)
        
        if size_reg:
            sizes = torch.sum(M, 1).repeat_interleave(self.dim).view(len(emb), -1)
            emb = emb * (sizes ** (1.0 / self.dim))
        
        return emb
