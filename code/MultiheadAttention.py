import torch
import copy
import torch.nn as nn
from torch.nn.functional import dropout
import torch.nn.functional as F
import math

def attention(query, key, value, mask = None, dropout = None):
    '''
    注意力机制
    :param query:
    :param key:
    :param value:
    :return:
    '''
    d_model = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_model)
    # masked_fill 当mask为0，将对应的额scores修改为-1e9
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_atten = F.softmax(scores,dim=-1)
    # 使用dropout随机位置0
    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten

# test input size=(2，2，4)  批次，句长，嵌入维度
q = torch.tensor([[[1.,2.,3.,4.],[5.,6.,7.,8.]],[[1.,2.,3.,4.],[5.,6.,7.,8.]]])
k = torch.tensor([[[1.,2.,3.,4.],[5.,6.,7.,8.]],[[1.,2.,3.,4.],[5.,6.,7.,8.]]])
v = torch.tensor([[[1.,2.,3.,4.],[5., 6., 7.,8.]],[[1.,2.,3.,4.],[5.,6.,7.,8.]]])

atten, p_atten = attention(q,k,v)
print("atten", atten)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # contiguous能够让转置之后的张量使用view操作
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

Multi_head = MultiHeadedAttention(h=2, d_model=4)
MulAtten = Multi_head.forward(q,k,v)
print("MulAtten",MulAtten)
