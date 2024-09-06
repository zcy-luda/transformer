'''
Author: Chengyu Zheng
Date: 2024-09-06
Description: 
'''

from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head=8):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        self.attention = ScaleDotProductAttention(d_model)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q = q.view(-1, self.nhead, q.shape(1), self.head_dim)
        k = k.view(-1, self.nhead, k.shape(1), self.head_dim)
        v = v.view(-1, self.nhead, v.shape(1), self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = out.transpose(1, 2).contiguous().view(-1, q.shape(1), self.d_model)
        out = self.w_concat(out)

        return out
