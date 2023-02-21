import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
<Rereferences>
1. https://arxiv.labs.arxiv.org/html/1706.03762
2. https://theaisummer.com/einsum-attention
'''

class SelfAttentionBlock(nn):

    def __init__(self, dim_embedding, dim_query, dim_value):
        super().__init__()
        self.embedding_to_qk = nn.Linear(dim_embedding,
                                         2 * dim_query,
                                         bias=False)
        self.embedding_to_v = nn.Linear(dim_embedding, dim_value, bias=False)
        self.scale_factor = dim_query**-0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3  # shape = (batch, tokens, dim_embedding)

        qk = self.embedding_to_qk(x)
        q, k = tuple(einops.rearrange(qk, 'b t (k d) -> k b t d', k=2))
        v = self.embedding_to_v(x)

        # QK^T /scale
        attention_scores = torch.einsum('b i d, b j d -> b i j', q,
                                        k) * self.scale_factor

        if mask is not None:
            assert mask.shape == attention_scores.shape[1:]
            attention_scores = attention_scores.masked_fill(mask, -np.inf)

        attention_weights = F.softmax(attention_scores, dim=-1)

        return torch.einsum('b i j, b j d -> b i d', attention_weights, v)


class MultiHeadSelfAttentionBlock(nn):

    def __init__(self, dim_embedding, num_heads, dim_head=None):
        # paper; dim_q = dim_k = dim_v = dim_embedding/num_heads
        self.num_heads = num_heads
        self.dim_embedding = dim_embedding
        self.dim_head = dim_embedding // num_heads if dim_head is None else dim_head
        self.embedding_to_qkv = nn.Linear(dim_embedding,
                                          self.dim_head * num_heads * 3,
                                          bias=False)
        self.scale_factor = self.dim_head**-0.5
        self.concat_linear = nn.Linear(num_heads * self.dim_head,
                                       dim_embedding,
                                       bias=False)

        # batch x tokens x heads x d
        # concat -> batch x tokens x (d*heads)
        # linear layer: (d*heads) -> d

    def forward(self, x, mask=None):
        qkv = self.embedding_to_qkv(x)
        q, k, v = tuple(einops.rearrange(qkv, 'b t (k h d) -> k b t h d', k=3))
        attention_scores = torch.einsum('b i h d, b j h d -> b h i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == attention_scores.shape[2:]
            attention_scores = attention_scores.masked_fill(mask, -np.inf)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.einsum('b h i j, b h j d -> b h i d', attention_weights, v)
        concat = einops.rearrange(attention_output, 'b h t d -> b t (h d)')
        final = self.concat_linear(concat)
        return final