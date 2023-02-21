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

class SelfAttentionBlock(nn.Module):

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


class MultiHeadSelfAttentionBlock(nn.Module):

    def __init__(self, dim_embedding, num_heads, dim_head=None):
        super().__init__()
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim_embedding, num_heads, dim_linear = 1024, dropout=0.1):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.num_heads = num_heads
        self.mhsa = MultiHeadSelfAttentionBlock(self.dim_embedding, self.num_heads)
        self.dropout = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(dim_embedding)
        self.lnorm2 = nn.LayerNorm(dim_embedding)
        self.ffn = nn.Sequential(
            nn.Linear(dim_embedding, dim_linear),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear, dim_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        y = self.dropout(self.mhsa(x, mask))
        y = self.lnorm1(y + x)
        y_ffn = self.ffn(y)
        y_ffn = self.lnorm2(y_ffn + y)
        return y_ffn

class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, dim_embedding, num_heads, dim_linear, dropout):
        super().__init__()
        self.block_list = [TransformerEncoderBlock(dim_embedding, num_heads, dim_linear, dropout) for _ in range(num_blocks)]
        self.module_list = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for module in self.module_list:
            y = module(x, mask)
        return y
