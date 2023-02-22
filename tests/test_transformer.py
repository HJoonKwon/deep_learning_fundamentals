import torch
import pytest
from sequence_models.transformer import *


def test_self_attention_block():
    batch_size = 4
    tokens = 5
    dim_embedding = 16
    dim_query = 4
    dim_value = 8
    x = torch.randn(batch_size, tokens, dim_embedding)

    block = SelfAttentionBlock(dim_embedding,
                               dim_query,
                               dim_value)

    output = block(x)
    assert output.shape == (batch_size, tokens, dim_value)


