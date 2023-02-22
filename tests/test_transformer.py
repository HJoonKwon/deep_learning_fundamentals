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


def test_multi_head_attention_block():
    batch_size = 4
    tokens = 5
    dim_embedding = 64
    num_heads = 4
    x = torch.randn(batch_size, tokens, dim_embedding)

    block = MultiHeadSelfAttentionBlock(dim_embedding,
                                        num_heads)
    output = block(x)
    assert output.shape == (batch_size, tokens, dim_embedding)


def test_transformer_encoder_block():
    batch_size = 4
    tokens = 5
    dim_embedding = 64
    num_heads = 4
    dim_linear = 1024
    dropout = 0.1
    x = torch.randn(batch_size, tokens, dim_embedding)

    block = TransformerEncoderBlock(dim_embedding,
                                    num_heads,
                                    dim_linear,
                                    dropout)
    output = block(x)
    assert output.shape == (batch_size, tokens, dim_embedding)
