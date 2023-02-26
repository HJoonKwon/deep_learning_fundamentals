import torch
import pytest
from sequence_models.swin_transformer import *


def test_patch_embedding():
    batch_size = 4
    img_size = 224
    patch_dim = 4
    in_channels = 3
    dim_embed = 512
    embedding_module = PatchEmbedding(
        img_size=img_size,
        patch_dim=patch_dim,
        in_channels=in_channels,
        dim_embed=dim_embed,
    )
    img = torch.randn(batch_size, in_channels, img_size, img_size)
    output = embedding_module(img)
    patches_resolution = img_size // patch_dim
    assert output.shape == (batch_size, patches_resolution ** 2, dim_embed)

    img_size = (224, 112)
    patch_dim = (8, 4)
    embedding_module = PatchEmbedding(
        img_size=img_size,
        patch_dim=patch_dim,
        in_channels=in_channels,
        dim_embed=dim_embed,
    )
    img = torch.randn(batch_size, in_channels, *img_size)
    output = embedding_module(img)
    patches_resolution = [img_size[0] //
                              patch_dim[0], img_size[1]//patch_dim[1]]
    num_patches = patches_resolution[0] * patches_resolution[1]
    assert output.shape == (batch_size, num_patches, dim_embed)
