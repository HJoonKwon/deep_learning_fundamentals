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

    img_size = (224, 112)
    patch_dim = (8, 4)
    embedding_module = PatchEmbedding(
        img_size=img_size,
        patch_dim=patch_dim,
        in_channels=in_channels,
        dim_embed=dim_embed,
        norm_layer=nn.LayerNorm
    )
    img = torch.randn(batch_size, in_channels, *img_size)
    output = embedding_module(img)
    patches_resolution = [img_size[0] //
                              patch_dim[0], img_size[1]//patch_dim[1]]
    num_patches = patches_resolution[0] * patches_resolution[1]
    assert output.shape == (batch_size, num_patches, dim_embed)


def test_window_partition():
    B, H, W, C = 4, 224, 224, 3
    ws = H//8
    x = torch.randn(B, H, W, C)
    num_windows = 64
    windows = window_partition(x, window_size=ws)
    assert windows.shape == (B*num_windows, ws, ws, 3)

def test_window_reverse():
    B, H, W, C = 4, 224, 224, 3
    ws = H//8
    windows = torch.randn(B*64, ws, ws, 3)
    x = window_reverse(windows, window_size=ws, H=H, W=W)
    assert x.shape == (B, H, W, C)

def test_window_multi_head_self_attention():
    B, N, C = 4, 16, 64
    num_windows = 16
    window_size = (4, 4)
    x = torch.randn(B*num_windows, N, C)
    mask = torch.randn(num_windows, N, N)
    model = WindowMultiHeadSelfAttention(
        dim=C,
        window_size=window_size,
        num_heads=4,
        attn_drop=0.1,
        proj_drop=0.1,
    )
    y = model(x, mask)
    assert y.shape == (B*num_windows, N, C)
