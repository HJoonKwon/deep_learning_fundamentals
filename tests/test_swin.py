import torch
import pytest
from sequence_models.swin_transformer import *


def test_patch_embedding():
    batch_size = 4
    img_size = 28
    patch_dim = 4
    in_channels = 3
    dim_embed = 32
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

    img_size = (28, 14)
    patch_dim = (7, 7)
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

    img_size = (28, 14)
    patch_dim = (7, 2)
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
    B, H, W, C = 4, 56, 56, 3
    ws = H//2
    x = torch.randn(B, H, W, C)
    num_windows = H//ws * W//ws
    windows = window_partition(x, window_size=ws)
    assert windows.shape == (B*num_windows, ws, ws, C)

    #https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L206
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows_o = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)
    assert windows.shape == windows_o.shape
    assert torch.equal(windows, windows_o)


def test_window_reverse():
    B, H, W, C = 4, 56, 56, 3
    ws = H//8
    windows = torch.randn(B*64, ws, ws, C)
    x = window_reverse(windows, window_size=ws, H=H, W=W)
    assert x.shape == (B, H, W, C)

    x_o = windows.view(-1, H // ws, W // ws, ws, ws, C)
    x_o = x_o.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)

    assert torch.equal(x, x_o)


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

def test_mlp():
    in_features = 10
    hidden_features = 20
    out_features = 10

    x = torch.randn(in_features)
    model = Mlp(in_features=in_features, hidden_features=hidden_features, out_features=out_features)
    y = model(x)
    assert y.shape == (out_features,)

def test_swin_transformer_block():
    dim = 48
    input_resolution = (56, 56)
    num_heads = 4
    window_size = 7
    shift_size = window_size // 2
    mlp_ratio = 2
    qkv_bias = True
    qk_scale = None
    drop = .1
    attn_drop = .1
    drop_path = 0.2
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm

    swin_block = SwinTransformerBlock(
        dim=dim,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path,
        act_layer=act_layer,
        norm_layer=norm_layer
    )

    x = torch.randn(4, 56*56, dim)
    y = swin_block(x)
    assert y.shape == (4, 56*56, dim)


def test_patch_merging_einops():
    input_resolution = (56, 56)
    dim = 48
    model = PatchMerging(input_resolution, dim)
    B, H, W, C = 4, *input_resolution, dim
    x = torch.randn(B, H*W, C)
    y = model(x)
    assert y.shape == (B, H//2 * W//2, 2*C)

    # test einops for patch merging
    B, H, W, C = 1, 6, 6, 1
    x = torch.randn(B, H, W, C)

    # einops(@joon9502)
    x1 = einops.rearrange(x, 'b (nh h) (nw w) c -> (b nh nw) h w c', nh=H//2, nw=W//2)
    x1 = einops.rearrange(x1, '(b nh nw) h w c -> b (nh nw) (w h c)', b = B, nh=H//2, nw=W//2)

    # original(https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py#L345)
    p0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C -> top left corner of all patches
    p1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C -> bottom left of all patches
    p2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C -> top right
    p3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C -> bottom right
    x2 = torch.cat([p0, p1, p2, p3], -1)
    x2 = x2.view(B, -1, 4 * C)
    assert x1.shape == x2.shape
    assert torch.equal(x1, x2)


def test_swin_layer():
    input_resolution = (56, 56)
    B, H, W, C = 4, *input_resolution, 48
    depth = 2
    patch_merge = PatchMerging
    x = torch.randn(B, H*W, C)
    num_heads = 4
    window_size = 7
    qkv_bias = True
    qk_scale = None
    drop = .1
    attn_drop = .1
    drop_path = 0.2
    swin_layer = SwinLayer(
        dim=C,
        depth=depth,
        input_resolution=input_resolution,
        num_heads=num_heads,
        window_size=window_size,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path,
        patch_merge=patch_merge,
        use_grad_checkpoint=False
    )
    y = swin_layer(x)
    assert y.shape == (B, H//2 * W//2, 2*C)

def test_swin_transformer():
    B = 4
    H, W, C = 224, 224, 3
    num_classes = 10
    model = SwinTransformer(img_size=224,
                            patch_dim=4,
                            in_channels=3,
                            num_classes=num_classes,
                            embed_dim=96,
                            depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24],
                            window_size=7,
                            mlp_ratio=4,
                            qkv_bias=True,
                            qk_scale=None,
                            drop=0.1,
                            attn_drop=0.1,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,
                            ape=False,
                            patch_norm=True,
                            use_grad_checkpoint=False)

    img = torch.randn(B, C, H, W)
    y = model(img)
    print(y.shape)
