# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
<Rereferences>
1. https://arxiv.org/abs/2103.14030
2. https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
3. https://towardsdatascience.com/a-comprehensive-guide-to-swin-transformer-64965f89d14c
4. https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/swin/modeling_swin.py
5. https://github.com/huggingface/transformers/blob/main/src/transformers/models/swin/modeling_swin.py
'''

import collections
import torch
import torch.nn as nn
import einops

from .utils import *


def relative_position_index(window_size):
    pass


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    windows = einops.rearrange(
        x, 'b (w1 n1) (w2 n2) c -> (b n1 n2) w1 w2 c', w1=window_size, w2=window_size)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    n1, n2 = H//window_size, W//window_size
    x = einops.rearrange(windows, '(b n1 n2) w1 w2 c -> b (n1 w1) (n2 w2) c',
                         w1=window_size, w2=window_size, n1=n1, n2=n2)
    return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.channel_reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm_layer = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, f"Wrong input size {H}x{W} != {L}"
        assert H % 2 == 0 and W % 2 == 0, f"Cannot merge patches since ({H}x{W} are not even)"
        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        x = einops.rearrange(x, 'b (nh h) (nw w) c -> (b nh nw) h w c', nh=H//2, nw=W//2)
        x = einops.rearrange(x, '(b nh nw) h w c -> b (nh nw) (w h c)', b=B, nh=H//2, nw=W//2) # same arrangement as the original one
        x = self.channel_reduction(self.norm_layer(x))

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        # self.layer = nn.Sequential()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_dim=4,
                 in_channels=3,
                 dim_embed=512,
                 norm_layer=None) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_dim = to_2tuple(patch_dim)
        patches_resolution = [img_size[0] //
                              patch_dim[0], img_size[1]//patch_dim[1]]
        num_patches = patches_resolution[0] * patches_resolution[1]
        dim_patch = patch_dim[0] * patch_dim[1] * in_channels
        self.img_size = img_size
        self.patch_dim = patch_dim
        self.patches_resolution = patches_resolution
        self.num_patches = num_patches
        self.dim_patch = dim_patch
        self.dim_embed = dim_embed

        self.embedding = nn.Conv2d(
            in_channels, dim_embed, kernel_size=self.patch_dim, stride=self.patch_dim)

        if norm_layer is not None:
            self.norm_layer = norm_layer(dim_embed)
        else:
            self.norm_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            flattened: (B, patch_size x patch_size, dim_embed)
        """
        B, C, H, W = x.shape
        # B x dim_embed x patches_resolution[0] x patches_resolution[1]
        embedded = self.embedding(x)
        flattened = einops.rearrange(embedded, 'b d ph pw -> b (ph pw) d')
        if self.norm_layer is not None:
            flattened = self.norm_layer(flattened)
        return flattened


class WindowMultiHeadSelfAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5 if qk_scale is None else qk_scale
        self.embedding_to_qkv = nn.Linear(
            dim, dim_head * num_heads * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.concat_linear = nn.Linear(num_heads * dim_head, dim)

        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (
                window_size, window_size)
        )
        # ((2M-1)^2, h)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])  # (M,)
        coords_w = torch.arange(self.window_size[1])  # (M,)
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing="ij"))  # (2, M, M)
        coords_flatten = torch.flatten(coords, 1)  # (2, M^2)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # (2, M^2, M^2)
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # (M^2, M^2, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (M^2, M^2)
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, H)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        # key, query, value -> attention score
        qkv = self.embedding_to_qkv(x)
        query, key, value = tuple(einops.rearrange(
            qkv, 'b n (k h d) -> k b h n d', k=3, h=self.num_heads))
        attention_scores = torch.einsum(
            'b h i d, b h j d -> b h i j', query, key) * self.scale

        # relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(
            -1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] *
            self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # (h, M^2, M^2)
        attention_scores = attention_scores + \
            relative_position_bias.unsqueeze(0)  # (1, h, M^2, M^2)

        if mask is not None:
            num_windows = mask.shape[0]
            attention_scores = einops.rearrange(
                attention_scores, '(num_w B) h i j -> B num_w h i j', num_w=num_windows)
            mask = mask.unsqueeze(1).unsqueeze(0)
            attention_scores += mask
            attention_scores = einops.rearrange(
                attention_scores, 'B num_w h i j -> (num_w B) h i j')

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_drop(attention_probs)

        context = torch.einsum(
            'b h i j, b h j d -> b h i d', attention_probs, value)
        context = einops.rearrange(context, 'b h i d -> b i (h d)')
        context = self.concat_linear(context)
        context = self.proj_drop(context)
        return context


class ShiftedWindowMultiHeadSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, mlp_ratio,
                 qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer) -> None:
        super().__init__()

        # Define modules to build a Swin block
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.window_attn = WindowMultiHeadSelfAttention(dim,
                                                        window_size,
                                                        num_heads,
                                                        qkv_bias=qkv_bias,
                                                        qk_scale=qk_scale,
                                                        attn_drop=attn_drop,
                                                        proj_drop=drop)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=dim*mlp_ratio,
                       out_features=dim,
                       act_layer=act_layer,
                       drop=drop)

        # stochastic depth
        # https://towardsdatascience.com/review-stochastic-depth-image-classification-a4e225807f4a
        self.drop_path = DropPath(
            drop_prob=drop_path, scale_by_keep=True) if drop_path > 0 else nn.Identity()

        self.shift_size = shift_size
        self.window_size = window_size
        self.input_resolution = input_resolution

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            image_mask = torch.zeros((1, height, width, 1), dtype=dtype)

            # define slices to create sub windows
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            # assign each area a distinct number
            area_num = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    image_mask[:, h_slice, w_slice, :] = area_num
                    area_num += 1

            mask_windows = window_partition(
                image_mask, window_size=self.window_size)  # (N, Wh, Ww, 1)
            mask_windows = einops.rearrange(
                mask_windows, 'n wh ww c -> n (wh ww c)')
            # 0 for corresnponding sub-window, (N, Wh*Ww, Wh*Ww)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x):

        # x: (B, H*W, dim)
        short_cut = x
        x = self.norm1(x)

        # rearrange input for window partition
        x = einops.rearrange(x, 'b (h w) c -> b h w c',
                             h=self.input_resolution[0], w=self.input_resolution[1])
        B, h, w, dim = x.shape

        # TODO:: pad hidden_states to multiples of window size

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # (B*num_windows, Wh, Ww, C)
        windows = window_partition(x, window_size=self.window_size)
        windows = einops.rearrange(windows, 'b Wh Ww c -> b (Wh Ww) c')

        # get attention mask and do shifted window MHS
        attn_mask = self.get_attn_mask(h, w, dtype=shifted_x.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(shifted_x.device)
        attn_windows = self.window_attn(windows, mask=attn_mask)

        # rearrange attention windows to reverse it to input size
        attn_windows = einops.rearrange(
            attn_windows, 'b (Wh Ww) c -> b Wh Ww c', Wh=self.window_size, Ww=self.window_size)
        shifted_attn_windows = window_reverse(
            attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_windows = torch.roll(shifted_attn_windows, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_windows = shifted_attn_windows

        attn_windows = einops.rearrange(
            attn_windows, 'b Wh Ww c -> b (Wh Ww) c')

        # apply stochastic depth
        hidden_states = short_cut + self.drop_path(attn_windows)

        output = self.drop_path(
            self.mlp(self.norm2(hidden_states))) + hidden_states
        return output


class SwinLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_grad_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_grad_checkpoint=False) -> None:

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_grad_checkpoint = use_grad_checkpoint

        self.swin_blocks = nn.ModuleList([SwinTransformerBlock(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if i % 2 == 0 else window_size//2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer
        ) for i in range(depth)])

    def forward(self, x):
        pass


class SwinTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # https://huggingface.co/spaces/Roll20/pet_score/blame/main/lib/timm/models/layers/drop.py
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
