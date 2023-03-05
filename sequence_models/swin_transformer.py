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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


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
        self.embedding_to_qkv = nn.Linear(dim, dim_head * num_heads * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.concat_linear =  nn.Linear(num_heads * dim_head, dim)

        # ((2M-1)^2, h)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] -1) *(2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0]) # (M,)
        coords_w = torch.arange(self.window_size[1]) # (M,)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")) # (2, M, M)
        coords_flatten = torch.flatten(coords, 1) # (2, M^2)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # (2, M^2, M^2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # (M^2, M^2, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # (M^2, M^2)
        self.register_buffer("relative_position_index", relative_position_index)


    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, H)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """


        # key, query, value -> attention score
        qkv = self.embedding_to_qkv(x)
        q, k, v = tuple(einops.rearrange(qkv, 'b n (k h d) -> k b h n d', k=3, h=self.num_heads))
        attentions = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # (h, M^2, M^2)
        attentions = attentions + relative_position_bias.unsqueeze(0) # (1, h, M^2, M^2)

        if mask is not None:
            num_windows = mask.shape[0]
            attentions = einops.rearrange(attentions, '(num_w B) h i j -> B num_w h i j', num_w=num_windows)
            mask = mask.unsqueeze(1).unsqueeze(0)
            attentions += mask
            attentions = einops.rearrange(attentions, 'B num_w h i j -> (num_w B) h i j')

        attention_probs = self.softmax(attentions)
        attention_probs = self.attn_drop(attention_probs)

        context = torch.einsum('b h i j, b h j d -> b h i d', attention_probs, v)
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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class SwinTransformerStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class SwinTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass
