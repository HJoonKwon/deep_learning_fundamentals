import torch
import torch.nn as nn
import einops

from .utils import *

'''
<Rereferences>
1. https://arxiv.org/abs/2103.14030
2. https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
3. https://towardsdatascience.com/a-comprehensive-guide-to-swin-transformer-64965f89d14c
4. https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/models/swin/modeling_swin.py
'''


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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


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
