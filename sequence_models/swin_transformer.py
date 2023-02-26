import collections
from itertools import repeat
import torch
import torch.nn as nn
import einops


'''
<Rereferences>
1. https://arxiv.org/abs/2103.14030
2. https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
3. https://towardsdatascience.com/a-comprehensive-guide-to-swin-transformer-64965f89d14c
4. https://github.com/huggingface/pytorch-image-models/blob/450b4affbd73393ee02b075d6298ef29b2eca74/timm/layers/helpers.py#L10
'''


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def window_partition():
    pass


def window_reverse():
    pass


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
                 dim_embed=512) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_dim = to_2tuple(patch_dim)
        patches_resolution = [img_size[0] //
                              patch_dim[0], img_size[1]//patch_dim[1]]
        num_patches = patches_resolution[0] * patches_resolution[1]
        dim_patch = patch_dim[0] * patch_dim[1] * in_channels
        self.patches_resolution = patches_resolution
        self.num_patches = num_patches
        self.dim_patch = dim_patch
        self.dim_embed = dim_embed

        self.embedding = nn.Conv2d(
            in_channels, dim_embed, kernel_size=self.patch_dim, stride=self.patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        embedded = self.embedding(x) # B x dim_embed x patches_resolution[0] x patches_resolution[1]
        flattened = einops.rearrange(embedded, 'b d ph pw -> b (ph pw d)')
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
