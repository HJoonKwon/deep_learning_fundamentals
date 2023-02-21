import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformer import TransformerEncoder

class VisionTransformer(nn.Moudle):
    def __init__(self,
                 img_size,
                 patch_dim=16,
                 in_channels=3,
                 dim_embed=512,
                 num_blocks=6,
                 num_heads=4,
                 dim_linear=1024,
                 dim_head=None,
                 dropout=0,
                 num_classes=10,
                 transformer=None,
                 classification=None):
        super().__init__()
        assert img_size % patch_dim == 0
        self.dim_token = patch_dim ** 2 * in_channels
        self.dim_head = dim_embed // num_heads if dim_head is None else dim_head
        self.num_tokens = (img_size // patch_dim) ** 2
        self.num_patch_w = self.num_patch_h = img_size // patch_dim
        self.dim_embed = dim_embed
        self.classification = classification

        # define layers
        self.projection = nn.Linear(self.dim_token, dim_embed)
        self.embed_dropout = nn.Dropout(dropout)

        if self.classification:
            # (batch_dim, num_token_dim, embed_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim_embed))
            self.pos_emb1D = nn.Parameter(
                torch.randn(self.num_tokens+1, self.dim_embed))
            self.mlp_head = nn.Linear(self.dim_embed, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(
                torch.randn(self.num_tokens, self.dim_embed))

        if transformer is None:
            self.transformer = TransformerEncoder(num_blocks,
                                                  dim_embed,
                                                  num_heads,
                                                  dim_linear,
                                                  dropout)
        else:
            self.transformer = transformer

    def forward(self, x, mask=None):
        batch = x.shape[0]
        flattened_patches = einops.rearrange(x, 'b c (patch_w m) (patch_h n)-> b (mn) (patch_w patch_h c)', m=self.num_patch_w, n=self.num_patch_h)
        embed_patches = self.projection(flattened_patches) # (batch, num_tokens, dim_embed)

        if self.classification:
            cls_token_batch = self.cls_token.expand([batch, -1, self.dim_embed])
            embed_patches = torch.concat(embed_patches, cls_token_batch, dim=1) # (batch, num_tokens+1, dim_embed)

        embed_patches += self.pos_emb1D
        embed_patches = self.embed_dropout(embed_patches)

        y = self.transformer(embed_patches)
        
