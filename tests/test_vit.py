import torch
from sequence_models.vit import *

# class VisionTransformer(nn.Moudle):
#     def __init__(self,
#                  img_size,
#                  patch_dim=16,
#                  in_channels=3,
#                  dim_embed=512,
#                  num_blocks=6,
#                  num_heads=4,
#                  dim_linear=1024,
#                  dim_head=None,
#                  dropout=0,
#                  num_classes=10,
#                  transformer=None,
#                  classification=None):

def test_vit_module():
    vit = VisionTransformer(
    img_size = 224,
    patch_dim = 16,
    in_channels = 3,
    dim_embed = 512,
    num_blocks = 6,
    num_heads = 4,
    dim_linear = 1024,
    dropout= 0,
    num_classes = 4,
    transformer = None,
    classification = True
    )

    img = torch.randn(4, 3, 224, 224)
    output = vit(img)
    assert output.shape == (4, 4)

