import sys
import torch
import torch.nn as nn
sys.path.append('models/')
sys.path.append('models/vit_library/')
from vit_library.segment_anything import sam_model_registry


class FOCAL_ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'FOCAL_ViT'
        self.net = sam_model_registry['vit_l']()

    def forward(self, x):
        x = self.net.image_encoder(x)
        return x
