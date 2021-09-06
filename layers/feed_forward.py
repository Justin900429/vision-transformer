"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
"""

from typing import Optional, Union
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, in_features: int,
                 factor: Union[int, float] = None,
                 act_layer: Optional = nn.GELU,
                 drop: float = 0.0):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features,
                      out_features=int(in_features * factor)),
            nn.Dropout(drop),
            act_layer(),
            nn.Linear(in_features=int(in_features * factor),
                      out_features=in_features),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.fc(x)

