"""Adapted from
https://github.com/rwightman/pytorch-image-models/blob/acd6c687fd1c0507128f0ce091829b233c8560b9/timm/models/layers/drop.py#L140-L168
"""
import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Implement the stochastic Depth.
    See paper: https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self,
                 drop_prob: float = 0.1):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # Directly return the value if
        # 1. Probability is 0.0
        # 2. Validation
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        # Work with all the dimension, e.x.
        #  For ConvNet: (32, 3, 4, 4) -> (32, 1, 1, 1)
        #  For Transformer: (32, 256, 64) -> (32, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

        # Change all the number to 0 and 1 (Binarize)
        random_tensor.floor_()
        output = (x / keep_prob) * random_tensor
        return output
