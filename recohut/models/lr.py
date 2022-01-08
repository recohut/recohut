# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models/lr.ipynb (unless otherwise specified).

__all__ = ['LR']

# Cell
import torch

from ..layers.common import FeaturesLinear

# Cell
class LR(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sigmoid(self.linear(x).squeeze(1))