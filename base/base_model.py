"""Base Model class."""
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """Base class for all models."""
    
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f'\nTrainable parameters: {params:,}'
