# encoding: utf-8
"""
@author:  Huy Q Can
@contact: huysk82000@gmail.com
"""

from abc import abstractmethod
from turtle import forward
import numpy as np
import torch.nn.functional as F
from torch import nn

from layers.conv_layer import conv3x3


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
        
    def encode(self, input):
        raise NotImplementedError
    
    def decode(self, input):
        raise NotImplementedError
    
    def sample(self, batch_size, current_device, **kwargs):
        raise NotImplementedError
    
    def generate(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, *input):
        pass
    
    @abstractmethod
    def loss_function(self, *input, **kwargs):
        pass
    
    
    
    
        