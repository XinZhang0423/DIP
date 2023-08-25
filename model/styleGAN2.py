import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

class MappingNetwork(nn.Module):
    # mapping的部分
    def __init__(self,features,n_layers):#features向量长度,n_layers全连接层个数
        
        super().__init__()
        layers = []
        
        for i in range(n_layers):
            layers.append(EqualizedLinear(features,features))
            layers.append(nn.LeakyReLU(negative_slope=0.2,inplace=True))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self,z):
        
        z = F.normalize(z,dim=1) # 先在channel标准化然后再输入？这里对z的维度不了解
                
        return self.net(z)

class Generator(nn.Module): 
    # synthesise generator
    def __init__(self,log_resolution,d_latent,n_features=16,max_features=512):
        """
        
        log_resolution is the log2​ of image resolution
        d_latent is the dimensionality of w
        n_features number of features in the convolution layer at the highest resolution (final block)
        max_features maximum number of features in any generator block
        
        """
        super().__init__()
        features = [256,512,128,64,32,16]
        #features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)] #Something like [512, 512, 256, 128, 64, 32]
        
        self.n_blocks = len(features)
        
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        
        self.blocks = nn.ModuleList(blocks)
        
        self.up_sample = UpSample()
        
    
    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):