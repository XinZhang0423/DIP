from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Sequential):
    
    def __init__(self,in_channels,out_channels,mid_channels=None):
        
        if mid_channels ==  None:
            mid_channels = out_channels
            
        super(DoubleConv, self).__init__(
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=(3,3),stride=1,padding=0),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=(3,3),stride=1,padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
            )


class Down(nn.Sequential):
    
    def __init__(self,in_channels,out_channels):
        super(Down,self).__init__(
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3), stride=(2, 2), padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )