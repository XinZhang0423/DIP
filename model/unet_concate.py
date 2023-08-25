from typing import Dict,Literal
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
           nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(3,3), stride=(2, 2), padding=0),
           nn.BatchNorm2d(num_features=in_channels),
           nn.LeakyReLU(negative_slope=0.2),
           DoubleConv(in_channels=in_channels,out_channels=out_channels)
       )
      
class Up(nn.Module):
  
   def __init__(self,in_channels,out_channels,bilinear=True):
      
       super(Up,self).__init__()
       if bilinear:
           self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
           self.conv = nn.Sequential(
               DoubleConv(in_channels=in_channels, out_channels=out_channels,mid_channels=in_channels//2),
               nn.ReplicationPad2d(padding=1),
               nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=1,padding=0 ),
               nn.BatchNorm2d(num_features=out_channels),
               nn.LeakyReLU(negative_slope=0.2),
           )
       else:
           # 转置卷积没啥用，我不打算用的，所以写着玩，反正也不用
           self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels//2,kernel_size=2,stride=2)
           self.conv = nn.Sequential(
               DoubleConv(in_channels=in_channels, out_channels=out_channels,mid_channels=in_channels//2),
               nn.ReplicationPad2d(padding=1),
               nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=1,padding=0 ),
               nn.BatchNorm2d(num_features=out_channels),
               nn.LeakyReLU(negative_slope=0.2),
           )
      
   def forward(self,x1,x2):
       x1 = self.up(x1)
      
       diff_x = x2.size()[2]-x1.size()[2]
       diff_y = x2.size()[3]-x1.size()[3]
       # print(x1.size(),x2.size())
       # print(diff_x,diff_y)
       x1 = F.pad(input=x1,pad = [diff_x//2,diff_x-diff_x//2,diff_y//2,diff_y-diff_y//2])
       # print(x1.size(),x2.size())
       x = torch.cat([x1,x2],dim=1)
       x = self.conv(x)
       return x


class OutConv(nn.Sequential):
   def __init__(self,in_channels,output_channels) -> None:
       super(OutConv,self).__init__(nn.Conv2d(in_channels=in_channels,
                                              out_channels=output_channels,kernel_size=1))
      
class UNet(nn.Module):
  
   def __init__(self,
                input_channels :int =1,
                output_channels:int =1,
                bilinear: bool =True,
                base_c :int = 32)->None:
       """


       Args:
           input_channels (int, optional): input_channels. Defaults to 1.
           output_channels (int, optional): output_channels. Defaults to 1.
           bilinear (bool, optional): upsampling mode->bilinear. Defaults to True.
           base_c (int, optional):number of  base channels. Defaults to 32.
       """
       super(UNet,self).__init__()
       self.input_channels = input_channels
       self.output_channels = output_channels
      
       self.InConv = DoubleConv(in_channels=input_channels,out_channels=base_c,mid_channels=base_c//2)# 1->16
       self.down1 = Down(in_channels=base_c,out_channels=base_c*2)   # 32->64
       self.down2 = Down(in_channels=base_c*2,out_channels=base_c*4) # 64->128
       self.down3 = Down(in_channels=base_c*4,out_channels=base_c*8) # 128->256
       factor: Literal[2, 1] = 2 if bilinear else 1
       self.down4 = Down(in_channels=base_c*8,out_channels=base_c*16//factor) # 256->256
      
       self.up1 = Up(in_channels=base_c*16,out_channels=base_c*8//factor,bilinear=bilinear) # 512-> 128
       self.up2 = Up(in_channels=base_c*8,out_channels=base_c*4//factor,bilinear=bilinear) # 256->64
       self.up3 = Up(in_channels=base_c*4,out_channels=base_c*2//factor,bilinear=bilinear) # 128->32
       self.up4 = Up(in_channels=base_c*2,out_channels=base_c,bilinear=bilinear) # 64->32
      
       self.out_conv = OutConv(in_channels=base_c,output_channels=output_channels)
      
   def forward(self,x :torch.Tensor)-> Dict[str,torch.Tensor]:
       x1 = self.InConv(x)
       # print(x1.size())
       x2 = self.down1(x1)
       # print(x2.size())
       x3 = self.down2(x2)
       # print(x3.size())
       x4 = self.down3(x3)

       # print(x5.size())
       # print('x5')
       x = self.up1(x5,x4)
       # print(x.size())
       x = self.up2(x, x3)
       # print(x.size())
       x = self.up3(x, x2)
       # print(x.size())

       logits = self.out_conv(x)
      
       return {'out':logits}
  
      
      
      

