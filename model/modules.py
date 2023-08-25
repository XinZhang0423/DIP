import torch
import torch.nn as nn
import pytorch_lightning as pl


import numpy as np

import os

from skimage.metrics import peak_signal_noise_ratio,structural_similarity

from .common import * # for Lipschitz Gaussian controle modules
from .swinUNETR.SwinUNetr import * # for SwinUNetr  encoder= swin transformer, decoder = cnn
from .swinUNET.SwinUNet import * # for SwinUNet  unet architecture with pure swin transformer
from .swinIR.swinIR import *  # for SwinIR  replace the bottleneck of unet by swin transformer
from .restormer.restormer import * 
from .spectformer.spectformer import *
from .DIP.DIP_blocks import * # for DIP blocks UNet Up, UNet Down, UNet Deep
# 论文附赠代码部分：
from torch.nn import Parameter



# the backbone of Full DIP (with Lipschitz Gaussian controle)
class Full_DIP_backbone(pl.LightningModule):

    def __init__(self, param_scale, 
                 config, suffix):
        super().__init__()
        # random_seed = 114514
        # pl.seed_everything(random_seed)
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']  # Learning rate
        self.iter_DIP = config['iters']  # Number of iterations
        self.param = param_scale  # Scaling parameter for normalisation
        self.path = "/home/xzhang/Documents/simplified_pipeline/data/results/images/"  # Path to save images
        self.suffix = suffix  # Suffix for experiment name
        self.repeat = config['repeat']  # Repetition count number of times to repeat (random seed)
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config  # Network configuration
        self.model_name = config['model_name']  # Model name
        self.num_layers = config['num_layers']  # Number of layers
        self.num_channels = config['num_channels']  # List of channel counts for each layer
        self.ln_lambda = config['ln_lambda']  # Lambda for layer normalization
        self.upsample_mode = config['upsampling_mode']  # Upsampling mode
        self.sigma = config['sigma']  # Sigma value
        self.initial_param = config['init']  # Initialization method
        self.initialize_network()  # Initialize the network architecture
        # for m in self.modules():
		# # 判断是否属于Conv2d
        #     if isinstance(m, nn.Conv2d):
        #         if self.initial_param == 'xavier_norm':
        #             torch.nn.init.xavier_normal_(m.weight.data)
        #         elif self.initial_param == 'xavier_uniform':
        #             torch.nn.init.xavier_uniform_(m.weight.data)
        #         elif self.initial_param == 'kaiming_norm':
        #             torch.nn.init.kaiming_normal_(m.weight.data)
        #         elif self.initial_param == 'kaiming_uniform':
        #             torch.nn.init.kaiming_uniform_(m.weight.data)
        
    def initialize_network(self):

        L_relu = 0.2
        num_channel =self.num_channels

        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        
        self.encoder_layers.append(
                          nn.Sequential(
                                   Conv(1, num_channel[0], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                   bn(num_channel[0],mean_only=(self.ln_lambda>0)),
                                   nn.LeakyReLU(L_relu),
                                   Conv(num_channel[0], num_channel[0], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                   bn(num_channel[0],mean_only=(self.ln_lambda>0)),
                                   nn.LeakyReLU(L_relu)))
        
        for i in range(len(self.num_channels)-1): 
            self.encoder_layers.append(
                        nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[i], num_channel[i], 3, stride=(2, 2), padding=0),
                                   nn.BatchNorm2d(num_channel[i]),
                                   nn.LeakyReLU(L_relu)))
            
            self.encoder_layers.append(
                        nn.Sequential(
                                   Conv(num_channel[i], num_channel[i+1], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                   bn(num_channel[i+1],mean_only=(self.ln_lambda>0)),
                                   nn.LeakyReLU(L_relu),
                                   Conv(num_channel[i+1], num_channel[i+1], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                   bn(num_channel[i+1],mean_only=(self.ln_lambda>0)),
                                   nn.LeakyReLU(L_relu)))
            
        for i in range(len(self.num_channels)-2):    
            self.decoder_layers.append(
                          nn.Sequential(Up(self.upsample_mode,num_channel[self.num_layers-i],self.sigma),
                                        Conv(num_channel[self.num_layers-i], num_channel[self.num_layers-i-1], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                        bn(num_channel[self.num_layers-i-1],mean_only=(self.ln_lambda>0)),
                                        nn.LeakyReLU(L_relu)))
            self.decoder_layers.append(
                          nn.Sequential( Conv(num_channel[self.num_layers-i-1], num_channel[self.num_layers-i-1], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication')   ,                          
                                        bn(num_channel[self.num_layers-i-1],mean_only=(self.ln_lambda>0)),
                                        nn.LeakyReLU(L_relu),
                                        Conv(num_channel[self.num_layers-i-1], num_channel[self.num_layers-i-1], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication')      ,                       
                                        bn(num_channel[self.num_layers-i-1],mean_only=(self.ln_lambda>0)),
                                        nn.LeakyReLU(L_relu)))
        self.decoder_layers.append(
                          nn.Sequential(Up(self.upsample_mode,num_channel[1],self.sigma),
                                        Conv(num_channel[1], num_channel[0], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                        bn(num_channel[0],mean_only=(self.ln_lambda>0)),
                                        nn.LeakyReLU(L_relu)))
        self.decoder_layers.append(
                          nn.Sequential(Conv(num_channel[0], num_channel[0], kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication')   ,                          
                                        bn(num_channel[0],mean_only=(self.ln_lambda>0)),
                                        nn.LeakyReLU(L_relu),
                                        Conv(num_channel[0], 1, kernel_size = 3, stride= 1, ln_lambda=self.ln_lambda, bias=True, pad='Replication'),
                                        bn(1,mean_only=(self.ln_lambda>0)))    )   
        self.positivity = nn.ReLU() 

        
    def forward(self, x):
        out = x
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out

    # 定义损失函数为输出和含噪图像的tensor的mse
    def DIP_loss(self, out, image_corrupt_torch):
        return torch.nn.MSELoss()(out, image_corrupt_torch) # for DIP and DD
        
    # 定义训练流程，计算一次前向传播返回loss，（中间有logger记录tensorboard和early stopping）
    def training_step(self, train_batch, batch_idx):
        image_net_input_torch, image_corrupt_torch = train_batch
        out = self.forward(image_net_input_torch)

        loss = self.DIP_loss(out, image_corrupt_torch)
        
        try:
            out_np = out.detach().numpy()
        except:
            out_np = out.cpu().detach().numpy()

        out_np = np.squeeze(out_np)
        out_np = out_np*self.param 

        # Create directory if it doesn't exist
        os.makedirs(self.path + format(self.suffix)  + '/train_' + str(self.repeat), exist_ok=True)
        np.save(self.path + format(self.suffix) + '/train_' + str(self.repeat) + '/iters_' + format(self.current_epoch) + '.npy',out_np)

        return loss
    
    #配置优化器，可以选择各种优化器
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5E-8) 
        return optimizer
    

# DIP 输入加噪音
# 直接在输入加入噪音（我试了一下，噪声不能太大不然会严重影响结果，因为输入是正则化以后得）  
# Full_DIP_noise_v0 v1,v2,v3,v4 5 different way to add noise
class Full_DIP_noise_v0(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
    def forward(self,x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
                    
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out

#  latent space add noise      
class Full_DIP_noise_v1(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
    def forward(self,x):
        out = x
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)
            
        if self.sigma_p == 0 :
            out = out
        else:
            # noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*out.size()])
            noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = out + noise
            
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out

# 在latent space 
class Full_DIP_noise_v2(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
    def forward(self,x):
        out = x
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)
            
        if self.sigma_p == 0 :
            out = out
        else:
            # noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*out.size()])
            noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*out.size()])
            out = out + noise
            min_vals, _ = out.min(dim=2, keepdim=True)
            max_vals, _ = out.max(dim=2, keepdim=True)
            min_vals,_ = min_vals.min(dim=3, keepdim=True)
            max_vals,_ = max_vals.max(dim=3, keepdim=True)
            out = (out - min_vals) / (max_vals - min_vals)
                        
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out
    
# 在latent space 先正则再加噪音
class Full_DIP_noise_v3(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
    def forward(self,x):
        out = x
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)
            
        if self.sigma_p == 0 :
            out = out
        else:
            min_vals, _ = out.min(dim=2, keepdim=True)
            max_vals, _ = out.max(dim=2, keepdim=True)
            min_vals,_ = min_vals.min(dim=3, keepdim=True)
            max_vals,_ = max_vals.max(dim=3, keepdim=True)
            out = (out - min_vals) / (max_vals - min_vals)
            # noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*out.size()])
            noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*out.size()])
            out = out + noise
                        
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out
    
# 在输入加噪音再正则    
class Full_DIP_noise_v4(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
        
    def forward(self,x):
        if self.sigma_p == 0 :
            out = x 
        else:
            # noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
            min_vals, _ = out.min(dim=2, keepdim=True)
            max_vals, _ = out.max(dim=2, keepdim=True)
            min_vals,_ = min_vals.min(dim=3, keepdim=True)
            max_vals,_ = max_vals.max(dim=3, keepdim=True)
        
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)
            
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out

# decoder backbone
class DIP_decoder_backbone(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
        
    def forward(self,x):
        if self.sigma_p == 0 :
            out = x 
        else:
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            out = x + noise
            min_vals, _ = out.min(dim=2, keepdim=True)
            max_vals, _ = out.max(dim=2, keepdim=True)
            min_vals,_ = min_vals.min(dim=3, keepdim=True)
            max_vals,_ = max_vals.max(dim=3, keepdim=True)
                        
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        
        return out


# Bernoulli sampling DIP   
class sampled_DIP(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.benoulli_p = config['benoulli']
        self.s_down = config['s_down']
        self.s_up = config['s_up']
        super().__init__(param_scale, 
                 config, suffix)
        
    def forward(self,x):
        sample_mask = torch.distributions.Bernoulli(probs=self.benoulli_p).sample(x.shape)  # 使用伯努利分布进行采样,p概率为1，否则为0
        sample_mask = sample_mask.float()  # 将采样结果转换为float类型，值为0或1
        sampled_x = x * (sample_mask * self.s_up + (1 - sample_mask) * self.s_down)  # 根据伯努利采样结果对像素进行缩放，为1则放大，为0则缩小
        out = sampled_x   
        
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)             
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out  
    

# pixel-wise masked DIP
class pw_masked_DIP(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        self.ratio = config['ratio']
        self.s_down = config['s_down']
        self.s_up = config['s_up']
        super().__init__(param_scale, 
                 config, suffix)
        
    def forward(self,x):
        mask = torch.distributions.Uniform(low=0, high=1).sample(x.shape)  # 生成与x形状相同的均匀分布采样
        mask = (mask >= self.ratio).float()  # 大于等于ratio的位置置为1，小于ratio的位置置为0
        masked_x = x *self.s_up* mask + x* self.s_down * (1 - mask)  # 对75%的像素乘以p，剩下的像素保持不变

        out = masked_x  
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)                     
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out

# random DIP
class random_DIP(Full_DIP_backbone):
    def __init__(self, param_scale, 
                config, suffix):
        self.sigma_p = config['sigma_p']
        super().__init__(param_scale, 
                 config, suffix)
        
    def forward(self,x):
        out = torch.distributions.Uniform(low=0,high=1).sample([1,1,128,128])    
        for i in range(len(self.encoder_layers)):
            out = self.encoder_layers[i](out)                    
        for i in range(len(self.decoder_layers)):
            out = self.decoder_layers[i](out)
        out = self.positivity(out)
        return out   

# two different ways to skip connections 
class DIP_skip_concat(Full_DIP_backbone):                       
    def __init__(self,param_scale,config,suffix):  
        self.sigma_p = config['sigma_p']                          
        super().__init__(param_scale,config,suffix)      
    
    def initialize_network(self):

        L_relu = 0.2
        num_channel =self.num_channels # 16 32 64 128 
        pad = [0,0]
        self.deep1 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(1, num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.down1 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[0], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu))

        self.deep2 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.down2 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.deep3 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.down3 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], 3, stride=(2, 2), padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.deep4 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[3]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[3], num_channel[3], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[3]),
                                   nn.LeakyReLU(L_relu))

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[3], num_channel[2], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[2]),
                                 nn.LeakyReLU(L_relu))

        self.deep5 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2]+num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[2], num_channel[2], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[2]),
                                   nn.LeakyReLU(L_relu))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[2], num_channel[1], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[1]),
                                 nn.LeakyReLU(L_relu))

        self.deep6 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1]+num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[1], num_channel[1], (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(num_channel[1]),
                                   nn.LeakyReLU(L_relu))

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                                 nn.ReplicationPad2d(1),
                                 nn.Conv2d(num_channel[1], num_channel[0], 3, stride=(1, 1), padding=pad[0]),
                                 nn.BatchNorm2d(num_channel[0]),
                                 nn.LeakyReLU(L_relu))

        self.deep7 = nn.Sequential(nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0]+num_channel[0], num_channel[0], (3, 3), stride=1, padding=pad[0]),
                                   nn.BatchNorm2d(num_channel[0]),
                                   nn.LeakyReLU(L_relu),
                                   nn.ReplicationPad2d(1),
                                   nn.Conv2d(num_channel[0], 1, (3, 3), stride=1, padding=pad[1]),
                                   nn.BatchNorm2d(1))
    
        self.positivity = nn.ReLU()
        
    def forward(self, x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
         # Encoder
        out1 = self.deep1(out)
        out = self.down1(out1)
        out2 = self.deep2(out)
        out = self.down2(out2)
        out3 = self.deep3(out)
        out = self.down3(out3)
        out = self.deep4(out)

        # Decoder
        out = self.up1(out)
        out_skip1 = torch.cat([out3,out],dim=1)
        out = self.deep5(out_skip1)
        
        out = self.up2(out)
        out_skip2 = torch.cat([out2,out],dim=1)
        out = self.deep6(out_skip2)
        

        out = self.up3(out)
        out_skip3 =torch.cat([out1,out],dim=1) 
        out = self.deep7(out_skip3)

        out = self.positivity(out)

        return out
# skip_cnn 
class DIP_skip_add(Full_DIP_backbone):   
                        
    def __init__(self,param_scale,config,suffix):      
        self.sigma_p = config['sigma_p']                      
        self.skip = config['skip']  
        self.kernel_size = config['kernel_size']
        self.mode = config['mode']
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        super().__init__(param_scale,config,suffix)    
        
    def initialize_network(self):
        self.unetskipadd = UnetSkipAdd(1,self.embed_dim,1,
                                       kernel_size=self.kernel_size,
                                       skip=self.skip,
                                       num_layer=self.num_layers,
                                       depths=self.depths,
                                       mode=self.mode)
        # init_weights(self.unetskipadd,init_type='kaiming')
        self.positivity = nn.ReLU()
    def forward(self, x):
        out = self.unetskipadd(x)
        out = self.positivity(out)
        return out


class Swin_Unetr(pl.LightningModule):

    def __init__(self, param_scale, 
                 config, suffix):
        super().__init__()
        # random_seed = 114514
        # pl.seed_everything(random_seed)
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.iter_DIP = config['iters']
        self.param = param_scale 
        self.path="/home/xzhang/Documents/simplified_pipeline/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.embed_dim = config['embed_dim']
        self.use_v2 = config['use_v2']
        self.sigma_p = config['sigma_p']
        self.initialize_network()
        # for m in self.modules():
		# # 判断是否属于Conv2d
        #     if isinstance(m, nn.Conv2d):
        #         if self.initial_param == 'xavier_norm':
        #             torch.nn.init.xavier_normal_(m.weight.data)
        #         elif self.initial_param == 'xavier_uniform':
        #             torch.nn.init.xavier_uniform_(m.weight.data)
        #         elif self.initial_param == 'kaiming_norm':
        #             torch.nn.init.kaiming_normal_(m.weight.data)
        #         elif self.initial_param == 'kaiming_uniform':
        #             torch.nn.init.kaiming_uniform_(m.weight.data)
        
    def initialize_network(self):

        self.swin_unetr = SwinUNETR(img_size=(128,128),in_channels=1,out_channels=1,
                                    depths=self.depths,#(2,2,2,2),
                                    num_heads=self.num_heads,#(3,6,12,24),
                                    feature_size=self.embed_dim ,#32
                                    norm_name='instance',
                                    drop_rate=0.,attn_drop_rate=0.,dropout_path_rate=0.,normalize=False,use_checkpoint=False,spatial_dims=2,downsample='merging',
                                    use_v2=False)
        
        self.positivity = nn.ReLU() 
        
    def forward(self, x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
        out=self.swin_unetr(out)
        # print(out.shape)
        out=self.positivity(out)
        return out


    
class Swin_Unet(pl.LightningModule):

    def __init__(self, param_scale, 
                 config, suffix):
        super().__init__()
        # random_seed = 114514
        # pl.seed_everything(random_seed)
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.iter_DIP = config['iters']
        self.param = param_scale 
        self.path="/home/xzhang/Documents/simplified_pipeline/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.embed_dim = config['embed_dim']
        self.use_v2 = config['use_v2']
        self.sigma_p = config['sigma_p']
        self.initialize_network()
        # for m in self.modules():
		# # 判断是否属于Conv2d
        #     if isinstance(m, nn.Conv2d):
        #         if self.initial_param == 'xavier_norm':
        #             torch.nn.init.xavier_normal_(m.weight.data)
        #         elif self.initial_param == 'xavier_uniform':
        #             torch.nn.init.xavier_uniform_(m.weight.data)
        #         elif self.initial_param == 'kaiming_norm':
        #             torch.nn.init.kaiming_normal_(m.weight.data)
        #         elif self.initial_param == 'kaiming_uniform':
        #             torch.nn.init.kaiming_uniform_(m.weight.data)
        
    def initialize_network(self):

        self.swin_unet = SwinTransformerSys(img_size=128, patch_size=4, in_chans=1, num_classes=1,
                 embed_dim=self.embed_dim, depths=self.depths, depths_decoder=[1, 2, 2, 2], num_heads=self.num_heads,
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="Dual up-sample")
        # self.positivity = nn.LeakyReLU(0.01)#nn.ReLU() 
        
    def forward(self, x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
            
        out=self.swin_unet(out)
        # print(out.shape)
        # out=self.positivity(out)
        return out


    
class Swin_IR(pl.LightningModule):

    def __init__(self, param_scale, 
                 config, suffix):
        super().__init__()
        # random_seed = 114514
        # pl.seed_everything(random_seed)
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.iter_DIP = config['iters']
        self.param = param_scale 
        self.path="/home/xzhang/Documents/simplified_pipeline/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.embed_dim = config['embed_dim']
        self.use_v2 = config['use_v2']
        self.sigma_p = config['sigma_p']
        self.initialize_network()
        # for m in self.modules():
		# # 判断是否属于Conv2d
        #     if isinstance(m, nn.Conv2d):
        #         if self.initial_param == 'xavier_norm':
        #             torch.nn.init.xavier_normal_(m.weight.data)
        #         elif self.initial_param == 'xavier_uniform':
        #             torch.nn.init.xavier_uniform_(m.weight.data)
        #         elif self.initial_param == 'kaiming_norm':
        #             torch.nn.init.kaiming_normal_(m.weight.data)
        #         elif self.initial_param == 'kaiming_uniform':
        #             torch.nn.init.kaiming_uniform_(m.weight.data)
        
    def initialize_network(self):

        self.swinIR = SwinIR(img_size=128, patch_size=4, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='nearest+conv', resi_connection='3conv')
        
    
        # self.positivity = nn.LeakyReLU(0.01)#nn.ReLU() 
        
    def forward(self, x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
            
        out=self.swinIR(out)
        # print(out.shape)
        # out=self.positivity(out)
        return out


class restormer(pl.LightningModule):

    def __init__(self, param_scale, 
                 config, suffix):
        super().__init__()
        # random_seed = 114514
        # pl.seed_everything(random_seed)
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.iter_DIP = config['iters']
        self.param = param_scale 
        self.path="/home/xzhang/Documents/simplified_pipeline/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.embed_dim = config['embed_dim']
        self.use_v2 = config['use_v2']
        self.sigma_p = config['sigma_p']
        self.initialize_network()
        # for m in self.modules():
		# # 判断是否属于Conv2d
        #     if isinstance(m, nn.Conv2d):
        #         if self.initial_param == 'xavier_norm':
        #             torch.nn.init.xavier_normal_(m.weight.data)
        #         elif self.initial_param == 'xavier_uniform':
        #             torch.nn.init.xavier_uniform_(m.weight.data)
        #         elif self.initial_param == 'kaiming_norm':
        #             torch.nn.init.kaiming_normal_(m.weight.data)
        #         elif self.initial_param == 'kaiming_uniform':
        #             torch.nn.init.kaiming_uniform_(m.weight.data)
        
    def initialize_network(self):

        self.restormer = Restormer( 
        inp_channels=1, 
        out_channels=1, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    )
    
        # self.positivity = nn.LeakyReLU(0.01)#nn.ReLU() 
        
    def forward(self, x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
            
        out=self.restormer(out)
        # print(out.shape)
        # out=self.positivity(out)
        return out


  

class spectformer(pl.LightningModule):

    def __init__(self, param_scale, 
                 config, suffix):
        super().__init__()
        # random_seed = 114514
        # pl.seed_everything(random_seed)
        # 训练参数，学习率，迭代次数，图片处理参数，图片储存位置， 图片储存名字， 图片训练次数
        self.lr = config['lr']
        self.iter_DIP = config['iters']
        self.param = param_scale 
        self.path="/home/xzhang/Documents/simplified_pipeline/data/results/images/"   
        self.suffix  = suffix
        self.repeat = config['repeat']
        
        # 网络参数相关,定义model类型， layer数量，以及每层layers的channels数量（列表形式）
        self.config = config
        self.model_name = config['model_name']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.embed_dim = config['embed_dim']
        self.use_v2 = config['use_v2']
        self.sigma_p = config['sigma_p']
        self.initialize_network()
        # for m in self.modules():
		# # 判断是否属于Conv2d
        #     if isinstance(m, nn.Conv2d):
        #         if self.initial_param == 'xavier_norm':
        #             torch.nn.init.xavier_normal_(m.weight.data)
        #         elif self.initial_param == 'xavier_uniform':
        #             torch.nn.init.xavier_uniform_(m.weight.data)
        #         elif self.initial_param == 'kaiming_norm':
        #             torch.nn.init.kaiming_normal_(m.weight.data)
        #         elif self.initial_param == 'kaiming_uniform':
        #             torch.nn.init.kaiming_uniform_(m.weight.data)
        
    def initialize_network(self):

        self.restormer = SpectFormer(img_size=128, patch_size=16, in_chans=1, num_classes=1, embed_dim=128, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 dropcls=0)
    
        # self.positivity = nn.LeakyReLU(0.01)#nn.ReLU() 
        
    def forward(self, x):
        if self.sigma_p == 0 :
            out = x 
        else:                     
            noise = torch.distributions.Uniform(low=0,high=self.sigma_p).sample([*x.size()])
            # noise = torch.distributions.Normal(loc=0,scale=self.sigma_p).sample([*x.size()])
            out = x + noise
            
        out=self.restormer(out)
        # print(out.shape)
        # out=self.positivity(out)
        return out

  
    
      

class BaggedDIPAverage(DIP_skip_add):
    def __init__(self, num_models=50,*param, **model_hyperparameters):
        super().__init__(*param,**model_hyperparameters)
        self.num_models = num_models
        self.models = nn.ModuleList([UnetSkipAdd(1,self.embed_dim,1,
                                       kernel_size=self.kernel_size,
                                       skip=self.skip,
                                       num_layer=self.num_layers,
                                       depths=self.depths,
                                       mode=self.mode) for _ in range(num_models)])
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # 对每个像素的输出进行平均
        averaged_output = torch.stack(outputs, dim=0).mean(dim=0)
        return averaged_output

      
    
    
    
    
    
    
    
    