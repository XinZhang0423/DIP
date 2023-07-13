import os
import torch
import pytorch_lightning as pl
import numpy as np
import os
import logging
from functools import partial
from model.modules import *
from ray import tune
from config.config_Full_DIP_noise import config_DIP_noise

def train(config,model, 
         path_target="/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
         suffix = 'noise_test'):
    logging.basicConfig(level=logging.WARNING)
    os.chdir("/home/xzhang/Documents/simplified_pipeline/")
    num_layers = config['num_layers']
    num_channels_type = config['num_channels']
    
    if num_channels_type == 'exponential':
        num_channels = [int(2**(4+i)) for i in range(num_layers+1)]
    elif num_channels_type == 'equal':
        num_channels = [config['nb_channels']] * (num_layers+1)
        
    config['num_channels'] = num_channels

    input_size = (128,128,1)
    
    # print(input_size)

    image_net_input = np.load('/home/xzhang/Documents/我的模型/data/corrupted_images/uniform_noise_0.npy')#random.uniform(low=0, high=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_corrupt = np.load(path_target) 
    
    image_net_input_torch = torch.Tensor(image_net_input)
    image_net_input_torch = image_net_input_torch.view(1,1,input_size[0],input_size[1],1)
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    # 标准化
    param = np.max(image_corrupt)
    image_corrupt_input_scaled = image_corrupt/param
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,128,128,1)
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
  
     # 加载数据
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 
    path_target_name = path_target.split('/')[-1].split('.')[0]
    # 加载模型
    path = f'{suffix}/{path_target_name}/{num_layers}_{num_channels[-1]}_{config["sigma_p"]}'
    model = model(param,config,suffix=path)
    
    trainer = pl.Trainer(max_epochs=config["iters"], callbacks=[pl.callbacks.ProgressBar(refresh_rate=0)])
    # 训练模型
    trainer.fit(model, train_dataloader)
suffix = '/home/xzhang/Documents/我的模型/data/corrupted_images/'

path_targets = ['target_padded.npy','gaussian_noise.npy','uniform_noise.npy','ground_truth_padded.npy']
for path_target in path_targets:
    path_target = suffix+path_target
    tune.run(partial(train,
            model=Full_DIP_noise_v0,
            path_target=path_target,
            suffix ='noise_theory'),config = config_DIP_noise )
