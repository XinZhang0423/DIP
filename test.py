
import os
import random
import torch
import pytorch_lightning as pl
import numpy as np
import os
import logging
from functools import partial
import sys
sys.path.append("/home/xzhang/Documents/simplified_pipeline/")
from model.modules import *
from ray import tune
from config.config_Full_DIP_noise import *
from config.config_swin_unetr import *


def train_swin_unetr(config,model,path_target=  '/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded_brain.npy' ,# "/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
         suffix = 'noise_test'):
    logging.basicConfig(level=logging.WARNING)
    os.chdir("/home/xzhang/Documents/simplified_pipeline/")

    input_size = (128,128,1)
    
    # print(input_size)
    # image_net_input = np.random.normal(loc=0, scale=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    # image_net_input = np.random.uniform(low=0, high=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_net_input = np.load('/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_mr.npy')
    image_net_input_scaled = (image_net_input - np.min(image_net_input))/(np.max(image_net_input)-np.min(image_net_input))
    image_corrupt = np.load(path_target) 
    
    image_net_input_torch = torch.Tensor(image_net_input_scaled)#
    image_net_input_torch = image_net_input_torch.view(1,1,input_size[0],input_size[1],1)
    image_net_input_torch = image_net_input_torch[:,:,:,:,0]
    
    # 标准化
    param = np.max(image_corrupt)
    image_corrupt_input_scaled = image_corrupt/param
    image_corrupt_torch = torch.Tensor(image_corrupt_input_scaled)
    image_corrupt_torch = image_corrupt_torch.view(1,1,128,128,1)
    image_corrupt_torch = image_corrupt_torch[:,:,:,:,0]
  
     # 加载数据
    # train_dataset = torch.utils.data.TensorDataset(image_corrupt_torch,image_corrupt_torch)
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 

    # 加载模型
    # path = f'{suffix}/{str(model.__name__)+"_"+config["init"]+"_cookie"}/{num_layers}_{num_channels[-1]}_{config["sigma_p"]}'
    path = f'{suffix}/{config["model_name"]+"_mr"}/{config["embed_dim"]}_{config["depths"][0]}_{config["num_heads"][0]}_{config["sigma_p"]}'
    model = model(param,config,suffix=path)
    
    trainer = pl.Trainer(max_epochs=config["iters"], callbacks=[pl.callbacks.ProgressBar(refresh_rate=0)])
    # 训练模型
    trainer.fit(model, train_dataloader)
    

model = Swin_Unetr
tune.run(partial(train_swin_unetr,
            model=model,
            path_target="/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded_brain.npy",
            suffix ='swin_unetr_brain'),config = config_swin_unetr )