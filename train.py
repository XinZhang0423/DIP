import os
import random
import torch
import pytorch_lightning as pl
import numpy as np
import os
import logging
from functools import partial
from model.modules import *
from ray import tune
from config.config_Full_DIP_noise import *
# corrupted_image = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy"))
# ground_truth = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_padded.npy"))
# mask = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/noisy_images/mask_padded.npy"))
# psnr_baseline = peak_signal_noise_ratio(ground_truth, corrupted_image, data_range=np.amax(corrupted_image)-np.amin(corrupted_image))
# mse_baseline = np.mean((corrupted_image - ground_truth)**2)
# print(config_sample)
# print(config_sample['benoulli'])
def train(config,model, 
         path_target=  '/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded_brain.npy' ,# "/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
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
    # image_net_input = np.random.normal(loc=0, scale=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_net_input = np.random.uniform(low=0, high=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    # image_net_input = np.load('/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_mr.npy')
    # image_net_input_scaled = (image_net_input - np.min(image_net_input))/(np.max(image_net_input)-np.min(image_net_input))
    image_corrupt = np.load(path_target) 
    
    image_net_input_torch = torch.Tensor(image_net_input)#image_net_input_scaled)#
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
    path = f'{suffix}/{str(model.__name__)}/{num_layers}_{num_channels[-1]}_{config["sigma_p"]}'
    model = model(param,config,suffix=path)
    
    trainer = pl.Trainer(max_epochs=config["iters"], callbacks=[pl.callbacks.ProgressBar(refresh_rate=0)])
    # 训练模型
    trainer.fit(model, train_dataloader)

def train_sample(config,model, 
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

    # image_net_input = np.random.uniform(low=0, high=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
    image_net_input = np.load('/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_mr.npy')
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
    # train_dataset = torch.utils.data.TensorDataset(image_corrupt_torch,image_corrupt_torch)
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 
    print(config['benoulli'])
    # 加载模型
    path = f'{suffix}/{str(model.__name__)}/{num_layers}_{num_channels[-1]}_{config["benoulli"]}_{config["s_down"]}_{config["s_up"]}'
    model = model(param,config,suffix=path)
    
    trainer = pl.Trainer(max_epochs=config["iters"], callbacks=[pl.callbacks.ProgressBar(refresh_rate=0)])
    # 训练模型
    trainer.fit(model, train_dataloader)



def train_mask(config,model, 
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

    image_net_input = np.random.uniform(low=0, high=1, size=input_size)# 7*7*1 因为做了4次上采样，也就是扩大了16倍
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
    # train_dataset = torch.utils.data.TensorDataset(image_corrupt_torch,image_corrupt_torch)
    train_dataset = torch.utils.data.TensorDataset(image_net_input_torch,image_corrupt_torch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1) 
    print(config['ratio'])
    # 加载模型
    # path = f'{suffix}/{str(model.__name__)}/{num_layers}_{num_channels[-1]}_{config["ratio"]}_{config["s_down"]}_{config["s_up"]}'
    path = f'{suffix}/{str(model.__name__)+"mr_input"}/{num_layers}_{num_channels[-1]}_{config["ratio"]}_{config["s_down"]}_{config["s_up"]}'
    model = model(param,config,suffix=path)
    
    trainer = pl.Trainer(max_epochs=config["iters"], callbacks=[pl.callbacks.ProgressBar(refresh_rate=0)])
    # 训练模型
    trainer.fit(model, train_dataloader)
    
models = [DIP_skip_concat,Full_DIP_noise_v0]
for model in models :
    tune.run(partial(train,
                model=model,
                path_target="/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
                suffix ='add_noise'),config = config_DIP_noise )
    
# tune.run(partial(train_sample,
#             model=sampled_DIP,
#             path_target="/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
#             suffix ='sampled_f'),config = config_sample )

# # tune.run(partial(train_mask,
# #             model=pw_masked_DIP,
# #             path_target="/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
# #             suffix ='pw_masked_f'),config = config_mask )

# tune.run(partial(train,
#             model=random_DIP,
#             path_target="/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy",
#             suffix ='random_input'),config = config_DIP_noise )


# 第一个实验，baseline测试，固定随机种子，分别对输入和权重进行随机，作为以后实验的baseline
# 重新测试不同的输入造成的影响， 分别对两种target images做50次不同实验,然后 分别画出unform, gaussian的区别，感觉没有区别，但是这是baseline数据哈，很有用的，以后都不用再跑一遍baseline了，因为这个就是最baseline

# 第一个改进方向，试图通过增加减少层数和通道数对baseline进行改进

# 第二个改进方向，1.试图去除encoder,2.试图使用deep decoder

# 第三个改进方向，1.试图添加LG,2.试图添加LG

# 第四个改进方向，1.试图添加noise在里面， 2.试图采用bernoulli采样

# 第五个改进方向，1. 试图增加residu connection 2, 分别测试skip connection(add,concatation)

