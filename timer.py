import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.cuda
from tqdm import tqdm
import numpy as np
from model.DIP.DIP_blocks import UnetSkipAdd
from model.swinUNETR.SwinUNetr import SwinUNETR

# model = UnetSkipAdd(1,16,1,3,3,3,2,'bilinear')
model = SwinUNETR((128,128),1,1,(2,2,2,2),(3,6,12,24),24)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model.to(device)

lossfunc = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
total_start_event = torch.cuda.Event(enable_timing=True)
total_end_event = torch.cuda.Event(enable_timing=True)
total_start_event.record()

image_net_input = np.load('/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_mr.npy')
image_net_input_scaled = (image_net_input - np.min(image_net_input))/(np.max(image_net_input)-np.min(image_net_input))
image_corrupt = np.load('/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded_brain.npy')

image_net_input_torch = torch.Tensor(image_net_input_scaled)#
image_net_input_torch = image_net_input_torch.view(1,1,128,128,1)
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


# 创建数据加载器

for epoch in range(num_epochs):
    model.train()
    
    with tqdm(total=len(train_dataloader),desc=f'Epoch [{epoch+1}/{num_epochs}]',unit='batch') as pbar:
        for batch_x,batch_y in train_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            
            loss = lossfunc(outputs,batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            pbar.set_postfix({'Loss':loss// (pbar.n // 1 + 1)})
            
total_end_event.record()
total_end_event.synchronize()
total_training_time_ms = total_start_event.elapsed_time(total_end_event)
print(f'Total Training Time: {total_training_time_ms:.2f} ms')
print("Training finished.")

# DIP  1000 27217.67 ms
# swinUNetr 1000 78689.38 ms add  84425.56 ms concat