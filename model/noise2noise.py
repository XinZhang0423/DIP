import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange
# 由下采样和小巧的network组成

def pair_downsampler(noisy_image):
    # img has shape B C H W
    c = noisy_image.size()[1]
    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]])
    filter1 = filter1.repeat(c,1, 1, 1)
    
    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]])
    filter2 = filter2.repeat(c,1, 1, 1)
    
    output1 = F.conv2d(noisy_image, filter1, stride=2, groups=c)
    output2 = F.conv2d(noisy_image, filter2, stride=2, groups=c)
    
    return output1, output2

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(noisy_img,model):
    # print(noisy_img.size())
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)
    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))
    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))
    loss = loss_res + loss_cons

    return loss

class network(nn.Module):
    
    def __init__(self,n_channels,chan_embed = 48):
        super(network,self).__init__()
        
        self.conv1 = nn.Conv2d(n_channels,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed,chan_embed,3,padding=1)
        self.conv3 = nn.Conv2d(chan_embed,n_channels,1)
        self.act = nn.LeakyReLU(0.2,inplace=True)
        
    def forward(self,x):
        x= self.act(self.conv1(x))
        x= self.act(self.conv2(x))
        x= self.conv3(x)
        
        return x

def train(model,optimizer,noisy_image):
    loss = loss_func(noisy_image,model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def denoise(model,noisy_image):
    with torch.no_grad():
        pred  = noisy_image - model(noisy_image)
    return pred

if __name__ == '__main__':
    max_epoch = 3000
    lr = 0.001
    step_size = 1500
    gamma = 0.5
    model = network(n_channels=1,chan_embed=48)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
    
    noisy_image = np.load('/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy')
    ground_truth = np.load('/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_padded.npy')
    
    noisy_image = torch.from_numpy(noisy_image)

    # # 使用view函数重新排列张量的维度
    noisy_image = noisy_image.view(1, 1, 128, 128)
    # # print(noisy_image.size())
    # # 将noisy_image转化成tensor
    
    # # training
    for epoch in trange(max_epoch,desc='training process'):
        train(model,optimizer,noisy_image)
        scheduler.step()
        
    denoised_img = denoise(model, noisy_image)
    denoised = denoised_img.squeeze(0).permute(1,2,0).numpy()
    np.save('denoised.npy',denoised)
    noisy_image = noisy_image.view(128, 128).numpy()
    
    denoised = np.load('denoised.npy')
    noisy_image = np.max(noisy_image) - noisy_image
    ground_truth = np.max(ground_truth) - ground_truth
    
    print(denoised)
    denoised = np.max(denoised) - denoised
    # print(ground_truth)
    # print(noisy_image)
    
    fig, ax = plt.subplots(1, 3,figsize=(15, 15))
    ax[0].imshow(ground_truth,cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Ground Truth')

    ax[1].imshow(noisy_image,cmap='gray')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title('Noisy Img')
    
    ax[2].imshow(denoised,cmap='gray')
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    plt.show();