import os
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plt
import time

from torch import bernoulli

# 非常高级的多进程调用！！！
class parallelIO():
    """
        基本思路是先创建一个和setting数量一样多的pool,每一个setting中的train都对应哪个一个thread即可
    
    """
    
    def __init__(self,root_path,models,columns,metrics,corrupted_image,ground_truth,mask):
        """
            _summary_

        Args:
            root_path (_type_): 文件根路径
            models (_type_): 需要计算的模型名列表
            columns (_type_): 需要保存的列名
            metrics (_type_): 需要计算的指标
            corrupted_image (_type_): np.array 目标图片
            ground_truth (_type_): np.array 真实图片
            mask (_type_): np.array 计算时所需要的图片
            
        """
        self.root_path = root_path
        self.models = models
        self.columns = columns
        self.metrics = metrics
        self.ci = corrupted_image
        self.gt = ground_truth
        self.msk =mask
    
    def models_loop(self):
        # 定义主循环
        self.results =list()
        for model in self.models:
            # 进入并读取每个model的文件夹
            tests_path = os.path.join(self.root_path,model)
            tests = sorted(os.listdir(tests_path), key=lambda x: int(x.split('_')[0]))
            
            with mp.Pool(processes=64) as pool:
                print(f'start calculation for {model}')
                tests_res = pool.map(partial(self.test_loop,tests_path=tests_path, model=model), tests)
                model_res = pd.concat(tests_res,axis=0)
                self.results.append(model_res)
    

    def __call__(self,filename):
        df= pd.concat(self.results,axis=0)
        path= os.path.join(self.root_path,filename)
        df.to_csv(path)
        return df

    
    def test_loop(self,test,tests_path,model):
        trains_path = os.path.join(tests_path, test)
        trains = sorted(os.listdir(trains_path), key=lambda x: int(x.split('_')[1]))
        num_layers = test.split("_")[0]
        num_channels = test.split("_")[1]
        sigma_p = test.split("_")[2]
        setting =[model,num_layers,num_channels,sigma_p]
        with ThreadPoolExecutor(max_workers=50) as executor:
            print(f'start calculation for {test}')
            start = time.time()
            trains_res = executor.map(partial(self.iter_loop,trains_path=trains_path,setting=setting), trains)
            test_res = pd.concat(trains_res,axis=0)
            end = time.time()
            execution = end - start  # 计算执行时间（秒）
            print(f"{test} finished", execution, "秒")
            return test_res
        
    def iter_loop(self,train,trains_path,setting):
        iters_path = os.path.join(trains_path,train)
        iters = sorted(os.listdir(iters_path), key=lambda x: int(x.split(".")[0].split("_")[1]))
        train_res_list = list()
        print(f'calculating for {setting} {train}')
        for iter in iters:
            image_path = os.path.join(iters_path, iter)
            image_np = np.load(image_path, allow_pickle=True)
            iteration = iter.split(".")[0].split("_")[1]
            for metric in self.metrics:
                if metric == 'loss':
                    res_loss = np.mean((self.ci - image_np) ** 2)
                elif metric == 'mse':
                    res_mse = np.mean((self.gt * self.msk - image_np * self.msk) ** 2)
                elif metric == 'psnr':
                    res_psnr = peak_signal_noise_ratio(self.gt, image_np, data_range=np.amax(self.gt) - np.amin(self.gt))
                elif metric == 'ssim':
                    res_ssim = structural_similarity(self.gt, image_np, data_range=np.amax(self.gt) - np.amin(self.gt))
            
            row = [
                *setting,#num_layers, num_channels, channels_type, upsample_mode, ln_lambda, sigma, 
                train,iteration, res_loss, res_mse, res_psnr, res_ssim
            ]

            iter_res = pd.DataFrame([row], columns=self.columns)
            train_res_list.append(iter_res)
        train_res = pd.concat(train_res_list,axis=0)
        return train_res
    
class parallelIO_sample(parallelIO):

    def test_loop(self,test,tests_path,model):
        trains_path = os.path.join(tests_path, test)
        trains = sorted(os.listdir(trains_path), key=lambda x: int(x.split('_')[1]))
        num_layers = test.split("_")[0]
        num_channels = test.split("_")[1]
        p = test.split("_")[2]
        s_down = test.split("_")[3]
        s_up = test.split("_")[4]
        setting =[model,num_layers,num_channels,p,s_down,s_up]
        with ThreadPoolExecutor(max_workers=50) as executor:
            print(f'start calculation for {test}')
            start = time.time()
            trains_res = executor.map(partial(self.iter_loop,trains_path=trains_path,setting=setting), trains)
            test_res = pd.concat(trains_res,axis=0)
            end = time.time()
            execution = end - start  # 计算执行时间（秒）
            print(f"{test} finished", execution, "秒")
            return test_res
        
        
class parallelIO_brain(parallelIO):    
     
    def iter_loop(self,train,trains_path,setting):
        iters_path = os.path.join(trains_path,train)
        iters = sorted(os.listdir(iters_path), key=lambda x: int(x.split(".")[0].split("_")[1]))
        train_res_list = list()
        print(f'calculating for {setting} {train}')
        for iter in iters:
            image_path = os.path.join(iters_path, iter)
            image_np = np.load(image_path, allow_pickle=True)
            iteration = iter.split(".")[0].split("_")[1]
            for metric in self.metrics:
                if metric == 'loss':
                    res_loss = np.mean((self.ci - image_np) ** 2)
                elif metric == 'mse':
                    res_mse = np.mean((self.gt - image_np ) ** 2)
                elif metric == 'psnr':
                    res_psnr = peak_signal_noise_ratio(self.gt, image_np, data_range=np.amax(self.gt) - np.amin(self.gt))
                elif metric == 'ssim':
                    res_ssim = structural_similarity(self.gt, image_np, data_range=np.amax(self.gt) - np.amin(self.gt))
            
            row = [
                *setting,#num_layers, num_channels, channels_type, upsample_mode, ln_lambda, sigma, 
                train,iteration, res_loss, res_mse, res_psnr, res_ssim
            ]

            iter_res = pd.DataFrame([row], columns=self.columns)
            train_res_list.append(iter_res)
        train_res = pd.concat(train_res_list,axis=0)
        return train_res
   
    
    
if __name__=='__main__':
    corrupted_image = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy"))
    ground_truth = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_padded.npy"))
    
    # corrupted_image = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded_brain.npy"))
    # ground_truth = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_brain.npy"))
    
    mask = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/noisy_images/mask_padded.npy"))
    root_path = '/home/xzhang/Documents/simplified_pipeline/data/results/images/skip_cookie'
    columns =  [
    'model', 'num_layers', 'num_channels', 'sigma_p','train',
    'iteration','loss', 'mse', 'psnr', 'ssim'
    ]
    
    # columns =  [
    # 'model', 'num_layers', 'num_channels', 'ratio_p','s_down','s_up','train',
    # 'iteration','loss', 'mse', 'psnr', 'ssim'
    # ]
    models = os.listdir(root_path)#['Deep_decoder']#'Deep_decoder',,'Full_DIP','DIP_decoder'
    metrics = ['loss', 'mse', 'psnr', 'ssim']
    start = time.time()
    io = parallelIO(root_path=root_path,columns=columns,models=models,metrics=metrics,corrupted_image=corrupted_image,ground_truth=ground_truth,mask=mask)
    print('start calculation')
    # io = parallelIO_sample(root_path=root_path,columns=columns,models=models,metrics=metrics,corrupted_image=corrupted_image,ground_truth=ground_truth,mask=mask)
    # print('start calculation')
    io.models_loop()
    end = time.time()
    df = io('skip_cookie.csv')
    execution = end - start  # 计算执行时间（秒）
    print("程序执行时间：", execution, "秒")

    # image = np.load('/home/xzhang/Documents/我的模型/data/results/images/models_test_new/Deep_decoder/3_128_exponential_bilinear_0_0/train_7/iters_27.npy')
    # image = np.max(image) - image
    # plt.imshow(image,cmap='gray')