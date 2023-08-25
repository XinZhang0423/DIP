import os
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from matplotlib import pyplot as plt
import time
# from .fbc import get_circular_statastic
from torch import bernoulli

# 非常高级的多进程调用！！！
class parallelIO():
    """
    My files are stored in the format of :
    experiment_name/model_name/config_setting/trains/iters.npy for example:
    skip_connection/DIP_add/3_16_0/train_0/iters_9.npy means the test to study the influence of different num_layers 3, embedding_dims 16, num_skip_connections 0
    The basic idea is to create a pool with the same number of processes as the number of settings. (3_16_0,3_16_1,3_16_2,3_16_3)
    Each 'train' within a setting corresponds to a separate thread for processing. for example, 50 trains (50 different random seed)
    """
    
    def __init__(self,root_path,models,columns,metrics,corrupted_image,ground_truth,mask):
        """
            _summary_

        Args:
            root_path (string): the path to the folder of experiment_name
            models (list of string): the list of models' names to be used
            columns (list of string): the setting names to be used
            metrics (list of string): metric names to be used
            corrupted_image (np.array):  target image to compute the loss
            ground_truth (np.array):  to compute mse,psnr,ssim
            mask (np.array): for phantom to calculate mse
            
        """
        self.root_path = root_path
        self.models = models
        self.columns = columns
        self.metrics = metrics
        self.ci = corrupted_image
        self.gt = ground_truth
        self.msk =mask
    
    def models_loop(self):
        # Define the main loop
        self.results =list()
        for model in self.models:
            
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
        # need to define manually the settings 
        # num_layers = test.split("_")[0]
        # num_channels = test.split("_")[1]
        # sigma_p = test.split("_")[2]
        embed_dims = test.split("_")[0]
        depth = test.split("_")[1]
        num_heads = test.split("_")[2]
        sigma_p = test.split("_")[3]
        # setting =[model,num_layers,num_channels,sigma_p]
        setting =[model,embed_dims,depth,num_heads,sigma_p]
        
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
                *setting, 
                train,iteration, res_loss, res_mse, res_psnr, res_ssim
            ]

            iter_res = pd.DataFrame([row], columns=self.columns)
            train_res_list.append(iter_res)
        train_res = pd.concat(train_res_list,axis=0)
        return train_res
    
class parallelIO_sample(parallelIO):
    """
    Subclass of ParallelIO for a specific use case.
    """
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
            execution = end - start 
            print(f"{test} finished", execution, "s")
            return test_res
        
        
class parallelIO_brain(parallelIO):    
    """
    Subclass of ParallelIO for a different use case. need to remove the mask for mse calculation
    """
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
                elif metric == 'FBC':
                    res_FBC = get_circular_statastic(image_np,self.gt)
            row = [
                *setting,
                train,iteration, res_loss, res_mse, res_psnr, res_ssim
            ]
            if 'FBC' in metrics:
                row = row.extend(res_FBC)
            iter_res = pd.DataFrame([row], columns=self.columns)
            train_res_list.append(iter_res)
        train_res = pd.concat(train_res_list,axis=0)
        return train_res
   

    
if __name__=='__main__':
    
    # corrupted_image = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded.npy"))
    # ground_truth = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_padded.npy"))
    
    corrupted_image = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/corrupted_images/target_padded_brain.npy"))
    ground_truth = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_brain.npy"))
    
    mask = np.squeeze(np.load("/home/xzhang/Documents/我的模型/data/noisy_images/mask_padded.npy"))
    root_path = '/home/xzhang/Documents/simplified_pipeline/data/results/images/dip_bagging' # to_do
    # columns =  [
    # 'model', 'embed_dims', 'depths', 'num_heads','sigma_p','train',
    # 'iteration','loss', 'mse', 'psnr', 'ssim'
    # ]
    columns =  [
    'model', 'num_layers', 'embed_dims', 'depths','skip','train',
    'iteration','loss', 'mse', 'psnr', 'ssim'
    ] # to_do
    # columns =  [
    # 'model', 'num_layers', 'num_channels','skip','sigma_p','train',
    # 'iteration','loss', 'mse', 'psnr', 'ssim'
    # ]
    
    # columns =  [
    # 'model', 'num_layers', 'num_channels', 'ratio_p','s_down','s_up','train',
    # 'iteration','loss', 'mse', 'psnr', 'ssim'
    # ]
    models = os.listdir(root_path)#['Deep_decoder']#'Deep_decoder',,'Full_DIP','DIP_decoder'
    metrics = ['loss', 'mse', 'psnr', 'ssim']
    
    start = time.time()
    # io = parallelIO(root_path=root_path,columns=columns,models=models,metrics=metrics,corrupted_image=corrupted_image,ground_truth=ground_truth,mask=mask)
    print('start calculation')
    io = parallelIO_brain(root_path=root_path,columns=columns,models=models,metrics=metrics,corrupted_image=corrupted_image,ground_truth=ground_truth,mask=mask)
    print('start calculation')
    io.models_loop()
    end = time.time()
    df = io('test_swin_unetr_v2.csv') # name of the file
    execution = end - start  
    print("time for calculation", execution, "s")

    # image = np.load('/home/xzhang/Documents/我的模型/data/results/images/models_test_new/Deep_decoder/3_128_exponential_bilinear_0_0/train_7/iters_27.npy')
    # image = np.max(image) - image
    # plt.imshow(image,cmap='gray_r')