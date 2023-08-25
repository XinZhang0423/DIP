
import os
import numpy as np
import pandas as pd

# Frequency band metric
def get_circular_statastic(img_it, img_gt, size=0.2):


    assert(size>0 and size<1)

    ftimage_it = np.fft.fft2(img_it)
    ftimage_it = abs(np.fft.fftshift(ftimage_it))

    ftimage_gt = np.fft.fft2(img_gt)
    ftimage_gt = abs(np.fft.fftshift(ftimage_gt))

    m_data = ftimage_it/(ftimage_gt+1e-8)
    m_data = np.clip(m_data, 0, 1)

    h,w = m_data.shape

    center = (int(w/2), int(h/2))
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    avg_mask_list = []
    pre_mask = np.zeros((h,w))
    for sz in np.linspace(size, 1, int(1/size)):

        radius = center[0]*sz#pow(center[0]**2+center[1]**2,0.5)
        mask = dist_from_center <= radius
        mask = mask.astype(np.int32)

        mask_sz = (mask-pre_mask).astype(np.int32)
        pre_mask = mask

        avg_mask_list.append(np.sum(mask_sz*m_data)/np.sum(mask_sz))

    return avg_mask_list


def fbc_to_csv(iters_path,target_filename,gt= np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_brain.npy")[:,:,0]):
    train_res_list = list()
    iters = sorted(os.listdir(iters_path), key=lambda x: int(x.split(".")[0].split("_")[1]))
    for iter in iters:
        image_path = os.path.join(iters_path, iter)
        image_np = np.load(image_path, allow_pickle=True)
        iteration = iter.split(".")[0].split("_")[1]

        res_FBC = get_circular_statastic(image_np,gt)

        row = [iteration,*res_FBC]
        iter_res = pd.DataFrame([row], columns=['iteration','lowest','low','medium','high','highest'])
        train_res_list.append(iter_res)
    train_res = pd.concat(train_res_list,axis=0)
    train_res.to_csv(target_filename)
    
# img_it = np.load("/home/xzhang/Documents/simplified_pipeline/data/results/images/swin_unetr_brain/Swin_Unetr_pre/48_2_3_0/average.npy")
# img_gt = np.load("/home/xzhang/Documents/我的模型/data/ground_truth/ground_truth_brain.npy")[:,:,0]

# avg_mask_list = get_circular_statastic(img_it, img_gt, size=0.2)
# print(avg_mask_list)
fbc_to_csv("/home/xzhang/Documents/simplified_pipeline/data/results/images/baseline_cookie/Full_DIP_backbone_normal/3_128_0/train_19",
         "/home/xzhang/Documents/simplified_pipeline/data/results/images/baseline_cookie/fbc.csv"  )