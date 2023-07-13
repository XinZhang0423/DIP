import matplotlib.pyplot as plt
import numpy as np

# image = np.load("/home/xzhang/Documents/simplified_pipeline/data/results/images/noise_test/<class 'model.modules.Full_DIP_noise_v3'>/3_128_1/train_3/iters_400.npy")

# image = np.max(image)-image

# plt.imshow(image,cmap='gray')
# plt.show()


import csv 
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv('/home/xzhang/Documents/simplified_pipeline/models_metrics_v0123.csv').iloc[:,1:]
print(df.head())
grouped_df = df.groupby(['model', 'num_layers', 'num_channels',  'sigma_p','train']).agg({
    'loss': 'min',
    'mse': 'min',
    'psnr': 'max',
    'ssim': 'max'
}).reset_index()
print(grouped_df.head())
data = grouped_df
data = data.drop(data[data['model']=='Full_DIP_noise_v0'].index)
data = data.drop(data[data['model']=='Full_DIP_noise_v3'].index)
sns.catplot(x='model', y='mse',row=None,hue = 'sigma_p',col= None, data=data, kind='boxen')
plt.suptitle('mse')
plt.show()

grouped_df_v2 = df.groupby(['model', 'num_layers', 'num_channels',  'sigma_p','train']).apply(lambda x: x.loc[x['mse'].idxmin(),['iteration','mse']]).reset_index()
grouped_df_v2.columns = ['model', 'num_layers', 'num_channels',  'sigma_p','train','min_mse_iteration','min_mse']
grouped_df_v3 = grouped_df_v2.groupby(['model', 'num_layers', 'num_channels',  'sigma_p']).apply(lambda x: pd.DataFrame({
                'best_mse' : [x.loc[x['min_mse'].idxmin(), 'min_mse']],
                'worst_mse' : [x.loc[x['min_mse'].idxmax(), 'min_mse']],
                'best_result_train': [x.loc[x['min_mse'].idxmin(), 'train']],
                'b_iteration': [x.loc[x['min_mse'].idxmin(), 'min_mse_iteration']],
                'worst_result_train': [x.loc[x['min_mse'].idxmax(), 'train']],
                'w_iteration': [x.loc[x['min_mse'].idxmax(), 'min_mse_iteration']]
            })).reset_index()

data = grouped_df_v3


# average image
grouped_df = grouped_df_v2.groupby(['model', 'num_layers','num_channels','sigma_p'])
def get_file_list(group):
    files = []
    for train, iteration in zip(group['train'], group['min_mse_iteration']):
        files.append(f'{train}/iters_{iteration}.npy')
    return files
# 应用函数并创建新的DataFrame
result_df = grouped_df.apply(get_file_list).reset_index(name='files')
print(result_df)
path_suffix = '/home/xzhang/Documents/simplified_pipeline/data/results/images/noise_test/'
for i,row in result_df.iterrows():
    average = np.zeros((128,128))
    for file in row['files']:
        file_name = path_suffix + f"{row['model']}/{row['num_layers']}_{row['num_channels']}_{row['sigma_p']}/{file}"
        average += np.load(file_name)
    average /= len(row['files'])
    np.save(file=path_suffix + f"{row['model']}/{row['num_layers']}_{row['num_channels']}_{row['sigma_p']}/average.npy",arr=average)
    
    
# delete average image
#plot image
fig, axs = plt.subplots(3, len(data), figsize=(5 * len(data), 15))
fig.subplots_adjust(hspace=0.3)
# 根据上面获得的最佳和最差图片，找到并且打印出来
# 遍历每一行
for i, row in data.iterrows():

    # 构造文件路径
    best_image_path = path_suffix + f"{row['model']}/{row['num_layers']}_{row['num_channels']}_{row['sigma_p']}/{row['best_result_train']}/iters_{row['b_iteration']}.npy"
    average_path = path_suffix + f"{row['model']}/{row['num_layers']}_{row['num_channels']}_{row['sigma_p']}/average.npy"
    worst_image_path = path_suffix + f"{row['model']}/{row['num_layers']}_{row['num_channels']}_{row['sigma_p']}/{row['worst_result_train']}/iters_{row['w_iteration']}.npy"

    best_image = np.load(best_image_path)
    average_image = np.load(average_path)
    worst_image = np.load(worst_image_path)
    # print(best_image.shape)
    # print(worst_image.shape)
    best_image = np.max(best_image) - best_image
    average_image = np.max(average_image)-average_image
    worst_image = np.max(worst_image) - worst_image


    axs[0,i].imshow(best_image, cmap='gray');
    axs[0,i].set_title(f"Best result from {row['model']}\nNum Layers: {row['num_layers']}\nNum Channels: {row['num_channels']}\nSigma: {row['sigma_p']}")
    axs[0,i].axis('off')
    
    axs[1,i].imshow(average_image, cmap='gray');
    axs[1,i].set_title(f"average_image from {row['model']}\nNum Layers: {row['num_layers']}\nNum Channels: {row['num_channels']}\nSigma: {row['sigma_p']}")
    axs[1,i].axis('off')
        
    axs[2,i].imshow(worst_image, cmap='gray');
    axs[2,i].set_title(f"Worst result from {row['model']}\nNum Layers: {row['num_layers']}\nNum Channels: {row['num_channels']}\nSigma: {row['sigma_p']}")
    axs[2,i].axis('off')
# 调整整个图像的布局和尺寸
plt.tight_layout()
plt.show();

for i,row in result_df.iterrows():
    try:
        # 删除文件
        os.remove(path_suffix + f"{row['model']}/{row['num_layers']}_{row['num_channels']}_{row['sigma_p']}/average.npy")
        print("文件删除成功！")
    except OSError as e:
        print("文件删除失败:", e)