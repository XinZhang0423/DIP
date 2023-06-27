
from ray import tune
'''
    config 主要分为三大类:
    1. config_dd
    2. config_dip
    3. config_my_dip_v1

'''



def config_set(model_name='Deep_decoder',random_seed=False,num_layers=3,num_channels='equal',nb_channels='128',
                  upsampling_mode='bilinear',ln_lambda=0,sigma=0,lr=0.01,sub_iter_DIP=800,repeat=tune.grid_search(list(range(50)))):
    config =  {
    "random_seed" : random_seed,
    'model_name': model_name,
    
    "lr" : lr,
    "sub_iter_DIP" : sub_iter_DIP, # Number of epochs in network optimization
    "repeat" :tune.repeat(list(range(50))),
    
    'num_layers': num_layers,
    'num_channels': num_channels,
    'nb_channels' : nb_channels,
    'upsampling_mode':upsampling_mode,
    'ln_lambda': ln_lambda,
    'sigma': sigma,#tune.grid_search([0.1,0.5]),

    }
    return config

def config_set_dip(model_name='My_DIP_ED',random_seed=False,num_up_layers=3,num_down_layers=3,num_channels='equal',add_noise=False,
                  upsampling_mode='bilinear',ln_lambda=0,sigma=0,lr=0.01,sub_iter_DIP=800,repeat=tune.grid_search(list(range(50)))):
    config_dip =  {
    "random_seed" : random_seed,
    'model_name': model_name,
    
    "lr" : lr,
    "sub_iter_DIP" : sub_iter_DIP, # Number of epochs in network optimization
    "repeat" :tune.repeat(list(range(50))),

    'num_up_layers':num_up_layers,#tune.grid_search([3]),
    'num_down_layers': num_down_layers,#tune.grid_search([3]),
    'num_channels': num_channels,#tune.grid_search(['exponential','equal']),
    'upsampling_mode': upsampling_mode,#tune.grid_search(['deconv','gaussian']),
    'add_noise' : add_noise,
    'ln_lambda': ln_lambda,#tune.grid_search([0,1,2,3]),
    'sigma': sigma,#tune.grid_search([0.1,0.5]),

    }
    return config_dip