from ray import tune

settings_config = {
    "random_seed" : False, # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
}

model_config = {
    'model_name': 'Full_DIP',
    'num_layers': 3,#tune.grid_search([3]),#tune.grid_search([3]),
    'num_channels': 'exponential',#tune.grid_search(['exponential','equal']),
    'upsampling_mode': 'bilinear',#'bilinear',#tune.grid_search(['deconv','gaussian']),
    'ln_lambda': 0,#tune.grid_search([0,1,2,3]),
    'sigma': 0,#tune.grid_search([0.1,0.5]),
    'sigma_p': 1,#tune.grid_search([1]),
    'init': '0',#tune.grid_search(['kaiming_uniform','kaiming_norm','xavier_norm','xavier_uniform']),
}

# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "lr" : 0.01, # Learning rate in network optimization
    "iters" : 1000, # Number of epochs in network optimization
}
# Merge 3 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys()),
    "repeat" : tune.grid_search(list(range(5))),
}


config_DIP_noise = {**settings_config, **model_config,**hyperparameters_config, **split_config}

sample ={
    
    'benoulli' : 0.1,#tune.grid_search([0.1,0.5,0.9]),
    's_down' : 0.1,#tune.grid_search([0.1,0.2,0.5]),
    's_up' : 1,#tune.grid_search([1,1.5,2]),
}

config_sample = {**config_DIP_noise,**sample}


mask ={
    
    'ratio' : 0.5,#tune.grid_search([0.1,0.5,0.9]),
    's_down' : tune.grid_search([0.5,1]),
    's_up' : tune.grid_search([1,1.5]),
}

config_mask= {**config_DIP_noise,**mask}