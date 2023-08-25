from ray import tune

settings_config = {
    "random_seed" : False, # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
}

model_config = {
    'model_name': 'bagging_dip',
    'num_layers': 3,#tune.grid_search([3]),#tune.grid_search([3]),
    'num_channels': 'exponential',#tune.grid_search(['exponential','equal':),
    'upsampling_mode': 'bilinear',#'bilinear',#tune.grid_search(['deconv','gaussian']),
    'ln_lambda': 0,#tune.grid_search([0,1,2,3]),
    'sigma': 0,#tune.grid_search([0.1,0.5]),
    'sigma_p': 0,# tune.grid_search([0.1,0.5,1]),
    'init': '0',#tune.grid_search(['kaiming_uniform','kaiming_norm','xavier_norm','xavier_uniform']), 
    'skip': 3,#tune.grid_search([4]),
    'kernel_size': 3,#tune.grid_search([3]),
    'mode': 'bilinear',
    'embed_dim': 16,# tune.grid_search([32]),
    'depths': 2,#tune.grid_search([2,3]),
} 

# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "lr" : 0.01, # Learning rate in network optimization
    "iters" : 800, # Number of epochs in network optimization
}
# Merge 3 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys()),
    "repeat" : tune.grid_search(list(range(50))),
}


config_DIP_skip = {**settings_config, **model_config,**hyperparameters_config, **split_config}
