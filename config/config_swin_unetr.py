from ray import tune

settings_config = {
    "random_seed" : False, # If True, random seed is used for reproducibility (must be set to False to vary weights initialization)
}

        
model_config = {
    'model_name': 'res_2_3_v2',
    'depths': (2,2,2,2),#tune.grid_search([(2,2,2,2),(4,4,4,4)]),
    'num_heads': (3,6,12,24),#tune.grid_search([(3,6,12,24),(6,12,24,48)]),
    'embed_dim': 24,#tune.grid_search([48]),
    'use_v2': True,#tune.grid_search([True,False]),
    'sigma_p': 0,#tune.grid_search([0,0.1]),

}

# Configuration dictionnary for hyperparameters to tune
hyperparameters_config = {
    "lr" : 0.001, # Learning rate in network optimization
    "iters" : 400, # Number of epochs in network optimization
}
# Merge 3 dictionaries
split_config = {
    "hyperparameters" : list(hyperparameters_config.keys()),
    "repeat" :tune.grid_search(list(range(50))),
}


config_swin_unetr = {**settings_config,**model_config,**hyperparameters_config,**split_config}