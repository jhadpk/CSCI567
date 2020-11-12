### DO NOT CHANGE ANYTHING IN THIS FILE ###

if __name__ == "__main__":
    import neural_networks
    config = {
        "random_seed": 42,
        "minibatch_size": 5,
        "input_file": 'mnist_subset.json',
        "num_epoch": 3,
        "dropout_rate": 0.0,
        "alpha": 0.0,
        "activation": "relu",
        "learning_rate": 0.01
    }
    import copy

    temp_config = copy.deepcopy(config)
    neural_networks.main(temp_config)
            
    temp_config["activation"] = "tanh"
    neural_networks.main(temp_config)
            
    temp_config = copy.deepcopy(config)
    temp_config["dropout_rate"] = 0.5
    temp_config["alpha"] = 0.9
    neural_networks.main(temp_config)

    temp_config = copy.deepcopy(config)
    temp_config["dropout_rate"] = 0.25
    temp_config["alpha"] = 0.9
    temp_config["activation"] = "tanh"
    neural_networks.main(temp_config)