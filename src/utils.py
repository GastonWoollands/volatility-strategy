import yaml

#------------------------------------------------------------------------------------------------

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

#------------------------------------------------------------------------------------------------

def create_config_values():
    """Create a dictionary with the global configuration values."""
    global config_values

    config_values = {
        "model": {
            "input_size": globals().get('input_size'),
            "hidden_size": globals().get('hidden_size'),
            "output_size": globals().get('output_size'),
            "dropout": globals().get('dropout'),
            "seq_length": globals().get('seq_length'),
            "batch_size": globals().get('batch_size'),
            "seed": globals().get('seed', 83),
            "criterion": globals().get('criterion', "MSELoss"),
            "optimizer": globals().get('optimizer', "Adam"),
            "learning_rate": globals().get('learning_rate'),
            "epochs": globals().get('epochs'),
        },
        "data": {
            "start_date": globals().get('start_date'),
            "end_date": globals().get('end_date'),
            "model_path": globals().get('save_model_path').as_posix(),
            "scaler_path": globals().get('save_scaler_path').as_posix(),
            "extra_vars": globals().get('extra_vars', []),
            "early_stopping_patience": globals().get('early_stopping_patience', 50),
        },
        "metrics": {
        }
    }
    return config_values