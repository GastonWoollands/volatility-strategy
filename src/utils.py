import yaml
from pathlib import Path

#------------------------------------------------------------------------------------------------

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

#------------------------------------------------------------------------------------------------

def create_config_values(context):
    """Create a dictionary with the global configuration values from the given context."""
    global config_values

    config_values = {
        "model": {
            "input_size": context.get('input_size', None),
            "hidden_size": context.get('hidden_size', None),
            "output_size": context.get('output_size', None),
            "dropout": context.get('dropout', None),
            "seq_length": context.get('seq_length', None),
            "batch_size": context.get('batch_size', None),
            "seed": context.get('seed', 83),
            "criterion": context.get('criterion', "MSELoss"),
            "optimizer": context.get('optimizer', "Adam"),
            "learning_rate": context.get('learning_rate', None),
            "epochs": context.get('epochs', None),
        },
        "data": {
            "start_date": context.get('start_date', None),
            "end_date": context.get('end_date', None),
            "model_path": str(context.get('save_model_path', Path()).as_posix()) if isinstance(context.get('save_model_path', None), Path) else context.get('save_model_path', None),
            "scaler_path": str(context.get('save_scaler_path', Path()).as_posix()) if isinstance(context.get('save_scaler_path', None), Path) else context.get('save_scaler_path', None),
            "extra_vars": context.get('extra_vars', []),
            "early_stopping_patience": context.get('early_stopping_patience', 50),
        },
        "metrics": {}
    }
    
    return config_values