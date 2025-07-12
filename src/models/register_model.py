import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Optional
import argparse


def register_model(
    model_id: str,
    ticker: str,
    n_ahead: int,
    version: int = 1,
    vix: bool = False,
    stage: str = "Staging",
    tags: Optional[Dict] = None
) -> None:
    """
    Register a model in MLFlow and transition it to the specified stage.
    
    Args:
        model_id (str): The MLFlow run ID containing the model
        ticker (str): The stock ticker symbol
        n_ahead (int): Number of days ahead for prediction
        version (int, optional): Model version number. Defaults to 1.
        vix (bool, optional): Whether VIX features were used. Defaults to False.
        stage (str, optional): Target stage for the model. Defaults to "Staging".
        tags (Dict, optional): Additional tags for the model. Defaults to None.
    """
    # Set default tags
    default_tags = {
        "ticker": ticker,
        "n_ahead": n_ahead,
        "vix": vix
    }
    
    # Merge with additional tags if provided
    if tags:
        default_tags.update(tags)
    
    # Register the model
    mlflow.register_model(
        model_uri=f"runs:/{model_id}/model",
        name=f"{ticker}_{n_ahead}",
        tags=default_tags
    )
    
    # Transition to specified stage
    client = MlflowClient()
    client.transition_model_version_stage(
        name=f"{ticker}_{n_ahead}",
        version=version,
        stage=stage
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Register and stage an MLFlow model.')
    
    parser.add_argument('--model-id', type=str, required=True,
                      help='The MLFlow run ID containing the model')
    parser.add_argument('--ticker', type=str, required=True,
                      help='The stock ticker symbol')
    parser.add_argument('--n-ahead', type=int, required=True,
                      help='Number of days ahead for prediction')
    parser.add_argument('--version', type=int, default=1,
                      help='Model version number (default: 1)')
    parser.add_argument('--vix', action='store_true',
                      help='Whether VIX features were used')
    parser.add_argument('--stage', type=str, default='Staging',
                      choices=['Staging', 'Production', 'Archived'],
                      help='Target stage for the model (default: Staging)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    register_model(
        model_id=args.model_id,
        ticker=args.ticker,
        n_ahead=args.n_ahead,
        version=args.version,
        vix=args.vix,
        stage=args.stage
    ) 