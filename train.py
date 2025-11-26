"""Main training script for product recommendation system."""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from src.data.loader import ProductDataLoader, create_train_test_split, set_random_seeds
from src.models.recommenders import create_recommender_ensemble
from src.evaluation.metrics import RecommendationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_and_evaluate(
    config: Dict[str, Any],
    output_dir: Path = Path("outputs")
) -> None:
    """Train and evaluate recommendation models.
    
    Args:
        config: Configuration dictionary.
        output_dir: Output directory for results.
    """
    output_dir.mkdir(exist_ok=True)
    
    # Set random seeds
    set_random_seeds(config.get('random_seed', 42))
    
    # Load data
    logger.info("Loading data...")
    loader = ProductDataLoader(config.get('data_dir', 'data'))
    products_df, users_df, interactions_df = loader.load_data()
    
    # Create train/test split
    logger.info("Creating train/test split...")
    train_df, test_df = create_train_test_split(
        interactions_df,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_seed', 42)
    )
    
    # Create models
    logger.info("Creating models...")
    models = create_recommender_ensemble()
    
    # Train models
    logger.info("Training models...")
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(train_df, products_df, users_df)
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = RecommendationEvaluator()
    
    k_values = config.get('evaluation', {}).get('k_values', [5, 10, 20])
    n_recommendations = config.get('evaluation', {}).get('n_recommendations', 20)
    
    results_df = evaluator.compare_models(models, test_df, products_df, k_values)
    
    # Save results
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(results_df)
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logger.info(f"Leaderboard saved to {leaderboard_path}")
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(results_df.round(4))
    
    print("\n" + "="*50)
    print("LEADERBOARD")
    print("="*50)
    print(leaderboard.round(4))
    
    # Save model artifacts
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        # Save model info (in a real implementation, you'd save the actual model)
        model_info = {
            'name': model.name,
            'is_fitted': model.is_fitted
        }
        
        model_path = models_dir / f"{name}_info.yaml"
        with open(model_path, 'w') as f:
            yaml.dump(model_info, f)
    
    logger.info("Training and evaluation completed successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and evaluate recommendation models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train and evaluate
    train_and_evaluate(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
