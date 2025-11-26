"""Utility functions for the recommendation system."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Save a model to disk.
    
    Args:
        model: Model object to save.
        filepath: Path where to save the model.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Union[str, Path]) -> Any:
    """Load a model from disk.
    
    Args:
        filepath: Path to the saved model.
        
    Returns:
        Loaded model object.
    """
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        filepath: Path where to save the config.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {filepath}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        filepath: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {filepath}")
    return config


def create_directory_structure(base_dir: Union[str, Path]) -> None:
    """Create the standard directory structure for the project.
    
    Args:
        base_dir: Base directory for the project.
    """
    base_dir = Path(base_dir)
    
    directories = [
        "data/raw",
        "data/processed",
        "models/checkpoints",
        "outputs/results",
        "outputs/logs",
        "notebooks",
        "scripts",
        "tests",
        "assets",
        "configs"
    ]
    
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Directory structure created in {base_dir}")


def validate_data_schema(
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame
) -> bool:
    """Validate that data follows the expected schema.
    
    Args:
        products_df: Products DataFrame.
        users_df: Users DataFrame.
        interactions_df: Interactions DataFrame.
        
    Returns:
        True if data is valid, False otherwise.
    """
    # Expected columns for products
    expected_product_cols = {
        'product_id', 'title', 'category', 'brand', 
        'price', 'description', 'rating', 'availability'
    }
    
    # Expected columns for users
    expected_user_cols = {
        'user_id', 'age', 'gender', 'location', 'preferred_categories'
    }
    
    # Expected columns for interactions
    expected_interaction_cols = {
        'user_id', 'product_id', 'timestamp', 'interaction_type'
    }
    
    # Check products schema
    if not expected_product_cols.issubset(set(products_df.columns)):
        missing_cols = expected_product_cols - set(products_df.columns)
        logger.error(f"Missing columns in products_df: {missing_cols}")
        return False
    
    # Check users schema
    if not expected_user_cols.issubset(set(users_df.columns)):
        missing_cols = expected_user_cols - set(users_df.columns)
        logger.error(f"Missing columns in users_df: {missing_cols}")
        return False
    
    # Check interactions schema
    if not expected_interaction_cols.issubset(set(interactions_df.columns)):
        missing_cols = expected_interaction_cols - set(interactions_df.columns)
        logger.error(f"Missing columns in interactions_df: {missing_cols}")
        return False
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(products_df['price']):
        logger.error("Price column must be numeric")
        return False
    
    if not pd.api.types.is_numeric_dtype(users_df['age']):
        logger.error("Age column must be numeric")
        return False
    
    if not pd.api.types.is_datetime64_any_dtype(interactions_df['timestamp']):
        logger.error("Timestamp column must be datetime")
        return False
    
    logger.info("Data schema validation passed")
    return True


def calculate_data_statistics(
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame
) -> Dict[str, Any]:
    """Calculate basic statistics about the dataset.
    
    Args:
        products_df: Products DataFrame.
        users_df: Users DataFrame.
        interactions_df: Interactions DataFrame.
        
    Returns:
        Dictionary with dataset statistics.
    """
    stats = {
        'n_products': len(products_df),
        'n_users': len(users_df),
        'n_interactions': len(interactions_df),
        'n_categories': products_df['category'].nunique(),
        'n_brands': products_df['brand'].nunique(),
        'avg_price': products_df['price'].mean(),
        'price_range': [products_df['price'].min(), products_df['price'].max()],
        'avg_rating': products_df['rating'].mean(),
        'avg_user_age': users_df['age'].mean(),
        'interaction_types': interactions_df['interaction_type'].value_counts().to_dict(),
        'sparsity': 1 - (len(interactions_df) / (len(users_df) * len(products_df)))
    }
    
    return stats


def print_data_statistics(stats: Dict[str, Any]) -> None:
    """Print dataset statistics in a formatted way.
    
    Args:
        stats: Statistics dictionary from calculate_data_statistics.
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    print(f"Products: {stats['n_products']:,}")
    print(f"Users: {stats['n_users']:,}")
    print(f"Interactions: {stats['n_interactions']:,}")
    print(f"Categories: {stats['n_categories']}")
    print(f"Brands: {stats['n_brands']}")
    print(f"Average Price: ${stats['avg_price']:.2f}")
    print(f"Price Range: ${stats['price_range'][0]:.2f} - ${stats['price_range'][1]:.2f}")
    print(f"Average Rating: {stats['avg_rating']:.2f}")
    print(f"Average User Age: {stats['avg_user_age']:.1f}")
    print(f"Sparsity: {stats['sparsity']:.4f}")
    
    print("\nInteraction Types:")
    for interaction_type, count in stats['interaction_types'].items():
        print(f"  {interaction_type}: {count:,}")
    print("="*50)


def create_sample_recommendations(
    model,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    n_users: int = 5,
    n_recommendations: int = 10
) -> pd.DataFrame:
    """Create sample recommendations for demonstration.
    
    Args:
        model: Trained recommendation model.
        products_df: Products DataFrame.
        users_df: Users DataFrame.
        n_users: Number of users to sample.
        n_recommendations: Number of recommendations per user.
        
    Returns:
        DataFrame with sample recommendations.
    """
    sample_users = users_df.sample(n_users)['user_id'].tolist()
    
    recommendations_data = []
    
    for user_id in sample_users:
        try:
            user_recs = model.recommend(user_id, n_recommendations)
            
            for rank, (product_id, score) in enumerate(user_recs, 1):
                product_info = products_df[products_df['product_id'] == product_id].iloc[0]
                
                recommendations_data.append({
                    'user_id': user_id,
                    'product_id': product_id,
                    'title': product_info['title'],
                    'category': product_info['category'],
                    'brand': product_info['brand'],
                    'price': product_info['price'],
                    'rating': product_info['rating'],
                    'recommendation_score': score,
                    'rank': rank
                })
        
        except Exception as e:
            logger.warning(f"Error generating recommendations for user {user_id}: {e}")
            continue
    
    return pd.DataFrame(recommendations_data)


def export_recommendations(
    recommendations_df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = 'csv'
) -> None:
    """Export recommendations to file.
    
    Args:
        recommendations_df: Recommendations DataFrame.
        filepath: Output file path.
        format: Export format ('csv', 'json', 'excel').
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        recommendations_df.to_csv(filepath, index=False)
    elif format == 'json':
        recommendations_df.to_json(filepath, orient='records', indent=2)
    elif format == 'excel':
        recommendations_df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Recommendations exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    from src.data.loader import ProductDataLoader, set_random_seeds
    
    set_random_seeds(42)
    
    # Load data
    loader = ProductDataLoader()
    products_df, users_df, interactions_df = loader.load_data()
    
    # Validate schema
    if validate_data_schema(products_df, users_df, interactions_df):
        # Calculate statistics
        stats = calculate_data_statistics(products_df, users_df, interactions_df)
        print_data_statistics(stats)
