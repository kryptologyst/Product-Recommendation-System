"""Data loading and preprocessing utilities for product recommendation system."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductDataLoader:
    """Handles loading and preprocessing of product recommendation data."""
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def generate_sample_data(self, n_products: int = 1000, n_users: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate realistic sample data for product recommendations.
        
        Args:
            n_products: Number of products to generate.
            n_users: Number of users to generate.
            
        Returns:
            Tuple of (products_df, users_df, interactions_df).
        """
        logger.info(f"Generating sample data: {n_products} products, {n_users} users")
        
        # Product categories and features
        categories = [
            "Electronics", "Clothing", "Books", "Home & Garden", "Sports",
            "Beauty", "Toys", "Automotive", "Health", "Food & Beverage"
        ]
        
        brands = [
            "TechCorp", "StyleBrand", "BookHouse", "GardenPro", "SportMax",
            "BeautyLux", "ToyWorld", "AutoTech", "HealthPlus", "FoodFresh"
        ]
        
        # Generate products
        products_data = []
        for i in range(n_products):
            category = random.choice(categories)
            brand = random.choice(brands)
            price = round(random.uniform(10, 1000), 2)
            
            # Generate realistic product descriptions
            description = self._generate_product_description(category, brand, price)
            
            products_data.append({
                "product_id": f"prod_{i:04d}",
                "title": f"{brand} {category} Product {i+1}",
                "category": category,
                "brand": brand,
                "price": price,
                "description": description,
                "rating": round(random.uniform(3.0, 5.0), 1),
                "availability": random.choice(["In Stock", "Limited Stock", "Out of Stock"])
            })
        
        products_df = pd.DataFrame(products_data)
        
        # Generate users
        users_data = []
        for i in range(n_users):
            users_data.append({
                "user_id": f"user_{i:04d}",
                "age": random.randint(18, 65),
                "gender": random.choice(["M", "F", "Other"]),
                "location": random.choice(["US", "UK", "CA", "AU", "DE"]),
                "preferred_categories": random.sample(categories, k=random.randint(1, 3))
            })
        
        users_df = pd.DataFrame(users_data)
        
        # Generate interactions (implicit feedback)
        interactions_data = []
        for user in users_df.itertuples():
            # Each user interacts with 10-50 products
            n_interactions = random.randint(10, 50)
            
            # Bias towards preferred categories
            preferred_cats = user.preferred_categories
            preferred_products = products_df[products_df["category"].isin(preferred_cats)]
            other_products = products_df[~products_df["category"].isin(preferred_cats)]
            
            # 70% interactions with preferred categories
            n_preferred = int(n_interactions * 0.7)
            n_other = n_interactions - n_preferred
            
            selected_preferred = preferred_products.sample(min(n_preferred, len(preferred_products)))
            selected_other = other_products.sample(min(n_other, len(other_products)))
            
            selected_products = pd.concat([selected_preferred, selected_other])
            
            for product in selected_products.itertuples():
                # Generate timestamp (last 6 months)
                timestamp = pd.Timestamp.now() - pd.Timedelta(days=random.randint(0, 180))
                
                interactions_data.append({
                    "user_id": user.user_id,
                    "product_id": product.product_id,
                    "timestamp": timestamp,
                    "interaction_type": random.choice(["view", "click", "add_to_cart", "purchase"]),
                    "rating": random.randint(1, 5) if random.random() < 0.3 else None
                })
        
        interactions_df = pd.DataFrame(interactions_data)
        
        # Save data
        self._save_data(products_df, users_df, interactions_df)
        
        return products_df, users_df, interactions_df
    
    def _generate_product_description(self, category: str, brand: str, price: float) -> str:
        """Generate realistic product descriptions."""
        descriptions = {
            "Electronics": f"High-quality {brand} electronic device with advanced features and reliable performance. Perfect for tech enthusiasts.",
            "Clothing": f"Stylish {brand} clothing item made from premium materials. Comfortable and fashionable design.",
            "Books": f"Engaging {brand} publication covering interesting topics. Well-written and informative content.",
            "Home & Garden": f"Practical {brand} home and garden solution. Durable construction and user-friendly design.",
            "Sports": f"Professional-grade {brand} sports equipment. Designed for performance and durability.",
            "Beauty": f"Premium {brand} beauty product with high-quality ingredients. Safe and effective formula.",
            "Toys": f"Fun and educational {brand} toy. Safe for children and encourages creativity and learning.",
            "Automotive": f"Reliable {brand} automotive accessory. Built to last and easy to install.",
            "Health": f"Health-focused {brand} product. Supports wellness and healthy lifestyle choices.",
            "Food & Beverage": f"Delicious {brand} food or beverage option. Made with quality ingredients."
        }
        
        base_desc = descriptions.get(category, f"Quality {brand} product in the {category} category.")
        
        if price > 500:
            base_desc += " Premium quality with advanced features."
        elif price < 50:
            base_desc += " Great value for money."
        
        return base_desc
    
    def _save_data(self, products_df: pd.DataFrame, users_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """Save generated data to CSV files."""
        products_df.to_csv(self.data_dir / "products.csv", index=False)
        users_df.to_csv(self.data_dir / "users.csv", index=False)
        interactions_df.to_csv(self.data_dir / "interactions.csv", index=False)
        logger.info(f"Data saved to {self.data_dir}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from CSV files.
        
        Returns:
            Tuple of (products_df, users_df, interactions_df).
        """
        try:
            products_df = pd.read_csv(self.data_dir / "products.csv")
            users_df = pd.read_csv(self.data_dir / "users.csv")
            interactions_df = pd.read_csv(self.data_dir / "interactions.csv")
            
            # Convert timestamp column
            interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
            
            logger.info(f"Loaded data: {len(products_df)} products, {len(users_df)} users, {len(interactions_df)} interactions")
            return products_df, users_df, interactions_df
            
        except FileNotFoundError:
            logger.warning("Data files not found. Generating sample data...")
            return self.generate_sample_data()


class TextFeatureExtractor:
    """Extract text features from product descriptions using TF-IDF."""
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize the text feature extractor.
        
        Args:
            max_features: Maximum number of features to extract.
            ngram_range: Range of n-grams to extract.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer and transform texts to features.
        
        Args:
            texts: List of text documents.
            
        Returns:
            TF-IDF feature matrix.
        """
        features = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        return features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to features using fitted vectorizer.
        
        Args:
            texts: List of text documents.
            
        Returns:
            TF-IDF feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the vectorizer."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        return self.vectorizer.get_feature_names_out().tolist()


def create_train_test_split(
    interactions_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create time-aware train/test split for interactions.
    
    Args:
        interactions_df: DataFrame with user interactions.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    # Sort by timestamp
    interactions_df = interactions_df.sort_values('timestamp')
    
    # For each user, keep the last interactions for testing
    train_data = []
    test_data = []
    
    for user_id in interactions_df['user_id'].unique():
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        # Calculate split point
        n_interactions = len(user_interactions)
        n_test = max(1, int(n_interactions * test_size))
        
        # Split chronologically
        train_interactions = user_interactions.iloc[:-n_test]
        test_interactions = user_interactions.iloc[-n_test:]
        
        train_data.append(train_interactions)
        test_data.append(test_interactions)
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    logger.info(f"Train/test split: {len(train_df)} train, {len(test_df)} test interactions")
    
    return train_df, test_df


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seeds set to {seed}")


if __name__ == "__main__":
    # Example usage
    set_random_seeds(42)
    
    loader = ProductDataLoader()
    products_df, users_df, interactions_df = loader.load_data()
    
    print("Products DataFrame:")
    print(products_df.head())
    print(f"\nShape: {products_df.shape}")
    
    print("\nUsers DataFrame:")
    print(users_df.head())
    print(f"\nShape: {users_df.shape}")
    
    print("\nInteractions DataFrame:")
    print(interactions_df.head())
    print(f"\nShape: {interactions_df.shape}")
