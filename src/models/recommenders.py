"""Recommendation models for product recommendation system."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import implicit
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    def __init__(self, name: str):
        """Initialize the recommender.
        
        Args:
            name: Name of the recommender model.
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the recommendation model.
        
        Args:
            interactions_df: DataFrame with user-product interactions.
            products_df: DataFrame with product information.
            users_df: Optional DataFrame with user information.
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: ID of the user.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of (product_id, score) tuples.
        """
        pass
    
    @abstractmethod
    def get_similar_items(self, product_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar items to a given product.
        
        Args:
            product_id: ID of the product.
            n_similar: Number of similar items to return.
            
        Returns:
            List of (product_id, similarity_score) tuples.
        """
        pass


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation using TF-IDF and cosine similarity."""
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize the content-based recommender.
        
        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: Range of n-grams to extract.
        """
        super().__init__("Content-Based")
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.product_features = None
        self.product_ids = None
        self.user_profiles = {}
    
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the content-based model."""
        logger.info("Fitting content-based recommender...")
        
        # Extract text features from product descriptions
        from .loader import TextFeatureExtractor
        
        extractor = TextFeatureExtractor(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        
        # Combine title and description for better features
        product_texts = products_df['title'] + ' ' + products_df['description']
        self.product_features = extractor.fit_transform(product_texts.tolist())
        self.product_ids = products_df['product_id'].tolist()
        self.vectorizer = extractor
        
        # Build user profiles based on interaction history
        for user_id in interactions_df['user_id'].unique():
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            interacted_products = user_interactions['product_id'].tolist()
            
            # Get indices of interacted products
            product_indices = [self.product_ids.index(pid) for pid in interacted_products if pid in self.product_ids]
            
            if product_indices:
                # Average the features of interacted products
                user_profile = np.mean(self.product_features[product_indices], axis=0)
                self.user_profiles[user_id] = user_profile
        
        self.is_fitted = True
        logger.info(f"Content-based model fitted for {len(self.user_profiles)} users")
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate content-based recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_profiles:
            # Return popular items for cold-start users
            return self._get_popular_items(n_recommendations)
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate cosine similarity between user profile and all products
        similarities = cosine_similarity([user_profile], self.product_features)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            product_id = self.product_ids[idx]
            score = similarities[idx]
            recommendations.append((product_id, score))
        
        return recommendations
    
    def get_similar_items(self, product_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar items based on content features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
        
        if product_id not in self.product_ids:
            return []
        
        product_idx = self.product_ids.index(product_id)
        product_features = self.product_features[product_idx]
        
        # Calculate similarity with all other products
        similarities = cosine_similarity([product_features], self.product_features)[0]
        
        # Get most similar items (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        similar_items = []
        for idx in similar_indices:
            similar_product_id = self.product_ids[idx]
            score = similarities[idx]
            similar_items.append((similar_product_id, score))
        
        return similar_items
    
    def _get_popular_items(self, n_items: int) -> List[Tuple[str, float]]:
        """Get popular items for cold-start users."""
        # This would typically be based on interaction counts
        # For now, return random items with low scores
        random_indices = np.random.choice(len(self.product_ids), size=n_items, replace=False)
        return [(self.product_ids[idx], 0.1) for idx in random_indices]


class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering using matrix factorization."""
    
    def __init__(self, factors: int = 50, regularization: float = 0.01, iterations: int = 50):
        """Initialize the collaborative filtering recommender.
        
        Args:
            factors: Number of latent factors.
            regularization: Regularization parameter.
            iterations: Number of training iterations.
        """
        super().__init__("Collaborative Filtering")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = None
        self.user_item_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
    
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the collaborative filtering model."""
        logger.info("Fitting collaborative filtering recommender...")
        
        # Create user-item matrix
        self._create_user_item_matrix(interactions_df)
        
        # Initialize ALS model
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42
        )
        
        # Fit the model
        self.model.fit(self.user_item_matrix)
        
        self.is_fitted = True
        logger.info(f"Collaborative filtering model fitted with {self.factors} factors")
    
    def _create_user_item_matrix(self, interactions_df: pd.DataFrame) -> None:
        """Create user-item interaction matrix."""
        # Create mappings
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['product_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Create sparse matrix
        user_indices = [self.user_mapping[user] for user in interactions_df['user_id']]
        item_indices = [self.item_mapping[item] for item in interactions_df['product_id']]
        
        # Use interaction counts as weights
        weights = np.ones(len(interactions_df))
        
        self.user_item_matrix = implicit.csc_matrix(
            (weights, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items))
        )
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate collaborative filtering recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            # Return popular items for cold-start users
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_mapping[user_id]
        
        # Get recommendations
        recommendations = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=n_recommendations,
            filter_already_liked_items=True
        )
        
        result = []
        for item_idx, score in recommendations:
            item_id = self.reverse_item_mapping[item_idx]
            result.append((item_id, score))
        
        return result
    
    def get_similar_items(self, product_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar items using item factors."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
        
        if product_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[product_id]
        
        # Get similar items
        similar_items = self.model.similar_items(item_idx, n_similar + 1)
        
        result = []
        for similar_idx, score in similar_items[1:]:  # Skip the item itself
            similar_item_id = self.reverse_item_mapping[similar_idx]
            result.append((similar_item_id, score))
        
        return result
    
    def _get_popular_items(self, n_items: int) -> List[Tuple[str, float]]:
        """Get popular items for cold-start users."""
        # Calculate item popularity from interaction counts
        item_counts = self.user_item_matrix.sum(axis=0).A1
        popular_indices = np.argsort(item_counts)[::-1][:n_items]
        
        result = []
        for idx in popular_indices:
            item_id = self.reverse_item_mapping[idx]
            score = item_counts[idx] / item_counts.max()  # Normalize score
            result.append((item_id, score))
        
        return result


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining content-based and collaborative filtering."""
    
    def __init__(self, content_weight: float = 0.6, collab_weight: float = 0.4):
        """Initialize the hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations.
            collab_weight: Weight for collaborative filtering recommendations.
        """
        super().__init__("Hybrid")
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.content_model = ContentBasedRecommender()
        self.collab_model = CollaborativeFilteringRecommender()
    
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit both component models."""
        logger.info("Fitting hybrid recommender...")
        
        # Fit both models
        self.content_model.fit(interactions_df, products_df, users_df)
        self.collab_model.fit(interactions_df, products_df, users_df)
        
        self.is_fitted = True
        logger.info("Hybrid model fitted successfully")
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate hybrid recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        content_recs = self.content_model.recommend(user_id, n_recommendations * 2)
        collab_recs = self.collab_model.recommend(user_id, n_recommendations * 2)
        
        # Combine recommendations
        combined_scores = {}
        
        # Add content-based scores
        for item_id, score in content_recs:
            combined_scores[item_id] = self.content_weight * score
        
        # Add collaborative filtering scores
        for item_id, score in collab_recs:
            if item_id in combined_scores:
                combined_scores[item_id] += self.collab_weight * score
            else:
                combined_scores[item_id] = self.collab_weight * score
        
        # Sort by combined score and return top recommendations
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_recommendations]
    
    def get_similar_items(self, product_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar items using hybrid approach."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
        
        # Get similar items from both models
        content_similar = self.content_model.get_similar_items(product_id, n_similar)
        collab_similar = self.collab_model.get_similar_items(product_id, n_similar)
        
        # Combine similarities
        combined_scores = {}
        
        for item_id, score in content_similar:
            combined_scores[item_id] = self.content_weight * score
        
        for item_id, score in collab_similar:
            if item_id in combined_scores:
                combined_scores[item_id] += self.collab_weight * score
            else:
                combined_scores[item_id] = self.collab_weight * score
        
        # Sort by combined score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_items[:n_similar]


class PopularityRecommender(BaseRecommender):
    """Simple popularity-based recommender."""
    
    def __init__(self):
        """Initialize the popularity recommender."""
        super().__init__("Popularity-Based")
        self.popular_items = []
    
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the popularity model."""
        logger.info("Fitting popularity-based recommender...")
        
        # Calculate item popularity based on interaction counts
        item_counts = interactions_df['product_id'].value_counts()
        self.popular_items = [(item_id, count) for item_id, count in item_counts.items()]
        
        self.is_fitted = True
        logger.info(f"Popularity model fitted with {len(self.popular_items)} items")
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Generate popularity-based recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Return top popular items
        return self.popular_items[:n_recommendations]
    
    def get_similar_items(self, product_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Get similar items (same as popular items for popularity model)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting similar items")
        
        return self.popular_items[:n_similar]


def create_recommender_ensemble() -> Dict[str, BaseRecommender]:
    """Create an ensemble of different recommendation models."""
    return {
        "popularity": PopularityRecommender(),
        "content_based": ContentBasedRecommender(),
        "collaborative_filtering": CollaborativeFilteringRecommender(),
        "hybrid": HybridRecommender()
    }


if __name__ == "__main__":
    # Example usage
    from .loader import ProductDataLoader, set_random_seeds
    
    set_random_seeds(42)
    
    # Load data
    loader = ProductDataLoader()
    products_df, users_df, interactions_df = loader.load_data()
    
    # Create and fit models
    models = create_recommender_ensemble()
    
    for name, model in models.items():
        print(f"\nFitting {name} model...")
        model.fit(interactions_df, products_df, users_df)
        
        # Test recommendations
        test_user = users_df['user_id'].iloc[0]
        recommendations = model.recommend(test_user, n_recommendations=5)
        
        print(f"Recommendations for user {test_user}:")
        for product_id, score in recommendations:
            product_info = products_df[products_df['product_id'] == product_id].iloc[0]
            print(f"  {product_info['title']} (score: {score:.3f})")
