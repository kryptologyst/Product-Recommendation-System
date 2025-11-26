"""Unit tests for the recommendation system."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data.loader import ProductDataLoader, TextFeatureExtractor, create_train_test_split, set_random_seeds
from src.models.recommenders import (
    ContentBasedRecommender, 
    CollaborativeFilteringRecommender, 
    HybridRecommender, 
    PopularityRecommender
)
from src.evaluation.metrics import RecommendationEvaluator
from src.utils.helpers import validate_data_schema, calculate_data_statistics


class TestDataLoader:
    """Test cases for ProductDataLoader."""
    
    def test_init(self):
        """Test ProductDataLoader initialization."""
        loader = ProductDataLoader()
        assert loader.data_dir.name == "data"
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        loader = ProductDataLoader()
        products_df, users_df, interactions_df = loader.generate_sample_data(n_products=10, n_users=5)
        
        assert len(products_df) == 10
        assert len(users_df) == 5
        assert len(interactions_df) > 0
        
        # Check required columns
        required_product_cols = ['product_id', 'title', 'category', 'brand', 'price', 'description']
        assert all(col in products_df.columns for col in required_product_cols)
        
        required_user_cols = ['user_id', 'age', 'gender', 'location', 'preferred_categories']
        assert all(col in users_df.columns for col in required_user_cols)
        
        required_interaction_cols = ['user_id', 'product_id', 'timestamp', 'interaction_type']
        assert all(col in interactions_df.columns for col in required_interaction_cols)


class TestTextFeatureExtractor:
    """Test cases for TextFeatureExtractor."""
    
    def test_init(self):
        """Test TextFeatureExtractor initialization."""
        extractor = TextFeatureExtractor()
        assert extractor.max_features == 1000
        assert extractor.ngram_range == (1, 2)
        assert not extractor.is_fitted
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        extractor = TextFeatureExtractor(max_features=100)
        texts = ["This is a test document", "Another test document"]
        
        features = extractor.fit_transform(texts)
        
        assert features.shape[0] == 2
        assert features.shape[1] <= 100
        assert extractor.is_fitted
    
    def test_transform_without_fit(self):
        """Test transform method without fitting."""
        extractor = TextFeatureExtractor()
        texts = ["Test document"]
        
        with pytest.raises(ValueError):
            extractor.transform(texts)


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def test_init(self):
        """Test ContentBasedRecommender initialization."""
        recommender = ContentBasedRecommender()
        assert recommender.name == "Content-Based"
        assert not recommender.is_fitted
    
    def test_fit(self):
        """Test model fitting."""
        recommender = ContentBasedRecommender()
        
        # Create sample data
        products_df = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2'],
            'title': ['Product 1', 'Product 2'],
            'description': ['Description 1', 'Description 2']
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1'],
            'product_id': ['prod_1', 'prod_2']
        })
        
        recommender.fit(interactions_df, products_df)
        assert recommender.is_fitted
        assert len(recommender.user_profiles) > 0
    
    def test_recommend(self):
        """Test recommendation generation."""
        recommender = ContentBasedRecommender()
        
        # Create sample data
        products_df = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2', 'prod_3'],
            'title': ['Product 1', 'Product 2', 'Product 3'],
            'description': ['Description 1', 'Description 2', 'Description 3']
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1'],
            'product_id': ['prod_1', 'prod_2']
        })
        
        recommender.fit(interactions_df, products_df)
        
        recommendations = recommender.recommend('user_1', n_recommendations=2)
        assert len(recommendations) == 2
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)


class TestCollaborativeFilteringRecommender:
    """Test cases for CollaborativeFilteringRecommender."""
    
    def test_init(self):
        """Test CollaborativeFilteringRecommender initialization."""
        recommender = CollaborativeFilteringRecommender()
        assert recommender.name == "Collaborative Filtering"
        assert not recommender.is_fitted
    
    def test_fit(self):
        """Test model fitting."""
        recommender = CollaborativeFilteringRecommender(factors=10, iterations=5)
        
        # Create sample data
        products_df = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2'],
            'title': ['Product 1', 'Product 2']
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1', 'user_2'],
            'product_id': ['prod_1', 'prod_2', 'prod_1']
        })
        
        recommender.fit(interactions_df, products_df)
        assert recommender.is_fitted
        assert recommender.model is not None


class TestPopularityRecommender:
    """Test cases for PopularityRecommender."""
    
    def test_init(self):
        """Test PopularityRecommender initialization."""
        recommender = PopularityRecommender()
        assert recommender.name == "Popularity-Based"
        assert not recommender.is_fitted
    
    def test_fit_and_recommend(self):
        """Test model fitting and recommendation."""
        recommender = PopularityRecommender()
        
        # Create sample data
        products_df = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2', 'prod_3'],
            'title': ['Product 1', 'Product 2', 'Product 3']
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1', 'user_2', 'user_2', 'user_2'],
            'product_id': ['prod_1', 'prod_2', 'prod_1', 'prod_1', 'prod_3']
        })
        
        recommender.fit(interactions_df, products_df)
        assert recommender.is_fitted
        assert len(recommender.popular_items) > 0
        
        recommendations = recommender.recommend('user_1', n_recommendations=2)
        assert len(recommendations) == 2


class TestHybridRecommender:
    """Test cases for HybridRecommender."""
    
    def test_init(self):
        """Test HybridRecommender initialization."""
        recommender = HybridRecommender()
        assert recommender.name == "Hybrid"
        assert not recommender.is_fitted
    
    def test_fit(self):
        """Test model fitting."""
        recommender = HybridRecommender()
        
        # Create sample data
        products_df = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2'],
            'title': ['Product 1', 'Product 2'],
            'description': ['Description 1', 'Description 2']
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1'],
            'product_id': ['prod_1', 'prod_2']
        })
        
        recommender.fit(interactions_df, products_df)
        assert recommender.is_fitted
        assert recommender.content_model.is_fitted
        assert recommender.collab_model.is_fitted


class TestRecommendationEvaluator:
    """Test cases for RecommendationEvaluator."""
    
    def test_init(self):
        """Test RecommendationEvaluator initialization."""
        evaluator = RecommendationEvaluator()
        assert evaluator.metrics == {}
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        evaluator = RecommendationEvaluator()
        
        relevant_items = {'item_1', 'item_2', 'item_3'}
        recommended_items = ['item_1', 'item_4', 'item_2', 'item_5']
        
        precision = evaluator._precision_at_k(relevant_items, recommended_items, k=3)
        expected = 2 / 3  # 2 relevant items out of 3 recommendations
        assert precision == expected
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        evaluator = RecommendationEvaluator()
        
        relevant_items = {'item_1', 'item_2', 'item_3'}
        recommended_items = ['item_1', 'item_4', 'item_2', 'item_5']
        
        recall = evaluator._recall_at_k(relevant_items, recommended_items, k=3)
        expected = 2 / 3  # 2 relevant items out of 3 total relevant items
        assert recall == expected
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        evaluator = RecommendationEvaluator()
        
        relevant_items = {'item_1', 'item_2', 'item_3'}
        recommended_items = ['item_1', 'item_4', 'item_2', 'item_5']
        
        ndcg = evaluator._ndcg_at_k(relevant_items, recommended_items, k=3)
        assert 0 <= ndcg <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        evaluator = RecommendationEvaluator()
        
        relevant_items = {'item_1', 'item_2', 'item_3'}
        recommended_items = ['item_1', 'item_4', 'item_2', 'item_5']
        
        hit_rate = evaluator._hit_rate_at_k(relevant_items, recommended_items, k=3)
        assert hit_rate == 1.0  # At least one relevant item in top-3


class TestHelpers:
    """Test cases for utility functions."""
    
    def test_validate_data_schema(self):
        """Test data schema validation."""
        # Valid data
        products_df = pd.DataFrame({
            'product_id': ['prod_1'],
            'title': ['Product 1'],
            'category': ['Electronics'],
            'brand': ['Brand 1'],
            'price': [100.0],
            'description': ['Description'],
            'rating': [4.5],
            'availability': ['In Stock']
        })
        
        users_df = pd.DataFrame({
            'user_id': ['user_1'],
            'age': [25],
            'gender': ['M'],
            'location': ['US'],
            'preferred_categories': [['Electronics']]
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1'],
            'product_id': ['prod_1'],
            'timestamp': [pd.Timestamp.now()],
            'interaction_type': ['view']
        })
        
        assert validate_data_schema(products_df, users_df, interactions_df)
        
        # Invalid data (missing column)
        invalid_products_df = products_df.drop('price', axis=1)
        assert not validate_data_schema(invalid_products_df, users_df, interactions_df)
    
    def test_calculate_data_statistics(self):
        """Test data statistics calculation."""
        products_df = pd.DataFrame({
            'product_id': ['prod_1', 'prod_2'],
            'category': ['Electronics', 'Clothing'],
            'brand': ['Brand 1', 'Brand 2'],
            'price': [100.0, 50.0],
            'rating': [4.5, 3.5]
        })
        
        users_df = pd.DataFrame({
            'user_id': ['user_1', 'user_2'],
            'age': [25, 30]
        })
        
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1'],
            'product_id': ['prod_1', 'prod_2'],
            'interaction_type': ['view', 'click']
        })
        
        stats = calculate_data_statistics(products_df, users_df, interactions_df)
        
        assert stats['n_products'] == 2
        assert stats['n_users'] == 2
        assert stats['n_interactions'] == 2
        assert stats['n_categories'] == 2
        assert stats['avg_price'] == 75.0


class TestTrainTestSplit:
    """Test cases for train/test split functionality."""
    
    def test_create_train_test_split(self):
        """Test train/test split creation."""
        interactions_df = pd.DataFrame({
            'user_id': ['user_1', 'user_1', 'user_1', 'user_2', 'user_2'],
            'product_id': ['prod_1', 'prod_2', 'prod_3', 'prod_1', 'prod_2'],
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='D')
        })
        
        train_df, test_df = create_train_test_split(interactions_df, test_size=0.2)
        
        assert len(train_df) + len(test_df) == len(interactions_df)
        assert len(test_df) > 0
        assert len(train_df) > 0


if __name__ == "__main__":
    pytest.main([__file__])
