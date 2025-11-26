#!/usr/bin/env python3
"""Simple test script to verify the recommendation system works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic functionality of the recommendation system."""
    print("Testing Product Recommendation System...")
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        from data.loader import ProductDataLoader, set_random_seeds
        set_random_seeds(42)
        
        loader = ProductDataLoader()
        products_df, users_df, interactions_df = loader.generate_sample_data(n_products=20, n_users=10)
        print(f"   ✓ Generated {len(products_df)} products, {len(users_df)} users, {len(interactions_df)} interactions")
        
        # Test popularity model
        print("2. Testing popularity model...")
        from models.recommenders import PopularityRecommender
        model = PopularityRecommender()
        model.fit(interactions_df, products_df, users_df)
        recommendations = model.recommend('user_0000', n_recommendations=3)
        print(f"   ✓ Generated {len(recommendations)} recommendations")
        
        # Test content-based model
        print("3. Testing content-based model...")
        from models.recommenders import ContentBasedRecommender
        cb_model = ContentBasedRecommender(max_features=100)
        cb_model.fit(interactions_df, products_df, users_df)
        cb_recommendations = cb_model.recommend('user_0000', n_recommendations=3)
        print(f"   ✓ Generated {len(cb_recommendations)} content-based recommendations")
        
        # Test evaluation
        print("4. Testing evaluation...")
        from evaluation.metrics import RecommendationEvaluator
        evaluator = RecommendationEvaluator()
        metrics = evaluator.evaluate_model(model, interactions_df, products_df, k_values=[5], n_recommendations=10)
        print(f"   ✓ Calculated {len(metrics)} evaluation metrics")
        
        print("\n🎉 All tests passed! The recommendation system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
