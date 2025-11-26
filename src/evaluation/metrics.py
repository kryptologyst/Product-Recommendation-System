"""Evaluation metrics and model comparison for recommendation systems."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.metrics = {}
    
    def evaluate_model(
        self,
        model,
        test_interactions: pd.DataFrame,
        products_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20],
        n_recommendations: int = 20
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model.
            test_interactions: Test set interactions.
            products_df: Product information DataFrame.
            k_values: List of k values for precision@k, recall@k, etc.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating {model.name} model...")
        
        metrics = {}
        
        # Get all unique users in test set
        test_users = test_interactions['user_id'].unique()
        
        # Calculate metrics for each k value
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_rate_scores = []
            
            for user_id in test_users:
                # Get user's test interactions
                user_test_items = set(test_interactions[test_interactions['user_id'] == user_id]['product_id'])
                
                if len(user_test_items) == 0:
                    continue
                
                # Get recommendations
                try:
                    recommendations = model.recommend(user_id, n_recommendations)
                    recommended_items = [item_id for item_id, _ in recommendations]
                    
                    # Calculate metrics
                    precision_k = self._precision_at_k(user_test_items, recommended_items, k)
                    recall_k = self._recall_at_k(user_test_items, recommended_items, k)
                    ndcg_k = self._ndcg_at_k(user_test_items, recommended_items, k)
                    hit_rate_k = self._hit_rate_at_k(user_test_items, recommended_items, k)
                    
                    precision_scores.append(precision_k)
                    recall_scores.append(recall_k)
                    ndcg_scores.append(ndcg_k)
                    hit_rate_scores.append(hit_rate_k)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating user {user_id}: {e}")
                    continue
            
            # Average metrics across users
            metrics[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            metrics[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            metrics[f'hit_rate@{k}'] = np.mean(hit_rate_scores) if hit_rate_scores else 0.0
        
        # Calculate additional metrics
        metrics['coverage'] = self._calculate_coverage(model, test_users, products_df, n_recommendations)
        metrics['diversity'] = self._calculate_diversity(model, test_users, products_df, n_recommendations)
        metrics['novelty'] = self._calculate_novelty(model, test_users, products_df, n_recommendations)
        
        logger.info(f"Evaluation completed for {model.name}")
        return metrics
    
    def _precision_at_k(self, relevant_items: set, recommended_items: List[str], k: int) -> float:
        """Calculate Precision@K."""
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & relevant_items)
        return relevant_recommended / k if k > 0 else 0.0
    
    def _recall_at_k(self, relevant_items: set, recommended_items: List[str], k: int) -> float:
        """Calculate Recall@K."""
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & relevant_items)
        return relevant_recommended / len(relevant_items) if len(relevant_items) > 0 else 0.0
    
    def _ndcg_at_k(self, relevant_items: set, recommended_items: List[str], k: int) -> float:
        """Calculate NDCG@K."""
        recommended_k = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _hit_rate_at_k(self, relevant_items: set, recommended_items: List[str], k: int) -> float:
        """Calculate Hit Rate@K."""
        recommended_k = recommended_items[:k]
        return 1.0 if len(set(recommended_k) & relevant_items) > 0 else 0.0
    
    def _calculate_coverage(self, model, test_users: List[str], products_df: pd.DataFrame, n_recommendations: int) -> float:
        """Calculate catalog coverage."""
        all_recommended_items = set()
        
        for user_id in test_users[:100]:  # Sample for efficiency
            try:
                recommendations = model.recommend(user_id, n_recommendations)
                recommended_items = [item_id for item_id, _ in recommendations]
                all_recommended_items.update(recommended_items)
            except:
                continue
        
        total_items = len(products_df)
        return len(all_recommended_items) / total_items if total_items > 0 else 0.0
    
    def _calculate_diversity(self, model, test_users: List[str], products_df: pd.DataFrame, n_recommendations: int) -> float:
        """Calculate intra-list diversity."""
        diversity_scores = []
        
        for user_id in test_users[:100]:  # Sample for efficiency
            try:
                recommendations = model.recommend(user_id, n_recommendations)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                if len(recommended_items) < 2:
                    continue
                
                # Calculate pairwise category diversity
                categories = []
                for item_id in recommended_items:
                    item_info = products_df[products_df['product_id'] == item_id]
                    if not item_info.empty:
                        categories.append(item_info.iloc[0]['category'])
                
                # Calculate diversity as 1 - (same category pairs / total pairs)
                total_pairs = len(categories) * (len(categories) - 1) / 2
                same_category_pairs = sum(1 for i in range(len(categories)) 
                                       for j in range(i+1, len(categories)) 
                                       if categories[i] == categories[j])
                
                diversity = 1.0 - (same_category_pairs / total_pairs) if total_pairs > 0 else 0.0
                diversity_scores.append(diversity)
                
            except:
                continue
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_novelty(self, model, test_users: List[str], products_df: pd.DataFrame, n_recommendations: int) -> float:
        """Calculate novelty (inverse of popularity)."""
        # Calculate item popularity
        item_popularity = products_df['product_id'].value_counts()
        max_popularity = item_popularity.max()
        
        novelty_scores = []
        
        for user_id in test_users[:100]:  # Sample for efficiency
            try:
                recommendations = model.recommend(user_id, n_recommendations)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                user_novelty = []
                for item_id in recommended_items:
                    popularity = item_popularity.get(item_id, 0)
                    novelty = 1.0 - (popularity / max_popularity) if max_popularity > 0 else 0.0
                    user_novelty.append(novelty)
                
                if user_novelty:
                    novelty_scores.append(np.mean(user_novelty))
                    
            except:
                continue
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def compare_models(
        self,
        models: Dict[str, any],
        test_interactions: pd.DataFrame,
        products_df: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """Compare multiple models and return results as DataFrame.
        
        Args:
            models: Dictionary of model name -> model instance.
            test_interactions: Test set interactions.
            products_df: Product information DataFrame.
            k_values: List of k values for evaluation.
            
        Returns:
            DataFrame with model comparison results.
        """
        logger.info("Comparing multiple models...")
        
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_interactions, products_df, k_values)
            metrics['model'] = model_name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Reorder columns to put model name first
        cols = ['model'] + [col for col in results_df.columns if col != 'model']
        results_df = results_df[cols]
        
        logger.info("Model comparison completed")
        return results_df
    
    def create_leaderboard(self, results_df: pd.DataFrame, primary_metric: str = 'ndcg@10') -> pd.DataFrame:
        """Create a leaderboard sorted by primary metric.
        
        Args:
            results_df: Results DataFrame from compare_models.
            primary_metric: Primary metric to sort by.
            
        Returns:
            Sorted leaderboard DataFrame.
        """
        if primary_metric not in results_df.columns:
            logger.warning(f"Primary metric {primary_metric} not found. Using first metric.")
            primary_metric = results_df.columns[1]
        
        leaderboard = results_df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
        leaderboard.index = leaderboard.index + 1  # Start ranking from 1
        
        return leaderboard


def calculate_popularity_bias(
    interactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    recommendations: Dict[str, List[Tuple[str, float]]]
) -> Dict[str, float]:
    """Calculate popularity bias metrics.
    
    Args:
        interactions_df: Interactions DataFrame.
        products_df: Products DataFrame.
        recommendations: Dictionary of user_id -> recommendations.
        
    Returns:
        Dictionary of popularity bias metrics.
    """
    # Calculate item popularity
    item_counts = interactions_df['product_id'].value_counts()
    total_interactions = len(interactions_df)
    item_popularity = item_counts / total_interactions
    
    # Calculate recommendation popularity distribution
    rec_popularity_scores = []
    
    for user_id, user_recs in recommendations.items():
        for item_id, score in user_recs:
            popularity = item_popularity.get(item_id, 0)
            rec_popularity_scores.append(popularity)
    
    if not rec_popularity_scores:
        return {'mean_popularity': 0.0, 'popularity_gini': 0.0}
    
    # Calculate metrics
    mean_popularity = np.mean(rec_popularity_scores)
    
    # Calculate Gini coefficient for popularity distribution
    sorted_scores = np.sort(rec_popularity_scores)
    n = len(sorted_scores)
    cumsum = np.cumsum(sorted_scores)
    popularity_gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    return {
        'mean_popularity': mean_popularity,
        'popularity_gini': popularity_gini
    }


def calculate_fairness_metrics(
    models: Dict[str, any],
    test_interactions: pd.DataFrame,
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    user_attribute: str = 'gender'
) -> pd.DataFrame:
    """Calculate fairness metrics across user groups.
    
    Args:
        models: Dictionary of model name -> model instance.
        test_interactions: Test set interactions.
        users_df: Users DataFrame.
        products_df: Products DataFrame.
        user_attribute: User attribute to analyze fairness across.
        
    Returns:
        DataFrame with fairness metrics by group.
    """
    evaluator = RecommendationEvaluator()
    fairness_results = []
    
    # Get unique values of the attribute
    attribute_values = users_df[user_attribute].unique()
    
    for model_name, model in models.items():
        for attr_value in attribute_values:
            # Filter users by attribute value
            group_users = users_df[users_df[user_attribute] == attr_value]['user_id'].tolist()
            group_interactions = test_interactions[test_interactions['user_id'].isin(group_users)]
            
            if len(group_interactions) == 0:
                continue
            
            # Evaluate model on this group
            metrics = evaluator.evaluate_model(model, group_interactions, products_df)
            metrics['model'] = model_name
            metrics[user_attribute] = attr_value
            fairness_results.append(metrics)
    
    return pd.DataFrame(fairness_results)


if __name__ == "__main__":
    # Example usage
    from ..data.loader import ProductDataLoader, create_train_test_split, set_random_seeds
    from ..models.recommenders import create_recommender_ensemble
    
    set_random_seeds(42)
    
    # Load and split data
    loader = ProductDataLoader()
    products_df, users_df, interactions_df = loader.load_data()
    train_df, test_df = create_train_test_split(interactions_df)
    
    # Create and fit models
    models = create_recommender_ensemble()
    for model in models.values():
        model.fit(train_df, products_df, users_df)
    
    # Evaluate models
    evaluator = RecommendationEvaluator()
    results_df = evaluator.compare_models(models, test_df, products_df)
    
    print("Model Comparison Results:")
    print(results_df.round(4))
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(results_df)
    print("\nLeaderboard:")
    print(leaderboard.round(4))
