# Product Recommendation System

A comprehensive product recommendation system implementing multiple recommendation algorithms including content-based filtering, collaborative filtering, and hybrid approaches.

## Features

- **Multiple Recommendation Models**: Content-based, collaborative filtering, hybrid, and popularity-based recommenders
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, Hit Rate, Coverage, Diversity, and Novelty metrics
- **Interactive Demo**: Streamlit-based web application for exploring recommendations
- **Realistic Data Generation**: Automatic generation of synthetic product and user data
- **Modern Architecture**: Clean, modular codebase with type hints and comprehensive documentation
- **Production Ready**: Proper project structure, configuration management, and testing framework

## Project Structure

```
├── src/
│   ├── data/
│   │   └── loader.py              # Data loading and preprocessing
│   ├── models/
│   │   └── recommenders.py        # Recommendation models
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation metrics
│   └── utils/
├── configs/
│   └── config.yaml                # Configuration file
├── notebooks/                      # Jupyter notebooks for analysis
├── scripts/                        # Utility scripts
├── tests/                         # Unit tests
├── assets/                        # Static assets
├── data/                          # Data directory
├── outputs/                        # Model outputs and results
├── demo.py                        # Streamlit demo application
├── train.py                       # Main training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Product-Recommendation-System.git
cd Product-Recommendation-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
# Start the Streamlit demo
streamlit run demo.py
```

The demo will be available at `http://localhost:8501` with the following features:
- **Overview**: System statistics and data distribution
- **User Recommendations**: Generate personalized recommendations for users
- **Product Similarity**: Find similar products to a given item
- **Model Comparison**: Compare different recommendation models
- **Data Analysis**: Explore the dataset with interactive visualizations

### 3. Train Models

```bash
# Train and evaluate models
python train.py --config configs/config.yaml --output-dir outputs
```

## Dataset Schema

The system works with three main data files:

### products.csv
- `product_id`: Unique product identifier
- `title`: Product title
- `category`: Product category
- `brand`: Product brand
- `price`: Product price
- `description`: Product description
- `rating`: Average product rating
- `availability`: Stock availability status

### users.csv
- `user_id`: Unique user identifier
- `age`: User age
- `gender`: User gender
- `location`: User location
- `preferred_categories`: List of preferred product categories

### interactions.csv
- `user_id`: User identifier
- `product_id`: Product identifier
- `timestamp`: Interaction timestamp
- `interaction_type`: Type of interaction (view, click, add_to_cart, purchase)
- `rating`: User rating (optional)

## Models

### Content-Based Filtering
Uses TF-IDF vectorization of product descriptions to build user profiles and recommend similar products based on content similarity.

### Collaborative Filtering
Implements matrix factorization using Alternating Least Squares (ALS) to learn user and item latent factors from interaction patterns.

### Hybrid Recommender
Combines content-based and collaborative filtering approaches with configurable weights to leverage both content and interaction signals.

### Popularity-Based Recommender
Simple baseline that recommends the most popular items based on interaction counts.

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Intra-list diversity based on product categories
- **Novelty**: Recommendation of less popular items

## Configuration

The system uses YAML configuration files for easy customization:

```yaml
# Random seed for reproducibility
random_seed: 42

# Data configuration
data_dir: "data"
test_size: 0.2

# Model configuration
models:
  content_based:
    max_features: 1000
    ngram_range: [1, 2]
  
  collaborative_filtering:
    factors: 50
    regularization: 0.01
    iterations: 50
  
  hybrid:
    content_weight: 0.6
    collab_weight: 0.4

# Evaluation configuration
evaluation:
  k_values: [5, 10, 20]
  n_recommendations: 20
```

## API Usage

### Basic Usage

```python
from src.data.loader import ProductDataLoader, set_random_seeds
from src.models.recommenders import create_recommender_ensemble
from src.evaluation.metrics import RecommendationEvaluator

# Set random seeds
set_random_seeds(42)

# Load data
loader = ProductDataLoader()
products_df, users_df, interactions_df = loader.load_data()

# Create and train models
models = create_recommender_ensemble()
for model in models.values():
    model.fit(interactions_df, products_df, users_df)

# Generate recommendations
user_id = "user_0001"
recommendations = models["hybrid"].recommend(user_id, n_recommendations=10)

# Find similar products
product_id = "prod_0001"
similar_products = models["content_based"].get_similar_items(product_id, n_similar=5)
```

### Evaluation

```python
from src.evaluation.metrics import RecommendationEvaluator

# Evaluate models
evaluator = RecommendationEvaluator()
results_df = evaluator.compare_models(models, test_df, products_df)

# Create leaderboard
leaderboard = evaluator.create_leaderboard(results_df)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Performance

The system is designed to handle:
- Up to 100,000 products
- Up to 50,000 users
- Up to 1,000,000 interactions

Performance can be improved by:
- Using sparse matrices for large datasets
- Implementing approximate nearest neighbor search
- Adding caching for frequently accessed data
- Using distributed computing for large-scale deployments

## Extending the System

### Adding New Models

1. Inherit from `BaseRecommender` class
2. Implement required methods: `fit()`, `recommend()`, `get_similar_items()`
3. Add to the model ensemble in `create_recommender_ensemble()`

### Adding New Metrics

1. Add metric calculation method to `RecommendationEvaluator`
2. Include in `evaluate_model()` method
3. Update comparison and leaderboard functions

### Custom Data Sources

1. Extend `ProductDataLoader` class
2. Implement custom data loading logic
3. Ensure data follows the expected schema

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce dataset size or use sparse matrices
2. **Slow Training**: Reduce model complexity or use fewer iterations
3. **Poor Recommendations**: Check data quality and model parameters
4. **Import Errors**: Ensure all dependencies are installed correctly

### Getting Help

- Check the logs for detailed error messages
- Verify data format matches expected schema
- Ensure all dependencies are properly installed
- Review configuration parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Citation

If you use this system in your research, please cite:

```bibtex
@software{product_recommendation_system,
  title={Product Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Product-Recommendation-System}
}
```
# Product-Recommendation-System
