"""Streamlit demo application for product recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.loader import ProductDataLoader, set_random_seeds
from models.recommenders import create_recommender_ensemble
from evaluation.metrics import RecommendationEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-item {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache data."""
    set_random_seeds(42)
    loader = ProductDataLoader()
    return loader.load_data()


@st.cache_data
def train_models(products_df, users_df, interactions_df):
    """Train and cache models."""
    from data.loader import create_train_test_split
    
    train_df, test_df = create_train_test_split(interactions_df)
    
    models = create_recommender_ensemble()
    for model in models.values():
        model.fit(train_df, products_df, users_df)
    
    return models, train_df, test_df


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛍️ Product Recommendation System</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        products_df, users_df, interactions_df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "User Recommendations", "Product Similarity", "Model Comparison", "Data Analysis"]
    )
    
    if page == "Overview":
        show_overview(products_df, users_df, interactions_df)
    elif page == "User Recommendations":
        show_user_recommendations(products_df, users_df, interactions_df)
    elif page == "Product Similarity":
        show_product_similarity(products_df, users_df, interactions_df)
    elif page == "Model Comparison":
        show_model_comparison(products_df, users_df, interactions_df)
    elif page == "Data Analysis":
        show_data_analysis(products_df, users_df, interactions_df)


def show_overview(products_df, users_df, interactions_df):
    """Show overview dashboard."""
    st.header("System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", len(products_df))
    
    with col2:
        st.metric("Total Users", len(users_df))
    
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    
    with col4:
        avg_interactions = len(interactions_df) / len(users_df)
        st.metric("Avg Interactions per User", f"{avg_interactions:.1f}")
    
    # Data distribution charts
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Product categories
        category_counts = products_df['category'].value_counts()
        fig_categories = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Product Categories Distribution"
        )
        st.plotly_chart(fig_categories, use_container_width=True)
    
    with col2:
        # Interaction types
        interaction_counts = interactions_df['interaction_type'].value_counts()
        fig_interactions = px.bar(
            x=interaction_counts.index,
            y=interaction_counts.values,
            title="Interaction Types Distribution"
        )
        st.plotly_chart(fig_interactions, use_container_width=True)
    
    # Price distribution
    st.subheader("Price Analysis")
    fig_price = px.histogram(
        products_df,
        x='price',
        nbins=30,
        title="Product Price Distribution"
    )
    st.plotly_chart(fig_price, use_container_width=True)


def show_user_recommendations(products_df, users_df, interactions_df):
    """Show user recommendation interface."""
    st.header("User Recommendations")
    
    # Train models
    with st.spinner("Training models..."):
        models, train_df, test_df = train_models(products_df, users_df, interactions_df)
    
    # User selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_user = st.selectbox(
            "Select a user:",
            options=users_df['user_id'].tolist(),
            format_func=lambda x: f"{x} (Age: {users_df[users_df['user_id']==x]['age'].iloc[0]})"
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
    
    # Model selection
    model_name = st.selectbox(
        "Select recommendation model:",
        options=list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    model = models[model_name]
    
    # Get user info
    user_info = users_df[users_df['user_id'] == selected_user].iloc[0]
    
    # Display user profile
    st.subheader("User Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Age", user_info['age'])
    
    with col2:
        st.metric("Gender", user_info['gender'])
    
    with col3:
        st.metric("Location", user_info['location'])
    
    with col4:
        preferred_cats = ', '.join(user_info['preferred_categories'])
        st.metric("Preferred Categories", preferred_cats)
    
    # Get recommendations
    try:
        recommendations = model.recommend(selected_user, n_recommendations)
        
        st.subheader(f"Recommendations ({model_name.replace('_', ' ').title()})")
        
        for i, (product_id, score) in enumerate(recommendations, 1):
            product_info = products_df[products_df['product_id'] == product_id].iloc[0]
            
            with st.container():
                st.markdown(f"""
                <div class="recommendation-item">
                    <h4>{i}. {product_info['title']}</h4>
                    <p><strong>Category:</strong> {product_info['category']}</p>
                    <p><strong>Brand:</strong> {product_info['brand']}</p>
                    <p><strong>Price:</strong> ${product_info['price']}</p>
                    <p><strong>Rating:</strong> {product_info['rating']}/5.0</p>
                    <p><strong>Recommendation Score:</strong> {score:.3f}</p>
                    <p><strong>Description:</strong> {product_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")


def show_product_similarity(products_df, users_df, interactions_df):
    """Show product similarity interface."""
    st.header("Product Similarity")
    
    # Train models
    with st.spinner("Training models..."):
        models, train_df, test_df = train_models(products_df, users_df, interactions_df)
    
    # Product selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_product = st.selectbox(
            "Select a product:",
            options=products_df['product_id'].tolist(),
            format_func=lambda x: products_df[products_df['product_id']==x]['title'].iloc[0]
        )
    
    with col2:
        n_similar = st.slider("Number of similar products:", 5, 15, 10)
    
    # Model selection
    model_name = st.selectbox(
        "Select similarity model:",
        options=list(models.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    model = models[model_name]
    
    # Display selected product
    product_info = products_df[products_df['product_id'] == selected_product].iloc[0]
    
    st.subheader("Selected Product")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Category", product_info['category'])
        st.metric("Brand", product_info['brand'])
        st.metric("Price", f"${product_info['price']}")
        st.metric("Rating", f"{product_info['rating']}/5.0")
    
    with col2:
        st.write("**Description:**")
        st.write(product_info['description'])
    
    # Get similar products
    try:
        similar_products = model.get_similar_items(selected_product, n_similar)
        
        st.subheader(f"Similar Products ({model_name.replace('_', ' ').title()})")
        
        for i, (product_id, similarity_score) in enumerate(similar_products, 1):
            similar_info = products_df[products_df['product_id'] == product_id].iloc[0]
            
            with st.container():
                st.markdown(f"""
                <div class="recommendation-item">
                    <h4>{i}. {similar_info['title']}</h4>
                    <p><strong>Category:</strong> {similar_info['category']}</p>
                    <p><strong>Brand:</strong> {similar_info['brand']}</p>
                    <p><strong>Price:</strong> ${similar_info['price']}</p>
                    <p><strong>Rating:</strong> {similar_info['rating']}/5.0</p>
                    <p><strong>Similarity Score:</strong> {similarity_score:.3f}</p>
                    <p><strong>Description:</strong> {similar_info['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error finding similar products: {e}")


def show_model_comparison(products_df, users_df, interactions_df):
    """Show model comparison and evaluation."""
    st.header("Model Comparison")
    
    # Train models
    with st.spinner("Training models..."):
        models, train_df, test_df = train_models(products_df, users_df, interactions_df)
    
    # Evaluation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        k_values = st.multiselect(
            "Select k values for evaluation:",
            options=[5, 10, 15, 20],
            default=[5, 10, 20]
        )
    
    with col2:
        n_recommendations = st.slider("Number of recommendations for evaluation:", 10, 50, 20)
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating models..."):
            evaluator = RecommendationEvaluator()
            results_df = evaluator.compare_models(models, test_df, products_df, k_values)
            
            # Display results
            st.subheader("Evaluation Results")
            st.dataframe(results_df.round(4), use_container_width=True)
            
            # Create leaderboard
            leaderboard = evaluator.create_leaderboard(results_df)
            
            st.subheader("Model Leaderboard")
            st.dataframe(leaderboard.round(4), use_container_width=True)
            
            # Visualization
            st.subheader("Performance Visualization")
            
            # Select metrics to plot
            metric_cols = [col for col in results_df.columns if col != 'model']
            selected_metrics = st.multiselect(
                "Select metrics to visualize:",
                options=metric_cols,
                default=metric_cols[:4]
            )
            
            if selected_metrics:
                # Create subplots
                fig = make_subplots(
                    rows=len(selected_metrics),
                    cols=1,
                    subplot_titles=selected_metrics,
                    vertical_spacing=0.1
                )
                
                for i, metric in enumerate(selected_metrics, 1):
                    fig.add_trace(
                        go.Bar(
                            x=results_df['model'],
                            y=results_df[metric],
                            name=metric,
                            showlegend=False
                        ),
                        row=i,
                        col=1
                    )
                
                fig.update_layout(
                    height=200 * len(selected_metrics),
                    title_text="Model Performance Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_data_analysis(products_df, users_df, interactions_df):
    """Show data analysis and insights."""
    st.header("Data Analysis")
    
    # User analysis
    st.subheader("User Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            users_df,
            x='age',
            nbins=20,
            title="User Age Distribution"
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_counts = users_df['gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="User Gender Distribution"
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Location analysis
    st.subheader("Geographic Distribution")
    location_counts = users_df['location'].value_counts()
    fig_location = px.bar(
        x=location_counts.index,
        y=location_counts.values,
        title="Users by Location"
    )
    st.plotly_chart(fig_location, use_container_width=True)
    
    # Interaction analysis
    st.subheader("Interaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactions over time
        interactions_df['date'] = pd.to_datetime(interactions_df['timestamp']).dt.date
        daily_interactions = interactions_df.groupby('date').size().reset_index(name='count')
        
        fig_time = px.line(
            daily_interactions,
            x='date',
            y='count',
            title="Daily Interactions Over Time"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Top products by interactions
        top_products = interactions_df['product_id'].value_counts().head(10)
        top_product_names = []
        for product_id in top_products.index:
            product_name = products_df[products_df['product_id'] == product_id]['title'].iloc[0]
            top_product_names.append(product_name)
        
        fig_top = px.bar(
            x=top_product_names,
            y=top_products.values,
            title="Top 10 Products by Interactions"
        )
        fig_top.update_xaxis(tickangle=45)
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Category analysis
    st.subheader("Category Analysis")
    
    # Category popularity
    category_interactions = interactions_df.merge(
        products_df[['product_id', 'category']],
        on='product_id'
    )['category'].value_counts()
    
    fig_category_pop = px.bar(
        x=category_interactions.index,
        y=category_interactions.values,
        title="Category Popularity (by Interactions)"
    )
    st.plotly_chart(fig_category_pop, use_container_width=True)
    
    # Price vs popularity
    product_interactions = interactions_df['product_id'].value_counts()
    product_popularity = products_df.merge(
        product_interactions.rename('interaction_count'),
        left_on='product_id',
        right_index=True,
        how='left'
    ).fillna(0)
    
    fig_price_pop = px.scatter(
        product_popularity,
        x='price',
        y='interaction_count',
        color='category',
        title="Product Price vs Popularity",
        hover_data=['title', 'brand']
    )
    st.plotly_chart(fig_price_pop, use_container_width=True)


if __name__ == "__main__":
    main()
