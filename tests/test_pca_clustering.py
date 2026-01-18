import pandas as pd
from src.feature_engineering import engineer_features
from src.analysis.eda_advanced import apply_pca, apply_kmeans

# Test that PCA and KMeans clustering run without errors and produce expected output
def test_pca_and_kmeans_pipeline():
    # Create a small sample DataFrame for testing
    df = pd.DataFrame({
        "Stress_Level (1-10)": [3, 5, 7, 8],
        "Sleep_Hours_per_Night": [8, 7, 6, 5],
        "Total_Score": [90, 85, 80, 70],
    })

    # Apply feature engineering to generate new variables
    df_features = engineer_features(df)
    pca_df = apply_pca(df_features)
    clustered = apply_kmeans(pca_df)

    # Validate the shapes and presence of expected columns
    assert len(pca_df) == len(df_features)
    assert "Cluster" in clustered.columns