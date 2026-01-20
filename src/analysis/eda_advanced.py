import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Advanced EDA: PCA and KMeans Clustering
def apply_pca(df, n_components=2):
    # Select only numeric columns for PCA
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    # Standardize the data before applying PCA
    pca = PCA(n_components=n_components)
    # Fit and transform the data
    components = pca.fit_transform(numeric_df)

    # Create a DataFrame for the PCA components
    pca_df = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return pca_df

# Apply KMeans clustering
def apply_kmeans(df, n_clusters=3):
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Fit and predict clusters
    clusters = kmeans.fit_predict(df)
    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters
    return df_clustered

# Plot PCA results with clusters
def plot_pca_clusters(pca_df, clusters, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=pca_df["PC1"],
        y=pca_df["PC2"],
        hue=clusters,
        palette="viridis"
    )
    plt.title("PCA + KMeans Clusters")
    plt.savefig(f"{output_dir}/eda_pca_kmeans.png")
    plt.close()