import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

def eda_distributions(df, output_dir="figures"):
    """Basic distributions and outliers."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[["Stress_Level_c", "Sleep_Hours_c"]])
    plt.title("Stress & Sleep Distributions")
    plt.savefig(f"{output_dir}/eda_distributions.png")
    plt.close()

def eda_pairplot(df, output_dir="figures"):
    """Quick look at linearity between variables."""
    os.makedirs(output_dir, exist_ok=True)
    sns.pairplot(df[["Stress_Level_c", "Sleep_Hours_c", "Total_Score"]], kind="reg")
    plt.savefig(f"{output_dir}/eda_pairplot.png")
    plt.close()

def eda_correlation(df, output_dir="figures"):
    """Correlation heatmap."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_dir}/eda_correlation.png")
    plt.close()

