import os
import seaborn as sns
import matplotlib.pyplot as plt

def eda_score_components(df, output_dir="figures"):
    """
    Perform EDA on the components contributing to Total_Score.
    Generates distribution plots and a correlation heatmap.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Columns representing different score components
    score_cols = [
        "Midterm_Score",
        "Final_Score",
        "Assignments_Avg",
        "Quizzes_Avg",
        "Participation_Score",
        "Projects_Score",
        "Total_Score"
    ]

    # 1) Plot distribution of each score component
    for col in score_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dist_{col}.png")
        plt.close()

    # 2) Correlation heatmap of score components
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[score_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Between Score Components")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/score_components_corr.png")
    plt.close()