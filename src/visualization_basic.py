import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)

def plot_distributions(df: pd.DataFrame, output_dir: str = "figures"):
    # Ensure the output directory exists (create if missing)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Plotting distributions of stress and sleep")
    # Create a single figure with two side-by-side subplots
    plt.figure(figsize=(10, 4))

    # Left plot: distribution of stress levels (1-10)
    plt.subplot(1, 2, 1)
    sns.histplot(df["Stress_Level (1-10)"], kde=True)
    plt.title("Stress Level Distribution")

    # Right plot: distribution of sleep hours per night
    plt.subplot(1, 2, 2)
    sns.histplot(df["Sleep_Hours_per_Night"], kde=True)
    plt.title("Sleep Hours per Night Distribution")

    # Save the combined figure
    path = f"{output_dir}/stress_sleep_distributions.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved distributions plot to %s", path)