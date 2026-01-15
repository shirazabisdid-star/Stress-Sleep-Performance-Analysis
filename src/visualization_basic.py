import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_distributions(df: pd.DataFrame, output_dir: str = "figures"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Plotting distributions of stress and sleep")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(df["Stress_Level (1-10)"], kde=True)
    plt.title("Stress Level Distribution")

    plt.subplot(1, 2, 2)
    sns.histplot(df["Sleep_Hours_per_Night"], kde=True)
    plt.title("Sleep Hours per Night Distribution")

    path = f"{output_dir}/stress_sleep_distributions.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved distributions plot to %s", path)


