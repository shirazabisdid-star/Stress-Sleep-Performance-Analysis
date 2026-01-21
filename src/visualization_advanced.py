import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger
logger = get_logger(__name__)

# Prepare ranges for interaction plotting
def prepare_interaction_ranges(df):
    stress_range = np.linspace(
        df["Stress_Level_c"].min(),
        df["Stress_Level_c"].max(),
        50
    )
# Determine representative sleep levels (25th, 50th, 75th percentiles)
    sleep_levels = [
        df["Sleep_Hours_c"].quantile(0.25),
        df["Sleep_Hours_c"].quantile(0.5),
        df["Sleep_Hours_c"].quantile(0.75),
    ]
    return stress_range, sleep_levels

# Predict interaction effect
def predict_interaction(model, stress_range, sleep_level):
    interaction = stress_range * sleep_level
# Create DataFrame for prediction
    X_pred = model.model.data.orig_exog.iloc[:len(stress_range)].copy()
    X_pred["Stress_Level_c"] = stress_range
    X_pred["Sleep_Hours_c"] = sleep_level
    X_pred["Stress_Sleep_Interaction_c"] = interaction
    X_pred["const"] = 1.0

    return model.predict(X_pred)

# Plot interaction effect
def plot_interaction_effect(df, model, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Plotting interaction effect")
    # Prepare ranges for plotting 
    stress_range, sleep_levels = prepare_interaction_ranges(df)

    # Set up plot aesthetics
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    plt.figure(figsize=(10, 6))

    # Plot predictions for each sleep level
    for i, sleep_level in enumerate(sleep_levels):
        y_pred = predict_interaction(model, stress_range, sleep_level)
        label = f"Sleep (centered) = {sleep_level:.2f}"
        plt.plot(
            stress_range,
            y_pred,
            label=label,
            color=palette[i],
            linewidth=3
        )

    # Finalize plot details
    plt.xlabel("Stress Level (centered)")
    plt.ylabel("Predicted Total Score", rotation=0, labelpad=40)
    plt.title("Interaction: Stress × Sleep")
    plt.legend(title="Sleep Level", frameon=True, facecolor="white", framealpha=1)
    plt.grid(alpha=0.3)

    path = f"{output_dir}/interaction_effect.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved interaction effect plot to %s", path)