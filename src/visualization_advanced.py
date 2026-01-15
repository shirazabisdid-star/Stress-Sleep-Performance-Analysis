import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
logger = get_logger(__name__)

def prepare_interaction_ranges(df):
    stress_range = np.linspace(
        df["Stress_Level_c"].min(),
        df["Stress_Level_c"].max(),
        50
    )

    sleep_levels = [
        df["Sleep_Hours_c"].quantile(0.25),
        df["Sleep_Hours_c"].quantile(0.5),
        df["Sleep_Hours_c"].quantile(0.75),
    ]

    return stress_range, sleep_levels

def predict_interaction(model, stress_range, sleep_level):
    interaction = stress_range * sleep_level

    X_pred = model.model.data.orig_exog.iloc[:len(stress_range)].copy()
    X_pred["Stress_Level_c"] = stress_range
    X_pred["Sleep_Hours_c"] = sleep_level
    X_pred["Stress_Sleep_Interaction_c"] = interaction
    X_pred["const"] = 1.0

    return model.predict(X_pred)

def plot_interaction_effect(df, model, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Plotting interaction effect")

    stress_range, sleep_levels = prepare_interaction_ranges(df)

    plt.figure(figsize=(8, 6))

    for sleep_level in sleep_levels:
        y_pred = predict_interaction(model, stress_range, sleep_level)
        label = f"Sleep (centered) = {sleep_level:.2f}"
        plt.plot(stress_range, y_pred, label=label)

    plt.xlabel("Stress Level (centered)")
    plt.ylabel("Predicted Total Score")
    plt.title("Interaction: Stress × Sleep")
    plt.legend()

    path = f"{output_dir}/interaction_effect.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved interaction effect plot to %s", path)