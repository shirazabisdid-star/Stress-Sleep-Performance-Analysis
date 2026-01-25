import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Build the exogenous variables for prediction (Stress, Sleep, Interaction, Constant)
def prepare_exog(stress_range, sleep_level):
    """
    Create a DataFrame containing the stress range, a fixed sleep level,
    and their interaction term, including a constant for regression prediction.
    """
    exog = pd.DataFrame({
        'Stress_Level_c': stress_range,
        'Sleep_Hours_c': sleep_level
    })

    # Add interaction term
    exog['Stress_Sleep_Interaction_c'] = (
        exog['Stress_Level_c'] * exog['Sleep_Hours_c']
    )

    # Add constant column for the regression model
    exog = sm.add_constant(exog, has_constant='add')

    return exog


# Prepare ranges for plotting the interaction effect
def prepare_interaction_ranges(df):
    """
    Generate a continuous range of stress values and representative
    sleep levels (25th, 50th, 75th percentiles) for interaction plotting.
    """
    stress_range = np.linspace(
        df["Stress_Level_c"].min(),
        df["Stress_Level_c"].max(),
        50
    )

    # Representative sleep levels
    sleep_levels = [
        df["Sleep_Hours_c"].quantile(0.25),
        df["Sleep_Hours_c"].quantile(0.5),
        df["Sleep_Hours_c"].quantile(0.75),
    ]

    return stress_range, sleep_levels


# Plot the interaction effect with confidence intervals
def plot_interaction_effect(df, model, output_dir="figures"):
    """
    Plot the interaction between Stress and Sleep, including 95% confidence intervals.
    Saves the figure to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Plotting interaction effect with confidence intervals")

    stress_range, sleep_levels = prepare_interaction_ranges(df)

    plt.figure(figsize=(8, 6))

    for sleep_level in sleep_levels:
        # Build exogenous variables for prediction
        exog = prepare_exog(stress_range, sleep_level)

        # Get prediction with confidence intervals
        predictions = model.get_prediction(exog)
        frame = predictions.summary_frame(alpha=0.05)

        # Extract mean prediction and confidence intervals
        y_pred = frame['mean']
        y_lower = frame['mean_ci_lower']   # CI for mean prediction
        y_upper = frame['mean_ci_upper']

        label = f"Sleep (centered) = {sleep_level:.2f}"

        # Plot prediction line
        line, = plt.plot(stress_range, y_pred, label=label)

        # Plot confidence interval band
        plt.fill_between(
            stress_range,
            y_lower,
            y_upper,
            color=line.get_color(),
            alpha=0.1
        )

    # Set y-axis limits based on observed data
    plt.ylim(df['Total_Score'].min(), df['Total_Score'].max())

    plt.xlabel("Stress Level (centered)")
    plt.ylabel("Total Score")
    plt.title("Interaction: Stress × Sleep (with 95% CI)")
    plt.legend()

    path = f"{output_dir}/interaction_effect_v2.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    logger.info("Saved improved interaction plot to %s", path)