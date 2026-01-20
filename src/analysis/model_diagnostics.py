import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_regression_diagnostics(model, output_dir="figures"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # extract residuals and fitted (predicted) values from the model
    residuals = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(10, 4))
    # Residuals vs Fitted plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, color="red")
    plt.title("Residuals vs Fitted")
    # Histogram of residuals
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regression_diagnostics.png")
    plt.close()


def compute_vif(df, features):
    # add constant column for VIF computation
    X = df[features].assign(const=1)
    # create empty DataFrame for VIF results
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    # compute VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif