import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_regression_diagnostics(model, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    residuals = model.resid
    fitted = model.fittedvalues

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, color="red")
    plt.title("Residuals vs Fitted")

    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/regression_diagnostics.png")
    plt.close()


def compute_vif(df, features):
    X = df[features].assign(const=1)
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif