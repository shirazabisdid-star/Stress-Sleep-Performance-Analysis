import pandas as pd
import statsmodels.api as sm
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_interaction_regression_model(df: pd.DataFrame):
    # Build a linear regression model including the interaction term

    logger.info("Building regression model with interaction")

    # Define predictors: centered stress, centered sleep, and their interaction
    predictors = [
        "Stress_Level_c",
        "Sleep_Hours_c",
        "Stress_Sleep_Interaction_c"
    ]

    # Extract predictor matrix and add constant term
    X = df[predictors]
    X = sm.add_constant(X)
    # Define dependent variable
    y = df["Total_Score"]

    # Fit OLS regression model
    model = sm.OLS(y, X).fit()
    logger.info("Model fitting completed")
    return model


def summarize_model(model) -> str:
    # Return a text summary of the fitted regression model
    logger.info("Generating model summary")
    return model.summary().as_text()

