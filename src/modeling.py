import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_interaction_regression_model(df: pd.DataFrame):
    logger.info("Building regression model with interaction")

    predictors = [
        "Stress_Level_c",
        "Sleep_Hours_c",
        "Stress_Sleep_Interaction_c"
    ]

    X = df[predictors]
    X = sm.add_constant(X)
    y = df["Total_Score"]

    model = sm.OLS(y, X).fit()
    logger.info("Model fitting completed")
    return model


def summarize_model(model) -> str:
    logger.info("Generating model summary")
    return model.summary().as_text()

