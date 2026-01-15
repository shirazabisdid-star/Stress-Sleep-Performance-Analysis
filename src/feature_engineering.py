import pandas as pd
from sklearn.decomposition import PCA
from src.utils.logger import get_logger

logger = get_logger(__name__)

def add_interaction_term(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding interaction term: Stress_Level * Sleep_Hours_per_Night")
    df = df.copy()
    df["Stress_Sleep_Interaction"] = (
        df["Stress_Level (1-10)"] * df["Sleep_Hours_per_Night"]
    )
    return df

def add_centered_variables(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Centering stress and sleep variables")
    df = df.copy()
    df["Stress_Level_c"] = df["Stress_Level (1-10)"] - df["Stress_Level (1-10)"].mean()
    df["Sleep_Hours_c"] = (
        df["Sleep_Hours_per_Night"] - df["Sleep_Hours_per_Night"].mean()
    )
    df["Stress_Sleep_Interaction_c"] = (
        df["Stress_Level_c"] * df["Sleep_Hours_c"]
    )
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_interaction_term(df)
    df = add_centered_variables(df)
    logger.info("Feature engineering completed with columns: %s", df.columns)
    return df

