import pandas as pd
from scipy.stats import pearsonr
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_correlations(df: pd.DataFrame) -> dict:
    # log start of correlation computation
    logger.info("Computing Pearson correlations")
    # Correlation: stress vs total score
    stress_total_corr, stress_total_p = pearsonr(
        df["Stress_Level_c"],
        df["Total_Score"]
    )
    # Correlation: sleep vs total score
    sleep_total_corr, sleep_total_p = pearsonr(
        df["Sleep_Hours_c"],
        df["Total_Score"]
    )
    # Correlation: stress vs sleep
    stress_sleep_corr, stress_sleep_p = pearsonr(
        df["Stress_Level_c"],
        df["Sleep_Hours_c"]
    )
    # return all correlations and p-values in a dictionary
    return {
        "stress_total": {
            "corr": stress_total_corr,
            "p_value": stress_total_p
        },
        "sleep_total": {
            "corr": sleep_total_corr,
            "p_value": sleep_total_p
        },
        "stress_sleep": {
            "corr": stress_sleep_corr,
            "p_value": stress_sleep_p
        },
    }

def extract_model_effects(model) -> dict:
    # log start of model coefficient extraction
    logger.info("Extracting model coefficients and p-values")
    params = model.params
    pvalues = model.pvalues
    # return coefficients and p-values for each predictor
    return {
        "beta_stress": {
            "coef": params.get("Stress_Level_c"),
            "p_value": pvalues.get("Stress_Level_c")
        },
        "beta_sleep": {
            "coef": params.get("Sleep_Hours_c"),
            "p_value": pvalues.get("Sleep_Hours_c")
        },
        "beta_interaction": {
            "coef": params.get("Stress_Sleep_Interaction_c"),
            "p_value": pvalues.get("Stress_Sleep_Interaction_c")
        },
    }