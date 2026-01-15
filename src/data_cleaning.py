import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def standardize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    missing_tokens = ["", " ", "NA", "N/A", "None", "?", "null", "Null"]
    return df.replace(missing_tokens, np.nan)


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
                continue
            except Exception:
                pass
            try:
                sample = df[col].dropna().astype(str).head(10)
                if sample.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}").any():
                    df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df


def detect_invalid_values(df: pd.DataFrame) -> None:
    if "Stress_Level (1-10)" in df.columns:
        invalid = df[(df["Stress_Level (1-10)"] < 1) | (df["Stress_Level (1-10)"] > 10)]
        logger.info("Invalid stress values: %d", len(invalid))
    if "Sleep_Hours_per_Night" in df.columns:
        invalid = df[df["Sleep_Hours_per_Night"] < 0]
        logger.info("Invalid sleep hours: %d", len(invalid))
    if "Total_Score" in df.columns:
        invalid = df[(df["Total_Score"] < 0) | (df["Total_Score"] > 100)]
        logger.info("Invalid total score values: %d", len(invalid))


def detect_outliers(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        
        if len(outliers) > 0:
            logger.info("Outliers in %s: %d", col, len(outliers))


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        if missing_ratio > 0.5:
            df = df.drop(columns=[col])
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def clean_full_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting full dataset cleaning pipeline")
    df = standardize_missing_tokens(df_raw)
    df = convert_column_types(df)
    detect_invalid_values(df)
    detect_outliers(df)
    df = handle_missing_values(df)
    logger.info("Final cleaned dataset shape: %s", df.shape)
    return df