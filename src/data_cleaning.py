import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def standardize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    # Replace common 'missing value' tokens with actual NaN values.
    missing_tokens = ["", " ", "NA", "N/A", "None", "?", "null", "Null"]
    return df.replace(missing_tokens, np.nan)


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()
    # Loop through all columns and try converting object-type columns
    for col in df.columns:
    # Only attempt conversion if the column is stored as object (string-like)
        if df[col].dtype == object:
        # Try converting the column to numeric (e.g., "3", "4.5")
            try:    
                df[col] = pd.to_numeric(df[col])
                continue # If successful, move to next column
            except Exception:
                pass # Not numeric → try next option
        # Try converting the column to datetime (e.g., "2020-01-01", "01/31/2020")
            try:     
                sample = df[col].dropna().astype(str).head(10)
                if sample.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}").any():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:   # If parsing fails, keep original dtype
                pass

    return df


def detect_invalid_values(df: pd.DataFrame) -> None:
    # Detect and log invalid values for key columns

     # Stress level should be between 1 and 10
    if "Stress_Level (1-10)" in df.columns:
        invalid = df[(df["Stress_Level (1-10)"] < 1) | (df["Stress_Level (1-10)"] > 10)]
        logger.info("Invalid stress values: %d", len(invalid))
    
    # Sleep hours cannot be negative
    if "Sleep_Hours_per_Night" in df.columns:
        invalid = df[df["Sleep_Hours_per_Night"] < 0]
        logger.info("Invalid sleep hours: %d", len(invalid))
    
    # Total score should be between 0 and 100
    if "Total_Score" in df.columns:
        invalid = df[(df["Total_Score"] < 0) | (df["Total_Score"] > 100)]
        logger.info("Invalid total score values: %d", len(invalid))


def detect_outliers(df: pd.DataFrame) -> None:
    # Detect outliers using the IQR rule (1.5 * IQR)

    # Select only numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Calculate Q1 and Q3 for the column
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        # Define lower and upper bounds for outliers
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        # Identify rows where the value is outside the allowed range
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        # Log the number of detected outliers (if any)
        if len(outliers) > 0:
            logger.info("Outliers in %s: %d", col, len(outliers))


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()
    # Loop through all columns and handle missing values
    for col in df.columns:
        # Calculate the percentage of missing values in the column
        missing_ratio = df[col].isna().mean()
        # If more than 50% of the column is missing → drop the entire column
        if missing_ratio > 0.5:  
            df = df.drop(columns=[col])
            continue
        # If the column is numeric → fill missing values with the median
        if df[col].dtype in [np.float64, np.int64]:  
            df[col] = df[col].fillna(df[col].median())
        # If the column is categorical/string → fill missing values with the mode
        else:   
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

# Full data cleaning pipeline
def clean_full_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Apply all cleaning steps in sequence
    logger.info("Starting full dataset cleaning pipeline")
    # Standardize all missing-value tokens (e.g., 'NA', 'None', blanks → np.nan)
    df = standardize_missing_tokens(df_raw)
    # Convert columns to their correct data types (numeric, categorical, etc.)
    df = convert_column_types(df)
    # Detect invalid values (values outside expected ranges)
    detect_invalid_values(df)
    # Detect outliers in numeric columns
    detect_outliers(df)
    # Handle missing values (imputation or removal)
    df = handle_missing_values(df)
    logger.info("Final cleaned dataset shape: %s", df.shape)
    
    return df