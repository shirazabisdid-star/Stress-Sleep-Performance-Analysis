import pandas as pd
from src.feature_engineering import engineer_features
from src.analysis.model_analysis import compute_correlations

def test_compute_correlations_output():
    # Create a small sample DataFrame for testing
    df = pd.DataFrame({
        "Stress_Level (1-10)": [3, 5, 7, 8],
        "Sleep_Hours_per_Night": [8, 7, 6, 5],
        "Total_Score": [90, 85, 80, 70],
    })

    # Apply feature engineering to generate new variables
    df_features = engineer_features(df)
    # Compute correlations between engineered features and Total_Score
    result = compute_correlations(df_features)

    # Verify expected keys exist in the result dictionary
    assert "stress_total" in result
    assert "sleep_total" in result
    assert "stress_sleep" in result

    #  Ensure each entry is a dictionary
    assert isinstance(result["stress_total"], dict)
    assert isinstance(result["sleep_total"], dict)
    assert isinstance(result["stress_sleep"], dict)

    # Check that correlation and p-value are present and of correct type
    assert "corr" in result["stress_total"]
    assert "p_value" in result["stress_total"]

    # Validate that correlation and p-value are floats
    assert isinstance(result["stress_total"]["corr"], float)
    assert isinstance(result["stress_total"]["p_value"], float)