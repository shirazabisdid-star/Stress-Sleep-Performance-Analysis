import pandas as pd
from src.feature_engineering import engineer_features
from src.analysis.model_analysis import compute_correlations

def test_compute_correlations_output():
    df = pd.DataFrame({
        "Stress_Level (1-10)": [3, 5, 7, 8],
        "Sleep_Hours_per_Night": [8, 7, 6, 5],
        "Total_Score": [90, 85, 80, 70],
    })

    df_features = engineer_features(df)
    result = compute_correlations(df_features)

    assert "stress_total" in result
    assert "sleep_total" in result
    assert "stress_sleep" in result

    assert isinstance(result["stress_total"], dict)
    assert isinstance(result["sleep_total"], dict)
    assert isinstance(result["stress_sleep"], dict)

    assert "corr" in result["stress_total"]
    assert "p_value" in result["stress_total"]

    assert isinstance(result["stress_total"]["corr"], float)
    assert isinstance(result["stress_total"]["p_value"], float)