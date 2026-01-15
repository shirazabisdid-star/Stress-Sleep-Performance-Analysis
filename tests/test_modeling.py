import pandas as pd
from src.feature_engineering import engineer_features
from src.modeling import build_interaction_regression_model

def test_build_interaction_regression_model_runs():
    df = pd.DataFrame({
        "Stress_Level (1-10)": [3, 5, 7, 8],
        "Sleep_Hours_per_Night": [8, 7, 6, 5],
        "Total_Score": [90, 85, 80, 70],
    })

    df_features = engineer_features(df)
    model = build_interaction_regression_model(df_features)

    assert hasattr(model, "params")
    assert len(model.params) > 0