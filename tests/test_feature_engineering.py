import pandas as pd
from src.feature_engineering import engineer_features

def test_engineer_features_creates_interaction():
    df = pd.DataFrame({
        "Stress_Level (1-10)": [5, 7],
        "Sleep_Hours_per_Night": [7, 6],
        "Total_Score": [85, 90],
    })

    df_features = engineer_features(df)

    assert "Stress_Sleep_Interaction" in df_features.columns
    assert len(df_features) == len(df)