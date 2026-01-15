import pandas as pd
from src.data_cleaning import clean_full_dataset

def test_clean_full_dataset_keeps_shape():
    df = pd.DataFrame({
        "Student_ID": [1, 2],
        "Attendance (%)": [95, -10],
        "Stress_Level (1-10)": [5, 7],
        "Sleep_Hours_per_Night": [7, 6],
        "Total_Score": [85, 90],
    })

    df_clean = clean_full_dataset(df)

    assert len(df_clean) == 2
    assert "Attendance (%)" in df_clean.columns