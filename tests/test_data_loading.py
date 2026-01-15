import pandas as pd
from src.data_loading import load_student_data

def test_load_student_data_structure():
    df = load_student_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Student_ID" in df.columns
    assert "Total_Score" in df.columns