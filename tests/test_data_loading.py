import pandas as pd
from src.data_loading import load_student_data

# Test that loading student data returns a DataFrame with expected structure
def test_load_student_data_structure():
    df = load_student_data()

    # Validate that the returned object is a DataFrame with expected columns
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Student_ID" in df.columns
    assert "Total_Score" in df.columns