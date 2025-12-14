from pathlib import Path
import pandas as pd
import numpy as np
from ej2c5 import (
    read_csv,
    clean_dataframe,
    dropna_specific_row_in_column,
    fillna_method,
)  # Asumiendo que tu script se llama tu_script.py


def test_read_csv():
    current_dir = Path(__file__).parent
    FILE_PATH = current_dir / "data/grades.csv"
    df = read_csv(FILE_PATH)
    assert not df.empty, "DataFrame should not be empty after reading the CSV file"


def test_clean_dataframe():
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "-", "Charlie", "Null"],
            "Maths": ["95", "na", "85", "Null", "80"],
            "Physics": ["90", "85", "-", "na", "75"],
        }
    )
    cleaned_df = clean_dataframe(df)
    assert (
        cleaned_df.isnull().sum().sum() <= 6
    ), "DataFrame should have NaN values after cleaning"
    assert (
        cleaned_df.dtypes["Maths"] == np.float64
    ), "'Maths' column should be converted to float type"


def test_dropna_specific_row_in_column():
    df = pd.DataFrame(
        {
            "Name": ["Alice", np.nan, "Bob", "Charlie"],
            "Maths": [95, 85, np.nan, 80],
        }
    )
    df_cleaned = dropna_specific_row_in_column(df, "Name")
    assert (
        len(df_cleaned) == 3
    ), "DataFrame should have fewer rows after dropping rows with NaN in 'Name'"


def test_fillna_method_ffill():
    df = pd.DataFrame(
        {
            "Maths": [np.nan, 85, np.nan, 80],
        }
    )
    df_filled = fillna_method(df, "Maths", fill_method="ffill")
    assert (
        df_filled.iloc[2]["Maths"] == 85
    ), "NaN value should have been forward-filled with previous value"


def test_fillna_method_mean():
    df = pd.DataFrame(
        {
            "Maths": [90, 80, np.nan, 80],
        }
    )
    df_filled = fillna_method(df, "Maths", fill_method="mean")
    assert (
        df_filled.iloc[2]["Maths"] == 83.33333333333333
    ), "NaN value should have been filled with the column's mean"
