import os
import pandas as pd
from src.utils.logger import get_logger

# Create a logger for this module
logger = get_logger(__name__)

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Default path to the dataset
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Students_Grading_Dataset_Biased.csv")

def load_student_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the student dataset from a CSV file."""

    # Verify the file exists before loading
    if not os.path.exists(data_path):
        logger.error("Data file not found at path: %s", data_path)
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Log the loading process
    logger.info("Loading data from %s", data_path)
    # Read the CSV into a DataFrame
    df = pd.read_csv(data_path)
     # Log the shape of the loaded dataset
    logger.info("Data loaded successfully with shape %s", df.shape)
    return df