import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "Students_Grading_Dataset_Biased.csv")

def load_student_data(data_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        logger.error("Data file not found at path: %s", data_path)
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Data loaded successfully with shape %s", df.shape)
    return df