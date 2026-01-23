import logging
import os
from datetime import datetime

# Utility function to set up a logger
def get_logger(name: str) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    # Create a log file with a timestamp
    log_filename = os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y%m%d')}_project.log"
    )
    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        console_handler = logging.StreamHandler()
        # Set formatter for both handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Apply the formatter to both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger