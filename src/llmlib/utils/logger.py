import logging
import sys

# Centralized logging config
logging.basicConfig(
    level=logging.INFO,  # or DEBUG for very detailed logs
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # terminal
        # logging.FileHandler("train.log"),  # optional file logging
    ],
)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance for the given module/class."""
    return logging.getLogger(name)
