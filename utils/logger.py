# ============================================================
# 🗂️  utils/logger.py
# ============================================================
"""Configure Python ``logging`` to output both to console & file."""
import logging
import sys
from datetime import datetime
from config import LOG_DIR

FMT = "%(asctime)s | %(levelname)8s | %(name)s: %(message)s"


def get_logger(name: str = "cattle_id") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)

    # —— console handler ------------------------------------------------
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter(FMT))
    logger.addHandler(c_handler)

    # —— file handler ---------------------------------------------------
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file    = LOG_DIR / f"run_{timestamp}.log"
    f_handler   = logging.FileHandler(log_file)
    f_handler.setFormatter(logging.Formatter(FMT))
    logger.addHandler(f_handler)

    logger.info("Logger initialized – logs will be saved to %s", log_file)
    return logger