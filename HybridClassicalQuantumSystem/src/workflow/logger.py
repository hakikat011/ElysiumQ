import logging

def setup_logger():
    logger = logging.getLogger('HybridSystem')
    logger.setLevel(logging.INFO)
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
