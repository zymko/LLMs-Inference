import logging

def create_logger(name, log_file="run.log", log_level=logging.INFO):
    """
    Create and return a logger object.
    
    Args:
        name (str): Name of the logger, typically the module name.
        log_file (str): Path to the log file.
        log_level: Logging level, e.g., logging.INFO, logging.DEBUG, etc.

    Returns:
        logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Define the logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
