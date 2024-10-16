import logging
import os

def setup_logger(log_file='endpoint.log', log_level=logging.INFO):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Full path for the log file
    log_path = os.path.join(log_dir, log_file)
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create a file handler that appends to the existing file
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setLevel(log_level)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    return logger

# # Usage in main.py or other files
# logger = setup_logger()

# # Example usage
# logger.info("Program started")
# logger.debug("This is a debug message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")