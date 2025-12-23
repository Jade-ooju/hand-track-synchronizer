import logging
import os
import json

def setup_logging(config_path="config/config.json"):
    """
    Sets up the logging configuration.
    """
    default_level = logging.INFO
    log_file = "pipeline.log"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            log_config = config.get('logging', {})
            level_str = log_config.get('level', 'INFO')
            default_level = getattr(logging, level_str.upper(), logging.INFO)
            log_file = log_config.get('file', 'pipeline.log')

    logging.basicConfig(
        level=default_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")
