import json
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_motion_data(self, file_path):
        logger.info(f"Loading motion data from {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from: {file_path}")
            return None
