import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_cropper import VideoCropper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cropping():
    """
    Crops the Utensil video into clips aligned with each motion JSON file.
    """
    video_path = os.path.join("data", "raw", "Utensil", "Utensil_Retake_001.mp4")
    json_dir = os.path.join("data", "raw", "Utensil")
    output_dir = os.path.join("data", "raw", "Utensil", "cropped")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use the same timestamp offset as the pipeline config
    # This is the first motion timestamp from the earliest JSON file
    calibrated_start = 1767431293.0344615
    
    logger.info(f"Running video cropping for Utensil data")
    logger.info(f"Video: {video_path}")
    logger.info(f"JSON Directory: {json_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Calibrated Start Time (Unix): {calibrated_start}")
    
    cropper = VideoCropper(video_path, output_dir)
    cropper.crop_to_motion_files(json_dir, start_timestamp_unix=calibrated_start)
    cropper.close()
    
    logger.info("Video cropping complete!")

if __name__ == "__main__":
    run_cropping()

