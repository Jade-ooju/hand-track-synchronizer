import sys
import os
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.video_cropper import VideoCropper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cropping():
    """
    Crops the Utensil video into clips aligned with each motion JSON file.
    """
    data_root = PROJECT_ROOT / "data" / "raw" / "Utensil"
    video_path = data_root / "Utensil_Retake_001.mp4"
    json_dir = data_root
    output_dir = data_root / "cropped"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Use the same timestamp offset as the pipeline config
    # This is the first motion timestamp from the earliest JSON file
    calibrated_start = 1767431293.0344615
    
    logger.info(f"Running video cropping for Utensil data")
    logger.info(f"Video: {video_path}")
    logger.info(f"JSON Directory: {json_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Calibrated Start Time (Unix): {calibrated_start}")
    
    cropper = VideoCropper(str(video_path), str(output_dir))
    cropper.crop_to_motion_files(str(json_dir), start_timestamp_unix=calibrated_start)
    cropper.close()
    
    logger.info("Video cropping complete!")

if __name__ == "__main__":
    run_cropping()

