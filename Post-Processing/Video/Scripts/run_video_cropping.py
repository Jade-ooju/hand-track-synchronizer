import sys
import os
import logging
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.video_cropper import VideoCropper

logging.basicConfig(level=logging.INFO)

def run_cropping():
    """
    Runs video cropping for test_002 with calibrated timestamp.
    """
    data_root = PROJECT_ROOT / "data" / "raw" / "test_002"
    video_path = data_root / "MR_View.mp4"
    json_dir = data_root
    output_dir = data_root / "cropped"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Calibration derived from manual inspection:
    # Shift applied: -4.24s relative to initial MTime estimate.
    # Calibrated Video Start Time (Unix Epoch): 1766488163.738
    calibrated_start = 1766488163.738
    
    print(f"Running cropping with Calibrated Start: {calibrated_start}")
    
    cropper = VideoCropper(str(video_path), str(output_dir))
    cropper.crop_to_motion_files(str(json_dir), start_timestamp_unix=calibrated_start)
    cropper.close()

if __name__ == "__main__":
    run_cropping()
