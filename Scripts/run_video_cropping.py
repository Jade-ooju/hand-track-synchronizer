import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_cropper import VideoCropper

logging.basicConfig(level=logging.INFO)

def run_cropping():
    """
    Runs video cropping for test_002 with calibrated timestamp.
    """
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\MR_View.mp4"
    json_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002"
    output_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\cropped"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calibration derived from manual inspection:
    # Shift applied: -4.24s relative to initial MTime estimate.
    # Calibrated Video Start Time (Unix Epoch): 1766488163.738
    calibrated_start = 1766488163.738
    
    print(f"Running cropping with Calibrated Start: {calibrated_start}")
    
    cropper = VideoCropper(video_path, output_dir)
    cropper.crop_to_motion_files(json_dir, start_timestamp_unix=calibrated_start)
    cropper.close()

if __name__ == "__main__":
    run_cropping()
