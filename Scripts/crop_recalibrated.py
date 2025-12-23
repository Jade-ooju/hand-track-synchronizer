import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_cropper import VideoCropper

logging.basicConfig(level=logging.INFO)

def run_recalibrated_crop():
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\MR_View.mp4"
    json_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002"
    output_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\cropped_recalib"
    
    # Calibration derived from User Feedback:
    # Content for J1 (927) was found in J2 (932).
    # J1 needs to point to the video segment currently assigned to J2.
    # Gap between J1 and J2 is ~4.24s.
    # Start_New = Start_Old - 4.24
    
    prev_calib = 1766488167.976
    diff = 4.2378
    calibrated_start = prev_calib - diff
    
    print(f"Recalibrated Start: {calibrated_start}")
    
    cropper = VideoCropper(video_path, output_dir)
    cropper.crop_to_motion_files(json_dir, start_timestamp_unix=calibrated_start)
    cropper.close()

if __name__ == "__main__":
    run_recalibrated_crop()
