"""
Improved video cropping that accounts for recording start delay.
Finds when recording actually begins and adjusts cropping accordingly.
"""
import sys
import os
import cv2
import numpy as np
import logging
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.video_cropper import VideoCropper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_recording_indicator(frame):
    """Detects green recording indicator."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = np.sum(mask > 0)
    return green_pixel_count > 100

def find_recording_start(video_path, max_frames=2000):
    """Finds when recording actually starts in the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if detect_recording_indicator(frame):
            cap.release()
            return frame_count, timestamp_ms / 1000.0  # Return in seconds
        
        frame_count += 1
    
    cap.release()
    logger.warning("Could not find recording start. Using frame 0.")
    return 0, 0.0

def get_unix_timestamp_from_frame(video_path, frame_number):
    """
    Extracts frame and prompts user for UNIX timestamp.
    In the future, this could use OCR.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Save frame for user to check
    temp_frame_path = "recording_start_frame.png"
    cv2.imwrite(temp_frame_path, frame)
    
    logger.info(f"Saved recording start frame to: {temp_frame_path}")
    logger.info("Please open this image and read the UNIX timestamp from the upper left corner.")
    
    unix_input = input("Enter UNIX timestamp from the frame (or press Enter to skip): ").strip()
    if unix_input:
        try:
            return float(unix_input)
        except ValueError:
            logger.warning("Invalid timestamp format")
    
    return None

def run_cropping():
    """
    Crops the Utensil video into clips aligned with each motion JSON file.
    Accounts for recording start delay.
    """
    data_root = PROJECT_ROOT / "data" / "raw" / "Utensil"
    video_path = data_root / "Utensil_Retake_001.mp4"
    json_dir = data_root
    output_dir = data_root / "cropped"
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Step 1: Finding when recording actually starts...")
    logger.info("="*70)
    
    # Find recording start frame
    recording_frame, recording_start_sec = find_recording_start(str(video_path))
    logger.info(f"Recording starts at frame {recording_frame}, {recording_start_sec:.3f} seconds into video")
    
    # Get UNIX timestamp from that frame
    logger.info("\n" + "="*70)
    logger.info("Step 2: Getting UNIX timestamp from recording start frame...")
    logger.info("="*70)
    unix_ts = get_unix_timestamp_from_frame(str(video_path), recording_frame)
    
    if unix_ts is None:
        logger.warning("Could not get UNIX timestamp. Using motion data timestamp.")
        logger.warning("Cropping may not be perfectly aligned.")
        # Fallback to motion data timestamp
        first_json = sorted([f for f in os.listdir(json_dir) 
                           if f.endswith('.json') and not f.endswith('metadata.json') 
                           and not f.endswith('validation.json')])[0]
        with open(Path(json_dir) / first_json, 'r') as f:
            data = json.load(f)
            unix_ts = data['trajectories'][0]['timestamps'][0]
            logger.info(f"Using first motion timestamp: {unix_ts}")
    
    # Calculate video start timestamp
    video_start_ts = unix_ts - recording_start_sec
    logger.info(f"\nVideo start timestamp calculation:")
    logger.info(f"  UNIX timestamp at recording start: {unix_ts}")
    logger.info(f"  Recording starts at: {recording_start_sec:.3f}s into video")
    logger.info(f"  Video start timestamp: {video_start_ts}")
    
    logger.info("\n" + "="*70)
    logger.info("Step 3: Cropping video to motion files...")
    logger.info("="*70)
    logger.info(f"Video: {video_path}")
    logger.info(f"JSON Directory: {json_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Video Start Timestamp (Unix): {video_start_ts}")
    logger.info(f"Recording Start Offset: {recording_start_sec:.3f}s (will be skipped)")
    
    cropper = VideoCropper(str(video_path), str(output_dir))
    
    # Override the video start timestamp
    cropper.video_start_ts = video_start_ts
    
    # Also need to adjust for recording start delay
    # The motion timestamps are absolute, but video has pre-recording content
    # So we need to ensure we only crop from when recording actually started
    
    files = sorted([os.path.join(json_dir, f) for f in os.listdir(json_dir) 
                   if f.endswith('.json') and not f.endswith('metadata.json') 
                   and not f.endswith('validation.json')])
    
    for json_path in files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if 'trajectories' not in data or not data['trajectories']:
                continue
            
            traj = data['trajectories'][0]
            if not traj.get('timestamps'):
                continue
            
            motion_start_ts = traj['timestamps'][0]
            motion_end_ts = traj['timestamps'][-1]
            
            # Calculate offset from video start (accounting for recording delay)
            start_offset = motion_start_ts - video_start_ts
            end_offset = motion_end_ts - video_start_ts
            
            # Ensure we don't crop before recording actually started
            if start_offset < recording_start_sec:
                logger.warning(f"{os.path.basename(json_path)}: Motion starts before recording. Adjusting...")
                start_offset = recording_start_sec
            
            if end_offset < recording_start_sec:
                logger.warning(f"{os.path.basename(json_path)}: Motion ends before recording. Skipping...")
                continue
            
            if start_offset > cropper.duration_sec:
                logger.warning(f"{os.path.basename(json_path)}: Motion starts after video ends. Skipping...")
                continue
            
            end_offset = min(cropper.duration_sec, end_offset)
            
            if end_offset <= start_offset:
                logger.warning(f"{os.path.basename(json_path)}: Invalid duration. Skipping...")
                continue
            
            # Output filename
            json_name = Path(json_path).stem
            output_path = os.path.join(output_dir, f"{json_name}.mp4")
            
            logger.info(f"\nCropping {os.path.basename(json_path)}:")
            logger.info(f"  Motion time: {motion_start_ts:.3f} to {motion_end_ts:.3f}")
            logger.info(f"  Video time: {start_offset:.3f}s to {end_offset:.3f}s")
            
            # Use timestamp-based cropping
            cropper._write_clip_by_time(start_offset * 1000.0, end_offset * 1000.0, output_path)
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
    
    cropper.close()
    logger.info("\n" + "="*70)
    logger.info("Video cropping complete!")
    logger.info("="*70)

if __name__ == "__main__":
    run_cropping()

