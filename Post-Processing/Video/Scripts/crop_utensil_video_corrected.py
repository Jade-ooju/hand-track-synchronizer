"""
Video cropping that accounts for recording start delay.
Accepts UNIX timestamp from first recording frame as argument.
"""
import sys
import os
import cv2
import numpy as np
import logging
import json
import argparse
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

def run_cropping(unix_timestamp=None):
    """
    Crops the Utensil video into clips aligned with each motion JSON file.
    Accounts for recording start delay.
    
    Args:
        unix_timestamp: UNIX timestamp from the first recording frame.
                       If None, will use motion data timestamp (less accurate).
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
    
    # Get or calculate UNIX timestamp
    logger.info("\n" + "="*70)
    logger.info("Step 2: Calculating video start timestamp...")
    logger.info("="*70)
    
    if unix_timestamp is None:
        logger.warning("No UNIX timestamp provided. Using first motion timestamp as fallback.")
        logger.warning("For accurate cropping, provide UNIX timestamp from first recording frame.")
        # Fallback to motion data timestamp
        first_json = sorted([f for f in os.listdir(json_dir) 
                           if f.endswith('.json') and not f.endswith('metadata.json') 
                           and not f.endswith('validation.json')])[0]
        with open(Path(json_dir) / first_json, 'r') as f:
            data = json.load(f)
            unix_ts = data['trajectories'][0]['timestamps'][0]
            logger.info(f"Using first motion timestamp: {unix_ts}")
            # Adjust: if motion starts before recording, we need to account for that
            video_start_ts = unix_ts - recording_start_sec
    else:
        unix_ts = unix_timestamp
        # Calculate video start: UNIX at recording start - recording start time
        video_start_ts = unix_ts - recording_start_sec
        logger.info(f"Using provided UNIX timestamp: {unix_ts}")
    
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
    logger.info(f"Recording Start Offset: {recording_start_sec:.3f}s (pre-recording content will be skipped)")
    
    cropper = VideoCropper(str(video_path), str(output_dir))
    
    # Override the video start timestamp
    cropper.video_start_ts = video_start_ts
    
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
            
            # Calculate offset from video start
            start_offset = motion_start_ts - video_start_ts
            end_offset = motion_end_ts - video_start_ts
            
            # Ensure we don't crop before recording actually started
            if start_offset < recording_start_sec:
                logger.warning(f"{os.path.basename(json_path)}: Motion starts before recording ({start_offset:.3f}s < {recording_start_sec:.3f}s). Adjusting to recording start.")
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
            logger.info(f"  Duration: {end_offset - start_offset:.3f}s")
            
            # Use timestamp-based cropping
            cropper._write_clip_by_time(start_offset * 1000.0, end_offset * 1000.0, output_path)
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
    
    cropper.close()
    logger.info("\n" + "="*70)
    logger.info("Video cropping complete!")
    logger.info("="*70)
    logger.info(f"All clips start from when recording actually began (after {recording_start_sec:.3f}s offset)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop Utensil video with recording start correction')
    parser.add_argument('--unix-timestamp', type=float,
                       help='UNIX timestamp from the first recording frame (read from test_frame_00510.png)')
    
    args = parser.parse_args()
    
    run_cropping(unix_timestamp=args.unix_timestamp)

