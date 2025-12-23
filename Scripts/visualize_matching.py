import sys
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader
from src.motion_loader import MotionLoader
from src.motion_matcher import MotionMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Viz")

def visualize_matching():
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\MR_View.mp4"
    json_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002"
    
    if not os.path.exists(video_path) or not os.path.exists(json_path):
        logger.error("Files not found.")
        return

    # Load Data
    logger.info("Loading Video...")
    v_loader = VideoLoader(video_path)
    # v_timestamps are relative ms from 0
    v_timestamps = v_loader.extract_frame_timestamps()
    v_loader.close()
    
    logger.info("Loading Motion...")
    m_loader = MotionLoader(json_path) # Now supports directory
    
    # Initialize Matcher
    matcher = MotionMatcher(m_loader)
    
    # Strategy: Use File Modified Time - Duration as the Video Start Timestamp
    video_mtime = os.path.getmtime(video_path)
    
    # Calculate duration
    if v_timestamps:
        v_duration_ms = v_timestamps[-1] - v_timestamps[0]
        v_duration_sec = v_duration_ms / 1000.0
    else:
        v_duration_sec = 0.0
        
    estimated_start_ts = video_mtime - v_duration_sec
    logger.info(f"Video Modified Time: {video_mtime}")
    logger.info(f"Video Duration (sec): {v_duration_sec}")
    logger.info(f"Estimated Video Start: {estimated_start_ts}")
    # Fix potential crash if m_loader.timestamps is empty
    if m_loader.timestamps:
        logger.info(f"Motion Range: {m_loader.timestamps[0]} to {m_loader.timestamps[-1]}")
    else:
        logger.warning("No motion timestamps found.")

    # Convert video timestamps to absolute Unix Seconds for matching
    # aligned_ts = (v_ts_ms / 1000.0) + estimated_start_ts
    v_timestamps_sec = [t / 1000.0 for t in v_timestamps]
    
    # Match using absolute timestamps (offset=estimated_start_ts)
    matches = matcher.match_timestamps(v_timestamps_sec, offset_ms=estimated_start_ts)
    
    weights = [m['weight'] for m in matches]
    
    # Prepare plot
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Weights
    plt.subplot(2, 1, 1)
    plt.plot(weights, label='Interpolation Weight', alpha=0.7)
    plt.title('Interpolation Weights (1.0 = Matched, 0/1 Clamped = Gap)')
    plt.ylabel('Weight (0-1)')
    # plt.xlabel('Frame Index') 
    
    # Subplot 2: Temporal Alignment
    plt.subplot(2, 1, 2)
    # Plot normalized/centered timestamps to see drift
    v_ts_np = np.array([m['aligned_ts'] for m in matches])
    prev_ts_np = np.array([m['prev_motion_ts'] for m in matches])
    next_ts_np = np.array([m['next_motion_ts'] for m in matches])
    
    # Difference to Previous Motion
    diff = v_ts_np - prev_ts_np
    # Interval between motion poses (Next - Prev)
    interval = next_ts_np - prev_ts_np
    
    plt.plot(diff, label='Video Time - Prev Motion', color='green')
    plt.plot(interval, label='Motion Interval', color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'Temporal Alignment (Video Start: {estimated_start_ts:.2f})')
    plt.ylabel('Delta (Seconds)')
    plt.xlabel('Frame Index')
    plt.legend()
    
    plt.tight_layout()
    output_file = 'docs/matching_visualization_test002.png'
    plt.savefig(output_file)
    logger.info(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    visualize_matching()
