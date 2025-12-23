import sys
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader
from src.motion_loader import MotionLoader
from src.motion_matcher import MotionMatcher
from src.interpolator import Interpolator
from src.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Comparison")

def compare_raw_synced():
    """
    Generate comparison visualization showing raw vs synced hand poses.
    
    Raw (Purple): Nearest pose - no interpolation
    Synced (Yellow): Interpolated pose - smooth animation
    """
    # PATHS
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\MR_View.mp4"
    json_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002"
    config_path = r"d:\OOJU\Projects\VideoSync\config\calibration.json"
    output_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\comparison_output.mp4"
    
    # Load Data
    logger.info("Loading Video...")
    v_loader = VideoLoader(video_path)
    timestamps_ms = v_loader.extract_frame_timestamps()
    v_timestamps_sec = [t / 1000.0 for t in timestamps_ms]
    
    logger.info("Loading Motion...")
    m_loader = MotionLoader(json_dir)
    
    # Match Timestamps
    calibrated_start = 1766488163.738
    logger.info(f"Matching with Start Time: {calibrated_start}")
    matcher = MotionMatcher(m_loader)
    matches = matcher.match_timestamps(v_timestamps_sec, offset_ms=calibrated_start)
    
    # Initialize Components
    interpolator = Interpolator()
    visualizer = Visualizer(
        width=int(v_loader.width), 
        height=int(v_loader.height),
        config_path=config_path if os.path.exists(config_path) else None
    )
    
    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, v_loader.fps, 
                          (int(v_loader.width), int(v_loader.height)))
    
    # Process Frames
    v_loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    logger.info("Rendering comparison frames...")
    
    frame_idx = 0
    pbar = tqdm(total=len(matches))
    GAP_THRESHOLD = 0.2
    
    while True:
        ret, frame = v_loader.cap.read()
        if not ret or frame_idx >= len(matches):
            break
        
        match = matches[frame_idx]
        prev_data, next_data = m_loader.get_surrounding_poses(match['aligned_ts'])
        
        # Only draw if within recording (no gaps)
        if prev_data and next_data:
            t1, d1 = prev_data
            t2, d2 = next_data
            
            if (t2 - t1) <= GAP_THRESHOLD:
                w = match['weight']
                
                # Get RAW pose (nearest)
                # If weight < 0.5, use prev, else use next
                raw_pose = d1['pose'] if w < 0.5 else d2['pose']
                raw_ts = t1 if w < 0.5 else t2
                
                # Get SYNCED pose (interpolated)
                synced_pose = interpolator.interpolate_pose(d1['pose'], d2['pose'], w)
                synced_ts = match['aligned_ts']
                
                # Get camera pose
                le_pose = None
                if d1['left_eye'] and d2['left_eye']:
                    le_pose = interpolator.interpolate_pose(d1['left_eye'], d2['left_eye'], w)
                    
                re_pose = None
                if d1['right_eye'] and d2['right_eye']:
                    re_pose = interpolator.interpolate_pose(d1['right_eye'], d2['right_eye'], w)
                    
                cam_pose = None
                if le_pose and re_pose:
                    pos = (np.array(le_pose['position']) + np.array(re_pose['position'])) / 2.0
                    cam_pose = {'position': pos.tolist(), 'rotation': le_pose['rotation']}
                elif le_pose:
                    cam_pose = le_pose
                elif re_pose:
                    cam_pose = re_pose
                
                # Draw both poses
                if cam_pose:
                    # Purple = Raw
                    visualizer.draw_hand_point(frame, raw_pose, cam_pose, 
                                              color=(128, 0, 128), label="RAW", 
                                              apply_calibration=True, radius=10)
                    
                    # Yellow = Synced
                    visualizer.draw_hand_point(frame, synced_pose, cam_pose, 
                                              color=(0, 255, 255), label="SYNCED", 
                                              apply_calibration=True, radius=10)
                    
                    # Calculate metrics
                    temporal_offset = abs(raw_ts - synced_ts)
                    position_diff = np.linalg.norm(
                        np.array(raw_pose['position']) - np.array(synced_pose['position'])
                    )
                    
                    # Draw info panel
                    visualizer.draw_info_panel(
                        frame, frame_idx, len(matches),
                        synced_ts, raw_ts, synced_ts, temporal_offset, position_diff
                    )
        
        out.write(frame)
        pbar.update(1)
        frame_idx += 1
    
    pbar.close()
    out.release()
    v_loader.close()
    logger.info(f"Comparison video saved to {output_path}")

if __name__ == "__main__":
    compare_raw_synced()
