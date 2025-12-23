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
logger = logging.getLogger("Pipe")

def run_visualization():
    # PATHS
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\MR_View.mp4"
    json_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002"
    output_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\viz_output.mp4"
    
    # 1. Load Data
    logger.info("Loading Video...")
    v_loader = VideoLoader(video_path)
    timestamps_ms = v_loader.extract_frame_timestamps()
    # Convert to seconds
    v_timestamps_sec = [t / 1000.0 for t in timestamps_ms]
    
    logger.info("Loading Motion...")
    m_loader = MotionLoader(json_dir)
    
    # 2. Match Timestamps
    # Manual Calibration from previous step
    calibrated_start = 1766488163.738
    
    logger.info(f"Matching with Start Time: {calibrated_start}")
    matcher = MotionMatcher(m_loader)
    matches = matcher.match_timestamps(v_timestamps_sec, offset_ms=calibrated_start) 
    # Note: match_timestamps expects offset in SAME UNIT as timestamps? 
    # Logic in MotionMatcher: `aligned_ts = timestamp + offset_ms` (if offset is seconds, var name is misleading but math works)
    # The `visualize_matching.py` passed `offset_ms` as the start unix timestamp (seconds).
    # So `match_timestamps` adds it.
    
    # 3. Setup Components
    interpolator = Interpolator()
    visualizer = Visualizer(width=int(v_loader.width), height=int(v_loader.height))
    
    # 4. Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, v_loader.fps, (int(v_loader.width), int(v_loader.height)))
    
    # 5. Loop Frames
    v_loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    logger.info("Rendering frames...")
    
    frame_idx = 0
    pbar = tqdm(total=len(matches))
    
    while True:
        ret, frame = v_loader.cap.read()
        if not ret:
            break
            
        if frame_idx >= len(matches):
            break
            
        match = matches[frame_idx]
        w = match['weight']
        
        # Only draw if we have valid interpolation (weight not 0/1 pinned to ends, or strictly inside range)
        # Actually MotionMatcher returns 0/1 if out of bounds.
        # We should check if we are significantly inside a valid range?
        # Let's verify via MotionLoader.get_surrounding_poses
        
        # We need generic interpolation!
        # MotionLoader update: get_surrounding returns "pose", "left_eye", "right_eye" dicts? Not exactly.
        # MotionLoader.get_surrounding_poses returns ((t1, data1), (t2, data2))
        # where data1 is self.poses[i] (legacy) OR updated structure?
        
        # CHECK MotionLoader changes:
        # I updated `get_surrounding_poses` to return:
        # curr = (ts, {'pose':..., 'left_eye':..., 'right_eye':...})
        
        # So we can fetch directly using timestamps from match result?
        # NO, 'match' result has prev_motion_ts/next_motion_ts
        # But we don't have the INDICES.
        # We should probably use `MotionLoader.get_surrounding_poses(match['aligned_ts'])` directly here?
        # Re-using Matcher result is good for analysis but accessing data might be repeating work.
        # Let's just use `m_loader.get_surrounding_poses(match['aligned_ts'])`.
        
        prev_data, next_data = m_loader.get_surrounding_poses(match['aligned_ts'])
        
        # GAP CHECK: If the interval is too large, we are likely between sessions.
        # Standard Quest hand tracking is 60Hz (~16ms).
        # Tolerance: 100-200ms.
        GAP_THRESHOLD = 0.2
        
        if prev_data and next_data:
            t1, d1 = prev_data
            t2, d2 = next_data
            
            if (t2 - t1) > GAP_THRESHOLD:
                # Gap detected, skip drawing
                out.write(frame)
                pbar.update(1)
                frame_idx += 1
                continue
            
            # Interpolate Hand
            
            # Hand Pose
            hand_pose = interpolator.interpolate_pose(d1['pose'], d2['pose'], w)
            
            # Camera interpolation (Eyes)
            # Create synthetic "Head" pose = Average of Left and Right Eyes?
            # Or just use Left Eye for now? Quest captures usually view from distinct eyes.
            # Let's try Centroid of Left/Right.
            
            # Interpolate Left Eye
            le_pose = None
            if d1['left_eye'] and d2['left_eye']:
                 le_pose = interpolator.interpolate_pose(d1['left_eye'], d2['left_eye'], w)
                 
            # Interpolate Right Eye
            re_pose = None
            if d1['right_eye'] and d2['right_eye']:
                 re_pose = interpolator.interpolate_pose(d1['right_eye'], d2['right_eye'], w)
                 
            cam_pose = None
            if le_pose and re_pose:
                # Average Position
                pos = (np.array(le_pose['position']) + np.array(re_pose['position'])) / 2.0
                # Rotation? Use Left Eye Rotation? Or Slerp(Left, Right, 0.5)?
                # Usually left/right eyes have same rotation (head rotation).
                cam_pose = {'position': pos.tolist(), 'rotation': le_pose['rotation']}
            elif le_pose:
                cam_pose = le_pose
            elif re_pose:
                cam_pose = re_pose
                
            # Draw
            if cam_pose:
                visualizer.draw_gizmo(frame, hand_pose, cam_pose)
                
                # Debug Text
                cv2.putText(frame, f"TS: {match['aligned_ts']:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
        pbar.update(1)
        frame_idx += 1
        
    pbar.close()
    out.release()
    v_loader.close()
    logger.info(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    run_visualization()
