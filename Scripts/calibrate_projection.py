import sys
import os
import cv2
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader
from src.motion_loader import MotionLoader
from src.motion_matcher import MotionMatcher
from src.interpolator import Interpolator
from src.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Calibration")

def calibrate_projection():
    """
    Interactive calibration tool for manual projection adjustment.
    
    Controls:
        W/S: Move forward/backward (Z offset)
        A/D: Move left/right (X offset)
        Q/E: Move up/down (Y offset)
        I/K: Pitch rotation
        J/L: Yaw rotation
        U/O: Roll rotation
        +/-: Increase/decrease FOV
        SPACE: Save calibration
        ESC: Exit
    """
    # PATHS
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_002\MR_View.mp4"
    json_dir = r"d:\OOJU\Projects\VideoSync\data\raw\test_002"
    config_path = r"d:\OOJU\Projects\VideoSync\config\calibration.json"
    
    # Load Data
    logger.info("Loading Video...")
    v_loader = VideoLoader(video_path)
    timestamps_ms = v_loader.extract_frame_timestamps()
    v_timestamps_sec = [t / 1000.0 for t in timestamps_ms]
    
    logger.info("Loading Motion...")
    m_loader = MotionLoader(json_dir)
    
    # Match Timestamps
    calibrated_start = 1766488163.738
    matcher = MotionMatcher(m_loader)
    matches = matcher.match_timestamps(v_timestamps_sec, offset_ms=calibrated_start)
    
    # Initialize Visualizer with config
    visualizer = Visualizer(
        width=int(v_loader.width), 
        height=int(v_loader.height),
        config_path=config_path if os.path.exists(config_path) else None
    )
    
    interpolator = Interpolator()
    
    # Calibration parameters
    step_pos = 0.01  # 1cm per keypress
    step_rot = 1.0   # 1 degree per keypress
    step_fov = 1.0   # 1 degree per keypress
    
    # Frame selection (middle frame for calibration)
    frame_idx = len(matches) // 2
    
    logger.info(f"\nCalibration Controls:")
    logger.info("  W/S: Z offset ±{:.3f}m".format(step_pos))
    logger.info("  A/D: X offset ±{:.3f}m".format(step_pos))
    logger.info("  Q/E: Y offset ±{:.3f}m".format(step_pos))
    logger.info("  I/K: Pitch ±{}°".format(step_rot))
    logger.info("  J/L: Yaw ±{}°".format(step_rot))
    logger.info("  U/O: Roll ±{}°".format(step_rot))
    logger.info("  +/-: FOV ±{}°".format(step_fov))
    logger.info("  SPACE: Save calibration")
    logger.info("  R: Reset to defaults")
    logger.info("  ESC: Exit\n")
    
    paused = True
    GAP_THRESHOLD = 0.2
    
    while True:
        # Get frame
        v_loader.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = v_loader.cap.read()
        if not ret:
            break
        
        # Get match data
        match = matches[frame_idx]
        prev_data, next_data = m_loader.get_surrounding_poses(match['aligned_ts'])
        
        # Interpolate and draw if valid
        if prev_data and next_data:
            t1, d1 = prev_data
            t2, d2 = next_data
            
            if (t2 - t1) <= GAP_THRESHOLD:
                w = match['weight']
                
                # Interpolate Hand
                hand_pose = interpolator.interpolate_pose(d1['pose'], d2['pose'], w)
                
                # Camera interpolation
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
                    
                # Draw with calibration applied
                if cam_pose:
                    visualizer.draw_gizmo(frame, hand_pose, cam_pose, apply_calibration=True)
        
        # Display calibration info
        info_y = 30
        cv2.putText(frame, f"Frame: {frame_idx}/{len(matches)}", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        cv2.putText(frame, f"Offset XYZ: [{visualizer.offset_pos[0]:.3f}, {visualizer.offset_pos[1]:.3f}, {visualizer.offset_pos[2]:.3f}]", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        cv2.putText(frame, f"Offset RPY: [{visualizer.offset_rot_euler[0]:.1f}, {visualizer.offset_rot_euler[1]:.1f}, {visualizer.offset_rot_euler[2]:.1f}]", 
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        cv2.putText(frame, f"FOV: {visualizer.fov_deg:.1f}", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        cv2.putText(frame, "[SPACE] Save | [R] Reset | [ESC] Exit", (10, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Calibration", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('w'):
            visualizer.set_calibration(offset_pos=visualizer.offset_pos + [0, 0, step_pos])
        elif key == ord('s'):
            visualizer.set_calibration(offset_pos=visualizer.offset_pos + [0, 0, -step_pos])
        elif key == ord('a'):
            visualizer.set_calibration(offset_pos=visualizer.offset_pos + [-step_pos, 0, 0])
        elif key == ord('d'):
            visualizer.set_calibration(offset_pos=visualizer.offset_pos + [step_pos, 0, 0])
        elif key == ord('q'):
            visualizer.set_calibration(offset_pos=visualizer.offset_pos + [0, -step_pos, 0])
        elif key == ord('e'):
            visualizer.set_calibration(offset_pos=visualizer.offset_pos + [0, step_pos, 0])
        elif key == ord('i'):
            visualizer.set_calibration(offset_rot_euler=visualizer.offset_rot_euler + [step_rot, 0, 0])
        elif key == ord('k'):
            visualizer.set_calibration(offset_rot_euler=visualizer.offset_rot_euler + [-step_rot, 0, 0])
        elif key == ord('j'):
            visualizer.set_calibration(offset_rot_euler=visualizer.offset_rot_euler + [0, step_rot, 0])
        elif key == ord('l'):
            visualizer.set_calibration(offset_rot_euler=visualizer.offset_rot_euler + [0, -step_rot, 0])
        elif key == ord('u'):
            visualizer.set_calibration(offset_rot_euler=visualizer.offset_rot_euler + [0, 0, step_rot])
        elif key == ord('o'):
            visualizer.set_calibration(offset_rot_euler=visualizer.offset_rot_euler + [0, 0, -step_rot])
        elif key == ord('+') or key == ord('='):
            visualizer.set_calibration(fov_deg=visualizer.fov_deg + step_fov)
        elif key == ord('-') or key == ord('_'):
            visualizer.set_calibration(fov_deg=visualizer.fov_deg - step_fov)
        elif key == ord(' '):
            visualizer.save_calibration(config_path)
            logger.info("✓ Calibration saved!")
        elif key == ord('r'):
            visualizer.set_calibration(offset_pos=[0, 0, 0], offset_rot_euler=[0, 0, 0], fov_deg=100)
            logger.info("Reset to defaults")
        elif key == ord('.'):
            frame_idx = min(frame_idx + 10, len(matches) - 1)
        elif key == ord(','):
            frame_idx = max(frame_idx - 10, 0)
    
    cv2.destroyAllWindows()
    v_loader.close()

if __name__ == "__main__":
    calibrate_projection()
