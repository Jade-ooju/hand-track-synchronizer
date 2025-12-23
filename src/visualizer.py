import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
import json
import os

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, width=1920, height=1080, fov_deg=100, config_path=None):
        """
        Initializes the Visualizer with configurable intrinsics and calibration.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fov_deg: Field of view in degrees
            config_path: Optional path to calibration config JSON
        """
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        
        # Manual calibration offsets
        self.offset_pos = np.array([0.0, 0.0, 0.0])  # XYZ position offset
        self.offset_rot_euler = np.array([0.0, 0.0, 0.0])  # Roll, Pitch, Yaw in degrees
        
        # Load from config if provided
        if config_path and os.path.exists(config_path):
            self.load_calibration(config_path)
        
        # Compute intrinsic matrix
        self._update_intrinsics()
        
        self.dist_coeffs = np.zeros(5)  # Assume no distortion
        logger.info(f"Visualizer initialized: FOV={self.fov_deg}, Offset Pos={self.offset_pos}, Offset Rot={self.offset_rot_euler}")
    
    def _update_intrinsics(self):
        """Recompute K matrix from current FOV."""
        f = (self.width / 2) / np.tan(np.radians(self.fov_deg / 2))
        self.K = np.array([
            [f, 0, self.width / 2],
            [0, f, self.height / 2],
            [0, 0, 1]
        ])
    
    def set_calibration(self, offset_pos=None, offset_rot_euler=None, fov_deg=None):
        """Update calibration parameters."""
        if offset_pos is not None:
            self.offset_pos = np.array(offset_pos)
        if offset_rot_euler is not None:
            self.offset_rot_euler = np.array(offset_rot_euler)
        if fov_deg is not None:
            self.fov_deg = fov_deg
            self._update_intrinsics()
        logger.debug(f"Calibration updated: Pos={self.offset_pos}, Rot={self.offset_rot_euler}, FOV={self.fov_deg}")
    
    def load_calibration(self, config_path):
        """Load calibration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.offset_pos = np.array(config.get('offset_pos', [0.0, 0.0, 0.0]))
            self.offset_rot_euler = np.array(config.get('offset_rot_euler', [0.0, 0.0, 0.0]))
            self.fov_deg = config.get('fov', self.fov_deg)
            self._update_intrinsics()
            logger.info(f"Loaded calibration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
    
    def save_calibration(self, config_path):
        """Save current calibration to JSON file."""
        config = {
            'offset_pos': self.offset_pos.tolist(),
            'offset_rot_euler': self.offset_rot_euler.tolist(),
            'fov': self.fov_deg
        }
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved calibration to {config_path}")
    
    def apply_offset(self, pose_world):
        """Apply manual calibration offset to a 3D pose.
        
        Args:
            pose_world: Dict with 'position' and 'rotation' keys
            
        Returns:
            Adjusted pose dict
        """
        # Apply position offset
        pos_adjusted = np.array(pose_world['position']) + self.offset_pos
        
        # Apply rotation offset
        rot_world = R.from_quat(pose_world['rotation'])
        rot_offset = R.from_euler('xyz', self.offset_rot_euler, degrees=True)
        rot_adjusted = rot_offset * rot_world
        
        return {
            'position': pos_adjusted.tolist(),
            'rotation': rot_adjusted.as_quat().tolist()
        }

    def project_point(self, point_world, camera_pose, check_bounds=True):
        """
        Projects a 3D world point to 2D image coordinates.
        
        Args:
            point_world (list/array): [x, y, z] in World Space.
            camera_pose (dict): {'position': [x,y,z], 'rotation': [x,y,z,w]} of the Camera.
            check_bounds: If True, return None if point is outside frame
            
        Returns:
            tuple: (u, v) pixel coordinates, or None if behind camera or out of bounds.
        """
        if point_world is None or camera_pose is None:
            return None
            
        P_world = np.array(point_world)
        if P_world.shape != (3,):
            P_world = P_world[:3]
            
        # Camera Extrinsics (World -> Camera)
        cam_pos = np.array(camera_pose['position'])
        cam_rot = R.from_quat(camera_pose['rotation'])
        
        # Vector from camera to point
        vec_cam_to_point = P_world - cam_pos
        
        # Rotate into camera frame
        P_cam = cam_rot.inv().apply(vec_cam_to_point)
        
        x, y, z = P_cam
        
        # Depth check
        if z <= 0.1: 
             return None
             
        # Flip Y for OpenCV convention (Screen Y is down)
        y = -y 
        
        # Projection
        uv = self.K @ np.array([x, y, z])
        uv = uv / uv[2]
        
        u, v = int(uv[0]), int(uv[1])
        
        # Bounds check
        if check_bounds:
            if not (0 <= u < self.width and 0 <= v < self.height):
                return None
        
        return u, v

    def draw_gizmo(self, img, pose_world, camera_pose, axis_length=0.1, apply_calibration=True):
        """
        Draws RGB axes at the pose location on the image.
        
        Args:
            img: Image to draw on
            pose_world: Pose dict with 'position' and 'rotation'
            camera_pose: Camera pose dict
            axis_length: Length of axes in meters
            apply_calibration: Whether to apply manual calibration offsets
        """
        # Apply calibration offset if requested
        if apply_calibration:
            pose_world = self.apply_offset(pose_world)
        
        origin = self.project_point(pose_world['position'], camera_pose)
        if origin is None:
            return
            
        # Calculate axis end points in World Space
        pos = np.array(pose_world['position'])
        rot = R.from_quat(pose_world['rotation'])
        
        # X (Red), Y (Green), Z (Blue)
        x_end = pos + rot.apply([axis_length, 0, 0])
        y_end = pos + rot.apply([0, axis_length, 0])
        z_end = pos + rot.apply([0, 0, axis_length])
        
        px_x = self.project_point(x_end, camera_pose)
        px_y = self.project_point(y_end, camera_pose)
        px_z = self.project_point(z_end, camera_pose)
        
        # Draw Lines
        cx, cy = origin
        thickness = 2
        
        if px_x: cv2.line(img, (cx, cy), px_x, (0, 0, 255), thickness)  # Red (X)
        if px_y: cv2.line(img, (cx, cy), px_y, (0, 255, 0), thickness)  # Green (Y)
        if px_z: cv2.line(img, (cx, cy), px_z, (255, 0, 0), thickness)  # Blue (Z)
        
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)  # Yellow center
    
    def draw_hand_point(self, img, pose_world, camera_pose, color, label="", apply_calibration=True, radius=8):
        """
        Draws a single hand point (simplified skeleton) with label.
        
        Args:
            img: Image to draw on
            pose_world: Pose dict
            camera_pose: Camera pose
            color: BGR color tuple
            label: Text label to display
            apply_calibration: Whether to apply offsets
            radius: Circle radius in pixels
        """
        # Apply calibration if requested
        if apply_calibration:
            pose_world = self.apply_offset(pose_world)
        
        point = self.project_point(pose_world['position'], camera_pose)
        if point is None:
            return
        
        x, y = point
        
        # Draw circle
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.circle(img, (x, y), radius + 2, (255, 255, 255), 2)  # White outline
        
        # Draw label
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2
            text_y = y - radius - 10
            
            # Background for text
            cv2.rectangle(img, 
                         (text_x - 3, text_y - text_size[1] - 3),
                         (text_x + text_size[0] + 3, text_y + 3),
                         (0, 0, 0), -1)
            cv2.putText(img, label, (text_x, text_y), font, font_scale, color, thickness)
    
    def draw_info_panel(self, img, frame_idx, total_frames, video_ts, raw_ts, synced_ts, temporal_offset, position_diff=None):
        """
        Draws information panel on the image.
        
        Args:
            img: Image to draw on
            frame_idx: Current frame index
            total_frames: Total frame count
            video_ts: Video timestamp
            raw_ts: Raw motion timestamp
            synced_ts: Synced motion timestamp  
            temporal_offset: Time difference between raw and video
            position_diff: Optional position difference in meters
        """
        panel_x = 10
        panel_y = 10
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 255, 0)  # Green
        
        # Draw semi-transparent background
        panel_height = line_height * (7 if position_diff else 6)
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x - 5, panel_y - 5), 
                     (panel_x + 400, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        y = panel_y + 20
        cv2.putText(img, f"Frame: {frame_idx} / {total_frames}", 
                   (panel_x, y), font, font_scale, color, thickness)
        y += line_height
        
        cv2.putText(img, f"Video Time: {video_ts:.3f}s", 
                   (panel_x, y), font, font_scale, color, thickness)
        y += line_height
        
        cv2.putText(img, f"Raw Motion Time: {raw_ts:.3f}s", 
                   (panel_x, y), font, font_scale, (128, 0, 128), thickness)  # Purple
        y += line_height
        
        cv2.putText(img, f"Synced Motion Time: {synced_ts:.3f}s", 
                   (panel_x, y), font, font_scale, (0, 255, 255), thickness)  # Yellow
        y += line_height
        
        cv2.putText(img, f"Temporal Offset: {temporal_offset*1000:.1f}ms", 
                   (panel_x, y), font, font_scale, color, thickness)
        y += line_height
        
        if position_diff is not None:
            cv2.putText(img, f"Position Diff: {position_diff*100:.2f}cm", 
                       (panel_x, y), font, font_scale, color, thickness)
            y += line_height
        
        # Legend
        legend_y = img.shape[0] - 60
        cv2.putText(img, "Purple = Raw (Nearest)", 
                   (panel_x, legend_y), font, font_scale, (128, 0, 128), thickness)
        cv2.putText(img, "Yellow = Synced (Interpolated)", 
                   (panel_x, legend_y + 30), font, font_scale, (0, 255, 255), thickness)
