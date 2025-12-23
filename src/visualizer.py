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
