import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, width=1920, height=1080, fov_deg=100):
        """
        Initializes the Visualizer with estimated intrinsics.
        MR/Quest 3 recording is typically 1920x1080 (single eye view crop).
        FOV varies, but ~100 vertical/horizontal is expected.
        """
        self.width = width
        self.height = height
        
        # Estimate Focal Length from FOV
        # f = (h / 2) / tan(fov / 2)
        # Using horizontal FOV assumption
        f = (width / 2) / np.tan(np.radians(fov_deg / 2))
        
        self.K = np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1]
        ])
        
        self.dist_coeffs = np.zeros(5) # Assume no distortion for generic overlay
        logger.info(f"Visualizer Intrinsic K:\n{self.K}")

    def project_point(self, point_world, camera_pose):
        """
        Projects a 3D world point to 2D image coordinates.
        
        Args:
            point_world (list/array): [x, y, z] in World Space.
            camera_pose (dict): {'position': [x,y,z], 'rotation': [x,y,z,w]} of the Camera.
            
        Returns:
            tuple: (u, v) pixel coordinates, or None if behind camera.
        """
        if point_world is None or camera_pose is None:
            return None
            
        P_world = np.array(point_world)
        if P_world.shape != (3,):
            P_world = P_world[:3]
            
        # Camera Extrinsics (World -> Camera)
        # T_cam = [R|t]
        # P_cam = Inv(T_cam) * P_world
        #       = R_inv * (P_world - t)
        
        cam_pos = np.array(camera_pose['position'])
        cam_rot = R.from_quat(camera_pose['rotation'])
        
        # Vector from camera to point
        vec_cam_to_point = P_world - cam_pos
        
        # Rotate into camera frame (Inverse of Camera Rotation)
        # scipy Rotation.inv()
        P_cam = cam_rot.inv().apply(vec_cam_to_point)
        
        # Check Z > 0 (in front of camera)
        # Convention: Camera looks down -Z (OpenGL) or +Z (OpenCV)?
        # Quest typically uses -Z forward. OpenCV uses +Z forward.
        # We need to flip Z if the coordinate systems mismatch.
        # Unity (Quest): Y up, Z forward (Left Handed?) NO, Unity is Y-Up, Z-Forward (Left Handed).
        # JSON Data: Likely Unity World Coordinates.
        # OpenCV Camera: Y down, Z forward.
        # Transformation: Unity World -> Unity Camera -> OpenCV Camera -> Image
        
        # Usually: 
        # Unity Camera Frame: +X Right, +Y Up, +Z Forward
        # OpenCV Camera Frame: +X Right, -Y Up (Down), +Z Forward
        # So we negate Y.
        
        # Let's try standard projection first.
        x, y, z = P_cam
        
        # If Z is negative, it's behind the camera (assuming +Z forward convention for projection)
        # If Unity calculates P_cam with +Z forward, we are good.
        if z <= 0.1: 
             return None
             
        # Standard Pinhole Projection
        # u = fx * (x/z) + cx
        # v = fy * (y/z) + cy
        # Note: If Y is Up in P_cam, and Down in Image, we negate Y term or P_cam.y
        
        # Flip Y for OpenCV convention (Screen Y is down)
        y = -y 
        
        uv = self.K @ np.array([x, y, z])
        uv = uv / uv[2]
        
        return int(uv[0]), int(uv[1])

    def draw_gizmo(self, img, pose_world, camera_pose, axis_length=0.1):
        """
        Draws RGB axes at the pose location on the image.
        """
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
        
        if px_x: cv2.line(img, (cx, cy), px_x, (0, 0, 255), thickness) # R (BGR format -> Red is last) -> Wait, OpenCV is BGR. (0,0,255) is RED.
        if px_y: cv2.line(img, (cx, cy), px_y, (0, 255, 0), thickness) # G
        if px_z: cv2.line(img, (cx, cy), px_z, (255, 0, 0), thickness) # B
        
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1) # Yellow center
