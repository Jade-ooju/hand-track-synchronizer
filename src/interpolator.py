import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import logging

logger = logging.getLogger(__name__)

class Interpolator:
    def __init__(self):
        pass

    def interpolate_pose(self, prev_pose, next_pose, weight):
        """
        Interpolates between two poses.
        
        Args:
            prev_pose (dict): Start pose {'position': [x,y,z], 'rotation': [x,y,z,w], 'gripper': float}
            next_pose (dict): End pose {'position': [x,y,z], 'rotation': [x,y,z,w], 'gripper': float}
            weight (float): Interpolation weight (0.0 to 1.0).
            
        Returns:
            dict: Interpolated pose.
        """
        # Clamp weight
        weight = max(0.0, min(1.0, weight))
        
        # 1. Position Interpolation (Lerp)
        p1 = np.array(prev_pose['position'])
        p2 = np.array(next_pose['position'])
        p_interp = p1 + (p2 - p1) * weight
        
        # 2. Rotation Interpolation (Slerp)
        # scipy expects [x, y, z, w]
        # Create Rotation object containing both keyframes
        key_rots = R.from_quat([prev_pose['rotation'], next_pose['rotation']])
        key_times = [0, 1]
        
        # Initialize Slerp
        slerp = Slerp(key_times, key_rots)
        
        # Interpolate
        r_interp = slerp([weight])
        
        # 3. Gripper Interpolation (Lerp)
        g1 = prev_pose.get('gripper', 0.0)
        g2 = next_pose.get('gripper', 0.0)
        g_interp = g1 + (g2 - g1) * weight
        
        return {
            'position': p_interp.tolist(),
            'rotation': r_interp.as_quat()[0].tolist(),
            'gripper': float(g_interp)
        }
