import sys
import os
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.interpolator import Interpolator

class TestInterpolator(unittest.TestCase):
    def setUp(self):
        self.interpolator = Interpolator()

    def test_position_lerp_midpoint(self):
        p1 = {'position': [0, 0, 0], 'rotation': [0, 0, 0, 1], 'gripper': 0.0}
        p2 = {'position': [10, 20, 30], 'rotation': [0, 0, 0, 1], 'gripper': 1.0}
        
        res = self.interpolator.interpolate_pose(p1, p2, 0.5)
        
        np.testing.assert_array_almost_equal(res['position'], [5, 10, 15])
        self.assertAlmostEqual(res['gripper'], 0.5)

    def test_rotation_slerp_90_degrees(self):
        # Rotate 90 deg around Z
        # q1: Identity
        # q2: 90 deg Z
        r1 = R.from_euler('z', 0, degrees=True)
        r2 = R.from_euler('z', 90, degrees=True)
        
        p1 = {'position': [0,0,0], 'rotation': r1.as_quat().tolist(), 'gripper': 0}
        p2 = {'position': [0,0,0], 'rotation': r2.as_quat().tolist(), 'gripper': 0}
        
        # 0.5 weight -> 45 degrees
        res = self.interpolator.interpolate_pose(p1, p2, 0.5)
        
        r_res = R.from_quat(res['rotation'])
        angles = r_res.as_euler('xyz', degrees=True)
        # Z is the 3rd component (index 2)
        
        self.assertAlmostEqual(angles[2], 45.0)

    def test_clamping(self):
        p1 = {'position': [0,0,0], 'rotation': [0,0,0,1]}
        p2 = {'position': [10,0,0], 'rotation': [0,0,0,1]}
        
        res_neg = self.interpolator.interpolate_pose(p1, p2, -0.5)
        np.testing.assert_array_equal(res_neg['position'], [0,0,0]) # Clamped to 0
        
        res_over = self.interpolator.interpolate_pose(p1, p2, 1.5)
        np.testing.assert_array_equal(res_over['position'], [10,0,0]) # Clamped to 1

if __name__ == '__main__':
    unittest.main()
