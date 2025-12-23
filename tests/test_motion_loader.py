import sys
import os
import logging
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motion_loader import MotionLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMotionLoader")

def test_motion_loader():
    json_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_001\Sync_TimeHead_20251217_231623.json"
    
    logger.info(f"Testing with motion file: {json_path}")
    
    try:
        loader = MotionLoader(json_path)
        
        # Test 1: Time Range
        min_t, max_t = loader.get_time_range()
        logger.info(f"Time Range: {min_t} to {max_t}")
        assert min_t is not None
        assert max_t is not None
        assert min_t <= max_t
        
        # Test 2: Get Pose at exact timestamp (min_t)
        pose_first = loader.get_pose_at_timestamp(min_t)
        assert pose_first is not None
        logger.info(f"First pose valid: {pose_first.keys()}")
        
        # Test 3: Get Surrounding Poses for a midpoint
        mid_t = (min_t + max_t) / 2
        prev_p, next_p = loader.get_surrounding_poses(mid_t)
        
        logger.info(f"Querying mid_t: {mid_t}")
        if prev_p:
            logger.info(f"Prev: {prev_p[0]}")
            assert prev_p[0] <= mid_t
        if next_p:
            logger.info(f"Next: {next_p[0]}")
            assert next_p[0] >= mid_t
            
        assert prev_p is not None or next_p is not None
        
        logger.info("Test passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_motion_loader()
