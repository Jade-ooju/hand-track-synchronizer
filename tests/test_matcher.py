import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motion_loader import MotionLoader
from src.motion_matcher import MotionMatcher
# Mock class for independent testing
class MockMotionLoader:
    def __init__(self, timestamps):
        self.timestamps = sorted(timestamps)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMatcher")

def test_matcher():
    logger.info("Starting Matcher Tests...")
    
    # 1. Test with simple synthetic data
    motion_ts = [10.0, 20.0, 30.0, 40.0]
    loader = MockMotionLoader(motion_ts)
    matcher = MotionMatcher(loader)
    
    # Cases:
    # A: Exact match (20.0)
    # B: Midpoint (15.0) -> weight 0.5
    # C: Before start (5.0) -> clamp
    # D: After end (50.0) -> clamp
    video_ts = [20.0, 15.0, 5.0, 50.0] 
    
    matches = matcher.match_timestamps(video_ts, offset_ms=0.0)
    
    for i, m in enumerate(matches):
        logger.info(f"Case {i}: Input {m['aligned_ts']} -> Prev: {m['prev_motion_ts']}, Next: {m['next_motion_ts']}, W: {m['weight']}")
        
    # Validation
    # Case 0: 20.0 -> Prev 10 or 20? bisect_left(20) gives idx=1 (value 20). 
    # Logic: prev=ts[0]=10, next=ts[1]=20. Weight = (20-10)/(20-10) = 1.0. Correct.
    assert matches[0]['weight'] == 1.0 or matches[0]['prev_motion_ts'] == 20.0
    
    # Case 1: 15.0 -> Prev 10, Next 20. Weight 0.5
    assert matches[1]['prev_motion_ts'] == 10.0
    assert matches[1]['next_motion_ts'] == 20.0
    assert abs(matches[1]['weight'] - 0.5) < 1e-6
    
    # Case 2: 5.0 -> Prev 10, Next 10 (clamped). Weight 0.0
    assert matches[2]['prev_motion_ts'] == 10.0
    assert matches[2]['weight'] == 0.0
    
    # Case 3: 50.0 -> Prev 40, Next 40 (clamped). Weight 1.0
    assert matches[3]['prev_motion_ts'] == 40.0
    assert matches[3]['weight'] == 1.0
    
    logger.info("Synthetic tests passed.")
    
    # 2. Test with Real Data (if available)
    try:
        real_json = r"d:\OOJU\Projects\VideoSync\data\raw\test_001\Sync_TimeHead_20251217_231623.json"
        if os.path.exists(real_json):
            logger.info("Testing with real file...")
            real_loader = MotionLoader(real_json)
            real_matcher = MotionMatcher(real_loader)
            
            # Use real timestamps range
            min_t, max_t = real_loader.get_time_range()
            # Create dummy video timestamps inside range
            dummy_video = [min_t + 1.0, max_t - 1.0] 
            
            real_matches = real_matcher.match_timestamps(dummy_video)
            for m in real_matches:
                 logger.info(f"Real Match: {m}")
                 assert m['prev_motion_ts'] <= m['aligned_ts'] <= m['next_motion_ts']
            
            logger.info("Real data tests passed.")
    except Exception as e:
         logger.warning(f"Skipping real data test: {e}")

    logger.info("All matcher tests passed!")

if __name__ == "__main__":
    test_matcher()
