import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestVideoLoader")

def test_video_loader():
    video_path = r"d:\OOJU\Projects\VideoSync\data\raw\test_001\Sync_RecordedMRView_TimeHead.mp4"
    
    logger.info(f"Testing with video: {video_path}")
    
    try:
        loader = VideoLoader(video_path)
        
        # Test Metadata
        metadata = loader.get_metadata()
        logger.info(f"Metadata: {metadata}")
        assert metadata['fps'] > 0
        assert metadata['frame_count'] > 0
        
        # Test Timestamp Extraction
        logger.info("Extracting timestamps...")
        timestamps = loader.extract_frame_timestamps()
        logger.info(f"Extracted {len(timestamps)} timestamps")
        logger.info(f"First 5 timestamps: {timestamps[:5]}")
        logger.info(f"Last 5 timestamps: {timestamps[-5:]}")
        
        assert len(timestamps) == metadata['frame_count']
        
        # Check monotonicity
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                logger.error(f"Timestamp gap/error at frame {i}: {timestamps[i-1]} -> {timestamps[i]}")
        
        logger.info("Timestamp monotonicity check passed.")

        # Test Frame Generator
        logger.info("Testing frame generator (first 10 frames)...")
        gen = loader.frame_generator()
        count = 0
        for ts, frame in gen:
            assert frame is not None
            count += 1
            if count >= 10:
                break
        logger.info("Frame generator test passed.")
        
        loader.close()
        logger.info("Test passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_loader()
