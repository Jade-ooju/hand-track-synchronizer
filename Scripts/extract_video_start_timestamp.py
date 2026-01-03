"""
Script to extract the video start timestamp from visible on-screen timestamps.
This finds the first frame where recording starts and extracts the UNIX timestamp.
"""
import sys
import os
import cv2
import numpy as np
import logging
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_recording_indicator(frame):
    """
    Detects if a frame shows the recording indicator (green circle).
    This is a simple color-based detection - you may need to adjust.
    
    Args:
        frame: BGR image frame
        
    Returns:
        bool: True if recording indicator is detected
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green color range (adjust these values based on your UI)
    # Green circle typically has high saturation and medium value
    lower_green = np.array([50, 100, 100])  # Adjust based on your green color
    upper_green = np.array([70, 255, 255])
    
    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Check if there are enough green pixels (indicating recording indicator)
    green_pixel_count = np.sum(mask > 0)
    
    # Threshold: if more than 100 green pixels, consider it recording
    return green_pixel_count > 100

def extract_timestamp_from_frame(frame, use_ocr=False):
    """
    Attempts to extract UNIX timestamp from frame.
    For now, this is a placeholder - you'll need to manually identify or use OCR.
    
    Args:
        frame: BGR image frame
        use_ocr: If True, attempts OCR (requires pytesseract)
        
    Returns:
        float or None: UNIX timestamp if found, None otherwise
    """
    if use_ocr:
        try:
            import pytesseract
            # Extract text from upper left region (where timestamp is displayed)
            height, width = frame.shape[:2]
            roi = frame[0:int(height*0.15), 0:int(width*0.3)]  # Upper left region
            
            # Convert to grayscale and enhance
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = pytesseract.image_to_string(thresh)
            
            # Look for UNIX: pattern
            match = re.search(r'UNIX:\s*(\d+\.?\d*)', text)
            if match:
                return float(match.group(1))
        except ImportError:
            logger.warning("pytesseract not available. Install with: pip install pytesseract")
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
    
    return None

def find_first_recording_frame(video_path, max_frames_to_check=1000):
    """
    Scans through video to find the first frame where recording starts.
    
    Args:
        video_path: Path to video file
        max_frames_to_check: Maximum number of frames to scan (to avoid scanning entire video)
        
    Returns:
        tuple: (frame_number, frame, timestamp_ms, unix_timestamp) or None
    """
    loader = VideoLoader(video_path)
    cap = loader.cap
    
    logger.info(f"Scanning video for first recording frame (max {max_frames_to_check} frames)...")
    
    frame_count = 0
    first_recording_frame = None
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while frame_count < max_frames_to_check:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Check if this frame has recording indicator
        if detect_recording_indicator(frame):
            logger.info(f"Found recording indicator at frame {frame_count}, timestamp {timestamp_ms:.1f}ms")
            first_recording_frame = (frame_count, frame, timestamp_ms)
            break
        
        frame_count += 1
    
    if first_recording_frame is None:
        logger.warning("Could not find recording indicator in first frames. You may need to adjust detection or scan more frames.")
        return None
    
    # Try to extract UNIX timestamp from the frame
    frame_num, frame, timestamp_ms = first_recording_frame
    unix_ts = extract_timestamp_from_frame(frame, use_ocr=False)
    
    return (frame_num, frame, timestamp_ms, unix_ts)

def interactive_timestamp_extraction(video_path):
    """
    Interactive mode: shows frames and lets user identify the first recording frame.
    """
    loader = VideoLoader(video_path)
    cap = loader.cap
    
    logger.info("Interactive mode: Use arrow keys to navigate, 's' to select frame, 'q' to quit")
    logger.info("Look for the frame where recording starts (green circle appears)")
    
    current_frame = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Draw frame number and timestamp on frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {current_frame}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Time: {timestamp_ms:.1f}ms", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Arrow keys: navigate, 's': select, 'q': quit", (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Find First Recording Frame", display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            logger.info(f"Selected frame {current_frame} at {timestamp_ms:.1f}ms")
            logger.info("Please manually read the UNIX timestamp from the frame and enter it below.")
            unix_input = input("Enter UNIX timestamp from the frame (or press Enter to skip): ").strip()
            if unix_input:
                try:
                    unix_ts = float(unix_input)
                    # Calculate video start: unix_ts - (timestamp_ms / 1000.0)
                    video_start = unix_ts - (timestamp_ms / 1000.0)
                    logger.info(f"\n{'='*60}")
                    logger.info(f"VIDEO START TIMESTAMP: {video_start}")
                    logger.info(f"{'='*60}")
                    logger.info(f"Frame {current_frame}: UNIX={unix_ts}, Video time={timestamp_ms/1000:.3f}s")
                    logger.info(f"Video start = UNIX - Video time = {unix_ts} - {timestamp_ms/1000:.3f} = {video_start}")
                    return video_start
                except ValueError:
                    logger.warning("Invalid timestamp format")
        elif key == 81 or key == 2:  # Left arrow
            current_frame = max(0, current_frame - 10)
        elif key == 83 or key == 3:  # Right arrow
            current_frame += 10
        elif key == 82 or key == 0:  # Up arrow
            current_frame = max(0, current_frame - 1)
        elif key == 84 or key == 1:  # Down arrow
            current_frame += 1
    
    cv2.destroyAllWindows()
    loader.close()
    return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract video start timestamp from on-screen timestamps')
    parser.add_argument('--video', default='data/raw/Utensil/Utensil_Retake_001.mp4',
                       help='Path to video file')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive mode to manually identify first recording frame')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically detect first recording frame (may need adjustment)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    if args.interactive:
        result = interactive_timestamp_extraction(args.video)
        if result:
            logger.info(f"\nUse this timestamp in your config: {result}")
    elif args.auto:
        result = find_first_recording_frame(args.video)
        if result:
            frame_num, frame, timestamp_ms, unix_ts = result
            logger.info(f"\nFound first recording frame:")
            logger.info(f"  Frame: {frame_num}")
            logger.info(f"  Video time: {timestamp_ms:.1f}ms ({timestamp_ms/1000:.3f}s)")
            if unix_ts:
                video_start = unix_ts - (timestamp_ms / 1000.0)
                logger.info(f"  UNIX timestamp: {unix_ts}")
                logger.info(f"  VIDEO START: {video_start}")
            else:
                logger.info("  Could not extract UNIX timestamp automatically.")
                logger.info("  Please use --interactive mode to manually enter it.")
    else:
        logger.info("Please specify --interactive or --auto mode")
        logger.info("Recommended: --interactive for accurate timestamp extraction")

if __name__ == "__main__":
    main()

