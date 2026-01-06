"""
Saves the first recording frame so you can visually read the UNIX timestamp.
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.video_loader import VideoLoader

def detect_recording_indicator(frame):
    """Detects green recording indicator."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = np.sum(mask > 0)
    return green_pixel_count > 100

def find_and_save_first_recording_frame(video_path, output_path="first_recording_frame.png"):
    """Finds first recording frame and saves it."""
    loader = VideoLoader(video_path)
    cap = loader.cap
    
    print(f"Scanning video for first recording frame...")
    
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while frame_count < 2000:  # Check first 2000 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        if detect_recording_indicator(frame):
            print(f"Found recording indicator at frame {frame_count}")
            print(f"Video timestamp: {timestamp_ms:.1f}ms ({timestamp_ms/1000:.3f}s)")
            
            # Save the frame
            cv2.imwrite(output_path, frame)
            print(f"\nFrame saved to: {output_path}")
            print("Please open this image and read the UNIX timestamp from the upper left corner.")
            print("Then use: python Scripts/get_video_start_simple.py")
            
            # Also save a few frames around it for context
            for offset in [-5, -3, -1, 1, 3, 5]:
                frame_idx = max(0, frame_count + offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(f"frame_{frame_idx:05d}.png", frame)
            
            loader.close()
            return frame_count, timestamp_ms
    
    print("Could not find recording indicator in first 2000 frames.")
    loader.close()
    return None, None

if __name__ == "__main__":
    video_path = PROJECT_ROOT / "data" / "raw" / "Utensil" / "Utensil_Retake_001.mp4"
    frame_num, timestamp_ms = find_and_save_first_recording_frame(str(video_path))
    
    if frame_num is not None:
        print(f"\nFrame {frame_num} at {timestamp_ms/1000:.3f}s contains the first recording indicator.")
        print("Check 'first_recording_frame.png' to read the UNIX timestamp.")

