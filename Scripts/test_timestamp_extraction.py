"""
Comprehensive test of timestamp extraction from video frames.
Extracts multiple frames around the recording start to verify detection.
"""
import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader

def detect_recording_indicator(frame):
    """Detects green recording indicator."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = np.sum(mask > 0)
    return green_pixel_count > 100, green_pixel_count

def test_video_scanning(video_path, max_frames=2000):
    """Scan video and report findings."""
    loader = VideoLoader(video_path)
    cap = loader.cap
    
    print("="*70)
    print("Video Timestamp Extraction Test")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"Total frames: {loader.total_frames}")
    print(f"FPS: {loader.fps:.2f}")
    print(f"Duration: {loader.total_frames / loader.fps:.2f}s")
    print(f"\nScanning first {max_frames} frames for recording indicator...\n")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    recording_frames = []
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0
        
        has_indicator, green_count = detect_recording_indicator(frame)
        
        if has_indicator:
            recording_frames.append({
                'frame': frame_count,
                'timestamp_ms': timestamp_ms,
                'timestamp_sec': timestamp_sec,
                'green_pixels': green_count
            })
            
            # Save first few recording frames
            if len(recording_frames) <= 5:
                filename = f"test_frame_{frame_count:05d}.png"
                cv2.imwrite(filename, frame)
                print(f"  Frame {frame_count:5d}: Recording detected | "
                      f"Time: {timestamp_sec:7.3f}s | "
                      f"Green pixels: {green_count:5d} | "
                      f"Saved: {filename}")
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 500 == 0:
            print(f"  Scanned {frame_count} frames...")
    
    loader.close()
    
    print("\n" + "="*70)
    print(f"Results: Found {len(recording_frames)} frames with recording indicator")
    print("="*70)
    
    if recording_frames:
        first = recording_frames[0]
        print(f"\nFirst recording frame:")
        print(f"  Frame number: {first['frame']}")
        print(f"  Video timestamp: {first['timestamp_ms']:.1f}ms ({first['timestamp_sec']:.3f}s)")
        print(f"  Saved as: test_frame_{first['frame']:05d}.png")
        print(f"\nTo get video start timestamp:")
        print(f"  1. Open test_frame_{first['frame']:05d}.png")
        print(f"  2. Read the UNIX timestamp from upper left corner")
        print(f"  3. Calculate: Video Start = UNIX - {first['timestamp_sec']:.3f}")
        print(f"\nOr use the simple calculator:")
        print(f"  python Scripts/get_video_start_simple.py")
    else:
        print("\nNo recording indicator found. You may need to:")
        print("  - Adjust the green color detection range")
        print("  - Check if recording starts later in the video")
        print("  - Manually inspect frames")
    
    return recording_frames

if __name__ == "__main__":
    video_path = "data/raw/Utensil/Utensil_Retake_001.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    recording_frames = test_video_scanning(video_path, max_frames=2000)
    
    if recording_frames:
        print(f"\n{'='*70}")
        print("Next steps:")
        print("="*70)
        print("1. Check the saved test_frame_*.png files")
        print("2. Read the UNIX timestamp from the first recording frame")
        print("3. Calculate video start using the formula shown above")
        print("4. Update config/pipeline_config_utensil.json with the new timestamp")
        print("5. Re-run the cropping script")

