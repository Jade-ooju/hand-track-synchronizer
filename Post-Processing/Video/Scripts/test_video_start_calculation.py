"""
Test script to verify video start timestamp calculation.
Uses example values from the user's video frame.
"""
import sys

def calculate_video_start(unix_ts, time_offset):
    """
    Calculate video start timestamp.
    
    Args:
        unix_ts: UNIX timestamp visible in the frame
        time_offset: TIME value (seconds) visible in the frame
    """
    video_start = unix_ts - time_offset
    
    print("="*60)
    print("Video Start Timestamp Calculation Test")
    print("="*60)
    print(f"\nInput values from video frame:")
    print(f"  UNIX timestamp: {unix_ts}")
    print(f"  TIME offset: {time_offset} seconds")
    print(f"\nCalculation:")
    print(f"  Video Start = UNIX - TIME")
    print(f"  Video Start = {unix_ts} - {time_offset}")
    print(f"  Video Start = {video_start}")
    print("\n" + "="*60)
    print(f"RESULT: Video Start Timestamp = {video_start}")
    print("="*60)
    
    return video_start

if __name__ == "__main__":
    # Test with values from the user's image
    # From the image: UNIX: 1767431369.00, TIME: 00:09.70
    test_unix = 1767431369.00
    test_time = 9.70  # 00:09.70 = 9.70 seconds
    
    print("Testing with example values from video frame:")
    print(f"  UNIX: {test_unix}")
    print(f"  TIME: {test_time} seconds\n")
    
    result = calculate_video_start(test_unix, test_time)
    
    print(f"\nThis timestamp ({result}) should be used in:")
    print("  - config/pipeline_config_utensil.json (timestamp_offset)")
    print("  - Scripts/crop_utensil_video.py (calibrated_start)")
    
    # Also test with different time formats
    print("\n" + "-"*60)
    print("Testing different TIME formats:")
    print("-"*60)
    
    # Test MM:SS.ms format
    time_str = "00:09.70"
    parts = time_str.split(':')
    time_seconds = float(parts[0]) * 60 + float(parts[1])
    print(f"\nTIME format '{time_str}':")
    result2 = calculate_video_start(test_unix, time_seconds)
    
    # Test SS.ms format
    time_str2 = "9.70"
    print(f"\nTIME format '{time_str2}':")
    result3 = calculate_video_start(test_unix, float(time_str2))

