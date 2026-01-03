"""
Simple script to calculate video start timestamp from a frame's visible timestamp.
Usage: Run this and enter the UNIX and TIME values you see in the first recording frame.
"""
import sys

def calculate_video_start():
    print("="*60)
    print("Video Start Timestamp Calculator")
    print("="*60)
    print("\nFind the FIRST frame where recording starts (green circle appears).")
    print("Read the UNIX and TIME values from the upper left corner of that frame.\n")
    
    try:
        unix_str = input("Enter UNIX timestamp (e.g., 1767431369.00): ").strip()
        time_str = input("Enter TIME value (e.g., 00:09.70 or just 9.70): ").strip()
        
        # Parse UNIX timestamp
        unix_ts = float(unix_str)
        
        # Parse TIME (handle both MM:SS.ms and SS.ms formats)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                time_offset = float(minutes) * 60 + float(seconds)
            else:
                time_offset = float(time_str)
        else:
            time_offset = float(time_str)
        
        # Calculate video start
        video_start = unix_ts - time_offset
        
        print("\n" + "="*60)
        print(f"VIDEO START TIMESTAMP: {video_start}")
        print("="*60)
        print(f"\nCalculation:")
        print(f"  UNIX timestamp from frame: {unix_ts}")
        print(f"  TIME offset in video: {time_offset:.3f} seconds")
        print(f"  Video start = {unix_ts} - {time_offset:.3f} = {video_start}")
        print(f"\nUse this value in your pipeline config as 'timestamp_offset'")
        print("="*60)
        
        return video_start
        
    except ValueError as e:
        print(f"Error: Invalid input format. {e}")
        return None
    except KeyboardInterrupt:
        print("\nCancelled.")
        return None

if __name__ == "__main__":
    calculate_video_start()

