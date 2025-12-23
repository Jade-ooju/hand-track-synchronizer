import logging
import bisect

logger = logging.getLogger(__name__)

class MotionMatcher:
    def __init__(self, motion_loader):
        """
        Initializes the MotionMatcher.
        
        Args:
            motion_loader: Instance of MotionLoader class containing loaded data.
        """
        self.motion_loader = motion_loader
        
    def match_timestamps(self, video_timestamps, offset_ms=0.0):
        """
        Matches video timestamps to motion timestamps.
        
        Args:
            video_timestamps (list): List of video frame timestamps in milliseconds.
            offset_ms (float): Calibration offset to add to video timestamps (video_ts + offset = motion_ts).
                               Default is 0.0.
            
        Returns:
            list: List of tuples (video_ts, prev_motion_ts, next_motion_ts, weight)
                  weight is 0.0 if at prev_motion_ts, 1.0 if at next_motion_ts.
        """
        matches = []
        
        if not self.motion_loader.timestamps:
            logger.warning("No motion timestamps available for matching.")
            return []

        # Assuming motion timestamps are in seconds? 
        # Wait, previous inspection showed timestamps like: 137208.35.
        # This looks like seconds since epoch or boot? 
        # Video timestamps from OpenCV (CAP_PROP_POS_MSEC) are in milliseconds from start of file (0.0).
        # We need to know if we are aligning absolute times or relative times.
        # The user description says "Extract precise frame timestamps from video" and "Handle two calibration challenges: Timestamp alignment".
        # If we don't have the start time of the video in the same clock as motion, we just have relative 0.0 -> duration.
        # The motion data likely has absolute timestamps.
        # The user mentioned "Handle Video and motion start times don't align".
        # For this task, "Calculates interpolation parameters", we probably assume the timestamps passed in are ALREADY in the same domain 
        # OR we treat the offset as the alignment parameter.
        # Let's assume input `video_timestamps` are raw (0...N) and `offset_ms` aligns them to `motion_timestamps`.
        # However, motion timestamps in the test file were 0-based relative? 
        # Let's check the test file again.
        # Scripts/inspect_json.py output: "Timestamp 0: 0.0382090062". 
        # Wait, that was a float > 0. It looks like relative time from start of recording.
        # If both are relative 0-based, then offset is just small drift correction.
        # If one is absolute, we need a large offset.
        # The `match_timestamps` method should support generic float alignment.
        
        # NOTE: Provide `offset` support to shift video time to motion time domain.
        
        motion_ts = self.motion_loader.timestamps
        min_motion = motion_ts[0]
        max_motion = motion_ts[-1]
        
        for v_ts in video_timestamps:
            # Apply offset to align domains
            aligned_ts = v_ts + offset_ms
            
            # Find surrounding keys
            # bisect_left returns insertion point i such that all e in a[:i] < x
            idx = bisect.bisect_left(motion_ts, aligned_ts)
            
            prev_ts = None
            next_ts = None
            weight = 0.0
            
            if idx == 0:
                # Before start or at start
                prev_ts = motion_ts[0]
                next_ts = motion_ts[0] # or motion_ts[1] if exists?
                if len(motion_ts) > 1:
                     next_ts = motion_ts[1] # For extrapolation? Or just Clamp?
                     # Requirement: "Identify previous and next motion data points (for interpolation)"
                     # "Handle edge cases (video start/end)"
                     # If before start, we can't interpolate. We clamp to start.
                     weight = 0.0 # aligned matches prev (start)
                else:
                    weight = 0.0
            elif idx >= len(motion_ts):
                # After end
                prev_ts = motion_ts[-1]
                next_ts = motion_ts[-1]
                weight = 1.0 # (or 0.0 relative to prev/next being same)
            else:
                # Between idx-1 and idx
                prev_ts = motion_ts[idx-1]
                next_ts = motion_ts[idx]
                
                # Calculate weight
                # Linear interpolation: val = prev * (1-w) + next * w
                # w = (current - prev) / (next - prev)
                denominator = next_ts - prev_ts
                if denominator > 1e-9: # Avoid division by zero
                    weight = (aligned_ts - prev_ts) / denominator
                else:
                    weight = 0.0 # Same timestamps
            
                # Clamp weight [0, 1] just in case of float issues, though strictly it should be inside
                weight = max(0.0, min(1.0, weight))

            matches.append({
                "video_ts_raw": v_ts,
                "aligned_ts": aligned_ts,
                "prev_motion_ts": prev_ts,
                "next_motion_ts": next_ts,
                "weight": weight
            })
            
        return matches
