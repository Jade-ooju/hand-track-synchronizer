import json
import logging
import os
import bisect
import numpy as np

logger = logging.getLogger(__name__)

class MotionLoader:
    def __init__(self, json_path):
        """
        Initializes the MotionLoader with a path to a JSON motion log.
        
        Args:
            json_path (str): Path to the JSON file.
        """
        self.json_path = json_path
        if not os.path.exists(json_path):
            logger.error(f"Motion file not found: {json_path}")
            raise FileNotFoundError(f"Motion file not found: {json_path}")
            
        self.timestamps = []
        self.poses = [] # List of dicts or numpy array? List of dicts matching structure
        self.load_data()

    def load_data(self):
        """
        Loads the JSON data and parses it into efficient structures.
        """
        logger.info(f"Loading motion data from {self.json_path}")
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
                
            if 'trajectories' not in data or not data['trajectories']:
                logger.error("No trajectories found in JSON.")
                raise ValueError("No trajectories found in JSON.")
            
            # Assume first trajectory is the main one for now
            trajectory = data['trajectories'][0]
            
            if 'timestamps' not in trajectory or 'poses' not in trajectory:
                logger.error("Missing timestamps or poses in trajectory.")
                raise ValueError("Missing timestamps or poses in trajectory.")
            
            # raw_timestamps are floats
            self.timestamps = trajectory['timestamps']
            raw_poses = trajectory['poses']
            
            if len(self.timestamps) != len(raw_poses):
                logger.warning(f"Timestamp count ({len(self.timestamps)}) does not match pose count ({len(raw_poses)}). Truncating to minimum.")
                min_len = min(len(self.timestamps), len(raw_poses))
                self.timestamps = self.timestamps[:min_len]
                raw_poses = raw_poses[:min_len]
                
            # Check monotonicity of timestamps
            if not all(self.timestamps[i] <= self.timestamps[i+1] for i in range(len(self.timestamps)-1)):
                logger.warning("Timestamps are not strictly sorted. Sorting now.")
                # Zip, sort, unzip
                combined = sorted(zip(self.timestamps, raw_poses), key=lambda x: x[0])
                self.timestamps, raw_poses = zip(*combined)
                self.timestamps = list(self.timestamps)
                raw_poses = list(raw_poses)
            
            # Convert poses to structured format
            self.poses = []
            for p in raw_poses:
                # p is a list of 7 floats: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w] (or similar)
                # Providing a structured dict
                if len(p) >= 7:
                    pose_dict = {
                        "position": p[0:3],
                        "rotation": p[3:7],
                        "gripper": 0.0 # Default value as not in 7D pose
                    }
                    self.poses.append(pose_dict)
                else:
                    logger.warning(f"Pose data has unexpected length: {len(p)}. Skipping.")
                    # Keep lists aligned! If we skip a pose, we must remove the timestamp? 
                    # Actually better to create a dummy or partial to keep alignment with timestamp index
                    # Or better yet, filtering should happen before zipping.
                    # For now assume valid 7D data based on inspection.
                    pose_dict = {
                        "position": p[0:3] if len(p)>=3 else [0,0,0],
                        "rotation": p[3:7] if len(p)>=7 else [0,0,0,1],
                        "gripper": 0.0
                    }
                    self.poses.append(pose_dict)
                    
            logger.info(f"Loaded {len(self.poses)} poses.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {e}")
            raise

    def get_time_range(self):
        """
        Returns the min and max timestamps.
        
        Returns:
            tuple: (min_timestamp, max_timestamp) or (None, None) if empty.
        """
        if not self.timestamps:
            return None, None
        return self.timestamps[0], self.timestamps[-1]

    def get_pose_at_timestamp(self, timestamp, tolerance=0.0):
        """
        Returns the pose exactly at, or closest to, the timestamp if within tolerance.
        
        Args:
            timestamp (float): The timestamp to query.
            tolerance (float): Max difference allowed. If 0.0, requires exact or very close float match.
            
        Returns:
            dict: Pose object or None.
        """
        if not self.timestamps:
            return None
            
        idx = bisect.bisect_left(self.timestamps, timestamp)
        
        # idx is where timestamp could be inserted. 
        # Check idx and idx-1 for closest match
        
        candidates = []
        if idx < len(self.timestamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        best_idx = -1
        min_diff = float('inf')
        
        for i in candidates:
            diff = abs(self.timestamps[i] - timestamp)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
                
        if best_idx != -1:
            if tolerance > 0 and min_diff > tolerance:
                return None
            return self.poses[best_idx]
            
        return None

    def get_surrounding_poses(self, timestamp):
        """
        Returns the two poses bounding the given timestamp for interpolation.
        
        Args:
            timestamp (float): The timestamp to query.
            
        Returns:
            tuple: ( (prev_time, prev_pose), (next_time, next_pose) )
                   Returns None for a side if out of bounds.
        """
        if not self.timestamps:
            return None, None
            
        idx = bisect.bisect_left(self.timestamps, timestamp)
        
        prev_data = None
        next_data = None
        
        # idx points to the first element >= timestamp
        
        # If timestamp is exactly at idx
        if idx < len(self.timestamps) and self.timestamps[idx] == timestamp:
            # Exact match. Both prev and next could be this? Or strictly surrounding?
            # Usually for interpolation, if t == t_i, we just return that.
            # But strictly speaking, "surrounding" implies t_i <= t <= t_{i+1}
            # Let's return the interval [t_{idx}, t_{idx+1}] if exists, or [t_{idx-1}, t_{idx}]?
            # Standard: return lower and upper bound.
            curr = (self.timestamps[idx], self.poses[idx])
            return curr, curr

        # If we are here, timestamp < self.timestamps[idx] (if idx valid)
        
        if idx < len(self.timestamps):
            next_data = (self.timestamps[idx], self.poses[idx])
        
        if idx > 0:
            prev_data = (self.timestamps[idx-1], self.poses[idx-1])
            
        return prev_data, next_data
