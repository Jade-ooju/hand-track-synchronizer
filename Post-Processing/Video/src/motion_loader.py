import json
import logging
import os
import bisect
import numpy as np

logger = logging.getLogger(__name__)

class MotionLoader:
    def __init__(self, json_path):
        """
        Initializes the MotionLoader with a path to a JSON motion log or directory of logs.
        
        Args:
            json_path (str): Path to the JSON file or directory.
        """
        self.json_path = json_path
        self.timestamps = []
        self.poses = []
        self.left_eye_poses = []
        self.right_eye_poses = []
        
        if os.path.isdir(json_path):
            self.load_directory(json_path)
        elif os.path.isfile(json_path):
            self.load_data(json_path)
        else:
            logger.error(f"Motion path not found: {json_path}")
            raise FileNotFoundError(f"Motion path not found: {json_path}")

    def load_directory(self, dir_path):
        """
        Loads all matching JSON files from a directory.
        """
        logger.info(f"Loading motion data from directory: {dir_path}")
        files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                        if f.endswith('.json') and not f.endswith('metadata.json') and not f.endswith('validation.json')])
        
        if not files:
            logger.warning(f"No valid JSON motion files found in {dir_path}")
            return

        for f in files:
            try:
                self.load_data(f, merge=True)
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
        
        # Sort combined data
        if self.timestamps:
             # Zip, sort, unzip
             # Combine all lists that need sorting
            combined = sorted(zip(self.timestamps, self.poses, self.left_eye_poses, self.right_eye_poses), key=lambda x: x[0])
            
            # Unzip
            self.timestamps, self.poses, self.left_eye_poses, self.right_eye_poses = zip(*combined)
            
            # Convert back to lists
            self.timestamps = list(self.timestamps)
            self.poses = list(self.poses)
            self.left_eye_poses = list(self.left_eye_poses)
            self.right_eye_poses = list(self.right_eye_poses)
            
        logger.info(f"Total poses loaded: {len(self.poses)}")

    def load_data(self, file_path, merge=False):
        """
        Loads the JSON data and parses it.
        Args:
            file_path: Path to file.
            merge: If true, extends existing lists instead of replacing.
        """
        logger.info(f"Loading motion data from {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if 'trajectories' not in data or not data['trajectories']:
                logger.warning(f"No trajectories found in JSON: {file_path}")
                return

            # Assume first trajectory is the main one for now
            trajectory = data['trajectories'][0]
            
            if 'timestamps' not in trajectory or 'poses' not in trajectory:
                logger.warning(f"Missing timestamps or poses in trajectory: {file_path}")
                return
            
            # raw_timestamps are floats
            current_timestamps = trajectory['timestamps']
            raw_poses = trajectory['poses']
            
            if len(current_timestamps) != len(raw_poses):
                logger.warning(f"Timestamp count ({len(current_timestamps)}) does not match pose count ({len(raw_poses)}). Truncating.")
                min_len = min(len(current_timestamps), len(raw_poses))
                current_timestamps = current_timestamps[:min_len]
                raw_poses = raw_poses[:min_len]
                
            # Convert poses to structured format
            current_poses = []
            for p in raw_poses:
                pose_dict = {
                    "position": p[0:3] if len(p)>=3 else [0,0,0],
                    "rotation": p[3:7] if len(p)>=7 else [0,0,0,1],
                    "gripper": 0.0
                }
                current_poses.append(pose_dict)
            
            # Process Eye Poses if available
            raw_left = trajectory.get('left_eye_poses', [])
            raw_right = trajectory.get('right_eye_poses', [])
            
            current_left = []
            for p in raw_left:
                 current_left.append({
                    "position": p[0:3] if len(p)>=3 else [0,0,0],
                    "rotation": p[3:7] if len(p)>=7 else [0,0,0,1]
                 })
                 
            current_right = []
            for p in raw_right:
                 current_right.append({
                     "position": p[0:3] if len(p)>=3 else [0,0,0],
                     "rotation": p[3:7] if len(p)>=7 else [0,0,0,1]
                 })

            # Pad eye poses if they are shorter than timestamps (though they should match)
            # Actually, standardizing on minimal length is safer
            min_len = min(len(current_timestamps), len(current_poses), len(current_left), len(current_right))
            if len(current_left) != len(current_timestamps):
                 # logger.warning(f"Eye pose count mismatch. Truncating to {min_len}")
                 pass
            
            current_timestamps = current_timestamps[:min_len]
            current_poses = current_poses[:min_len]
            current_left = current_left[:min_len]
            current_right = current_right[:min_len]
            
            if merge:
                self.timestamps.extend(current_timestamps)
                self.poses.extend(current_poses)
                self.left_eye_poses.extend(current_left)
                self.right_eye_poses.extend(current_right)
            else:
                self.timestamps = current_timestamps
                self.poses = current_poses
                self.left_eye_poses = current_left
                self.right_eye_poses = current_right
                
                # Check monotonicity only if not merging (merging sorts at the end)
                if not all(self.timestamps[i] <= self.timestamps[i+1] for i in range(len(self.timestamps)-1)):
                    logger.warning("Timestamps are not strictly sorted. Sorting now.")
                    combined = sorted(zip(self.timestamps, self.poses, self.left_eye_poses, self.right_eye_poses), key=lambda x: x[0])
                    self.timestamps, self.poses, self.left_eye_poses, self.right_eye_poses = zip(*combined)
                    self.timestamps = list(self.timestamps)
                    self.poses = list(self.poses)
                    self.left_eye_poses = list(self.left_eye_poses)
                    self.right_eye_poses = list(self.right_eye_poses)
                    
            if not merge:
                logger.info(f"Loaded {len(self.poses)} poses.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON {file_path}: {e}")
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
            return {
                "pose": self.poses[best_idx],
                "left_eye": self.left_eye_poses[best_idx] if best_idx < len(self.left_eye_poses) else None,
                "right_eye": self.right_eye_poses[best_idx] if best_idx < len(self.right_eye_poses) else None
            }
            
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
            # Standard: return lower and upper bound.
            curr = (self.timestamps[idx], {
                "pose": self.poses[idx],
                "left_eye": self.left_eye_poses[idx],
                "right_eye": self.right_eye_poses[idx]
            })
            return curr, curr

        # If we are here, timestamp < self.timestamps[idx] (if idx valid)
        
        if idx < len(self.timestamps):
            next_data = (self.timestamps[idx], {
                "pose": self.poses[idx],
                "left_eye": self.left_eye_poses[idx],
                "right_eye": self.right_eye_poses[idx]
            })
        
        if idx > 0:
            prev_data = (self.timestamps[idx-1], {
                "pose": self.poses[idx-1],
                "left_eye": self.left_eye_poses[idx-1],
                "right_eye": self.right_eye_poses[idx-1]
            })
            
        return prev_data, next_data
