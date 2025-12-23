import cv2
import os
import logging
import json
from pathlib import Path

# Setup logger
logger = logging.getLogger(__name__)

class VideoCropper:
    def __init__(self, video_path, output_dir):
        """
        Initializes the VideoCropper.
        
        Args:
            video_path (str): Path to the source video file.
            output_dir (str): Directory where cropped videos will be saved.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get Video Metadata
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Calculate Video Start Time (Absolute) using Mtime - Duration heursitic
        # This assumes file modification time is the END of recording.
        self.video_mtime = os.path.getmtime(video_path)
        self.video_start_ts = self.video_mtime - self.duration_sec
        
        logger.info(f"Video Initialized: {video_path}")
        logger.info(f"Duration: {self.duration_sec:.2f}s, FPS: {self.fps}")
        logger.info(f"Estimated Start Time (Unix): {self.video_start_ts}")

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def crop_to_motion_files(self, motion_dir, start_timestamp_unix=None):
        """
        Iterates over JSON files in a directory and crops the video to match each file's duration.
        
        Args:
            motion_dir (str): Directory containing motion JSON files.
            start_timestamp_unix (float): Optional explicit start timestamp. If None, uses heuristic.
        """
        if start_timestamp_unix is not None:
            self.video_start_ts = start_timestamp_unix
            logger.info(f"Using Explicit Video Start Time: {self.video_start_ts}")
            
        files = sorted([os.path.join(motion_dir, f) for f in os.listdir(motion_dir) 
                        if f.endswith('.json') and not f.endswith('metadata.json') and not f.endswith('validation.json')])
        
        if not files:
            logger.warning(f"No valid JSON files found in {motion_dir}")
            return

        for json_path in files:
            self.process_single_file(json_path)

    def process_single_file(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            if 'trajectories' not in data or not data['trajectories']:
                logger.warning(f"Skipping {os.path.basename(json_path)}: No trajectories.")
                return
                
            traj = data['trajectories'][0]
            if not traj.get('timestamps'):
                logger.warning(f"Skipping {os.path.basename(json_path)}: No timestamps.")
                return

            motion_start_ts = traj['timestamps'][0]
            motion_end_ts = traj['timestamps'][-1]
            
            # Calculate offset from video start
            start_offset = motion_start_ts - self.video_start_ts
            end_offset = motion_end_ts - self.video_start_ts
            
            # Check bounds
            if end_offset < 0:
                logger.warning(f"Skipping {os.path.basename(json_path)}: Motion ends before video starts.")
                return
            if start_offset > self.duration_sec:
                logger.warning(f"Skipping {os.path.basename(json_path)}: Motion starts after video ends.")
                return
                
            # Clamp
            start_offset = max(0.0, start_offset)
            end_offset = min(self.duration_sec, end_offset)
            
            if end_offset <= start_offset:
                logger.warning(f"Skipping {os.path.basename(json_path)}: Duration <= 0.")
                return
                
            # Output filename
            json_name = Path(json_path).stem
            output_path = os.path.join(self.output_dir, f"{json_name}.mp4")
            
            # Use timestamp-based cropping to handle VFR
            self._write_clip_by_time(start_offset * 1000.0, end_offset * 1000.0, output_path)
            
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")

    def _write_clip_by_time(self, start_ms, end_ms, output_path):
        logger.info(f"Writing clip: {output_path} (Time {start_ms:.1f}ms - {end_ms:.1f}ms)")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # mp4v for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Seek to slightly before start to ensure we don't miss the first frame
        # (Seeking might land on IFRAME)
        seek_to = max(0, start_ms - 2000) 
        self.cap.set(cv2.CAP_PROP_POS_MSEC, seek_to)
        
        # Scan forward
        frames_written = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            current_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # Check range
            if current_ms < start_ms:
                continue
            if current_ms > end_ms:
                break
                
            out.write(frame)
            frames_written += 1
            
        out.release()
        logger.info(f"Clip saved. Written {frames_written} frames.")
