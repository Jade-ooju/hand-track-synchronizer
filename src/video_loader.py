import cv2
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class VideoLoader:
    def __init__(self, video_path):
        """
        Initializes the VideoLoader with a path to a video file.
        
        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            raise IOError(f"Could not open video file: {video_path}")
            
        logger.info(f"Successfully opened video: {video_path}")

    def get_metadata(self):
        """
        Returns metadata about the video.
        
        Returns:
            dict: Dictionary containing fps, resolution (width, height), frame_count, and duration_sec.
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = 0.0
        if fps > 0:
            duration = frame_count / fps
            
        metadata = {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": duration
        }
        return metadata

    def extract_frame_timestamps(self):
        """
        Extracts precise timestamps for each frame in the video.
        
        Returns:
            list: List of timestamps in milliseconds.
        """
        timestamps = []
        frame_idx = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            # We only need to grab the frame to advance the counter and get the timestamp
            # retrieve is not strictly necessary if we just rely on grab, but grab handles the advancement
            ret = self.cap.grab()
            if not ret:
                break
                
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(timestamp)
            frame_idx += 1
            
        logger.info(f"Extracted timestamps for {len(timestamps)} frames.")
        return timestamps

    def get_frame_at_timestamp(self, timestamp_ms, tolerance_ms=20):
        """
        Retrieves the frame closest to the specified timestamp.
        
        Args:
            timestamp_ms (float): Target timestamp in milliseconds.
            tolerance_ms (float): Maximum allowed difference in milliseconds.
            
        Returns:
            np.ndarray: The frame image, or None if not found/out of range.
        """
        # Set position by milliseconds
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(f"Could not read frame at timestamp {timestamp_ms}")
            return None
            
        # Verify actual timestamp
        actual_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        diff = abs(actual_timestamp - timestamp_ms)
        
        if diff > tolerance_ms:
            logger.warning(f"Frame at {actual_timestamp}ms is too far from requested {timestamp_ms}ms (diff: {diff}ms)")
            
        return frame
        
    def frame_generator(self):
        """
        Yields frames one by one correctly.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            yield timestamp, frame

    def close(self):
        """
        Releases the video capture resource.
        """
        if self.cap:
            self.cap.release()
