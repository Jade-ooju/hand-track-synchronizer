import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CompressionService:
    """
    Utility service to handle video compression using FFmpeg.
    """
    
    def __init__(self, ffmpeg_path="ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        try:
            subprocess.run([self.ffmpeg_path, "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found. Please ensure it is installed and in your PATH.")
            raise RuntimeError("FFmpeg not found.")

    def compress_video(self, input_path, output_path, codec="libx265", crf=32, preset="veryfast", resolution=None):
        """
        Compresses a video file using specified parameters.
        
        Args:
            input_path (str): Path to input video.
            output_path (str): Path to output video.
            codec (str): FFmpeg video codec (default: libx265).
            crf (int): Constant Rate Factor (default: 32).
            preset (str): FFmpeg preset (default: veryfast).
            resolution (tuple): Optional (width, height) to scale to.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not os.path.exists(input_path):
            logger.error(f"Input video not found: {input_path}")
            return False
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
        ]
        
        if resolution:
            width, height = resolution
            cmd.extend(["-vf", f"scale={width}:{height}"])
            
        # Audio handling (copy if exists, otherwise ignore)
        cmd.extend(["-c:a", "copy"])
        
        # Output path MUST be the last argument
        cmd.extend(["-y", str(output_path)])
        
        logger.info(f"Compressing {input_path} -> {output_path} (codec={codec}, crf={crf})")
        
        try:
            # We use capture_output=True to keep logs clean unless there is an error
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully compressed {input_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Compression failed for {input_path}")
            logger.error(f"FFmpeg error: {e.stderr}")
            return False

    def get_video_size(self, file_path):
        """Returns file size in bytes."""
        return os.path.getsize(file_path)

    def get_compression_stats(self, input_path, output_path):
        """Returns a dict with compression statistics."""
        input_size = self.get_video_size(input_path)
        output_size = self.get_video_size(output_path)
        reduction = (1 - output_size / input_size) * 100
        
        return {
            "input_size_mb": input_size / (1024 * 1024),
            "output_size_mb": output_size / (1024 * 1024),
            "reduction_percent": reduction
        }
