import os
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.compression_service import CompressionService

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("OptimizeDataset")

def optimize_dataset(root_dir, codec="libx265", crf=32, resolution=None, dry_run=False):
    """
    Recursively finds .mp4 files in root_dir and compresses them if they don't have a '_compressed' suffix.
    """
    service = CompressionService()
    root_path = Path(root_dir)
    
    video_files = list(root_path.rglob("*.mp4"))
    
    # Filter out already compressed files and visualization files if desired
    # For now, let's target files that are large or match raw data patterns
    targets = []
    for f in video_files:
        if "_compressed" in f.name:
            continue
        # Skip common visualization outputs to avoid re-compressing them unless specifically requested
        if f.name.startswith("viz_") or f.name.startswith("comparison_"):
            continue
        targets.append(f)
        
    logger.info(f"Found {len(targets)} potential target videos for optimization.")
    
    total_saved = 0
    
    for input_file in targets:
        output_file = input_file.with_name(f"{input_file.stem}_compressed.mp4")
        
        if output_file.exists():
            logger.info(f"Skipping {input_file.name}, compressed version already exists.")
            continue
            
        if dry_run:
            logger.info(f"[DRY RUN] Would compress {input_file} to {output_file}")
            continue
            
        success = service.compress_video(
            str(input_file), 
            str(output_file), 
            codec=codec, 
            crf=crf, 
            resolution=resolution
        )
        
        if success:
            stats = service.get_compression_stats(str(input_file), str(output_file))
            saved = stats['input_size_mb'] - stats['output_size_mb']
            total_saved += saved
            logger.info(f"  Saved: {saved:.2f} MB ({stats['reduction_percent']:.1f}%)")
            
    logger.info(f"\nOptimization complete. Total storage saved: {total_saved:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Optimize Video Dataset Storage')
    parser.add_argument('--dir', default='data/raw', help='Root directory to search for videos')
    parser.add_argument('--codec', default='libx265', help='Codec to use (libx264, libx265, vp9)')
    parser.add_argument('--crf', type=int, default=32, help='CRF value (lower is higher quality, 23-32 recommended)')
    parser.add_argument('--scale', help='Resolution scaling (e.g., 512:512)')
    parser.add_argument('--dry-run', action='store_true', help='Only list files that would be compressed')
    
    args = parser.parse_args()
    
    resolution = None
    if args.scale:
        try:
            w, h = map(int, args.scale.split(':'))
            resolution = (w, h)
        except ValueError:
            logger.error("Invalid scale format. Use W:H")
            return

    optimize_dataset(args.dir, codec=args.codec, crf=args.crf, resolution=resolution, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
