import sys
import os
import json
import cv2
import numpy as np
import logging
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video_loader import VideoLoader
from src.motion_loader import MotionLoader
from src.motion_matcher import MotionMatcher
from src.interpolator import Interpolator
from src.visualizer import Visualizer
from src.compression_service import CompressionService

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Pipeline")

def validate_calibration(config_path, skip_check=False, force_recalibrate=False):
    """
    Stage 0: Calibration Validation
    Returns: True if calibration is OK, False if user wants to exit
    """
    if force_recalibrate:
        logger.info("Force recalibration requested...")
        run_calibration_tool()
        return True
    
    if not os.path.exists(config_path):
        logger.warning(f"No calibration file found at {config_path}")
        logger.warning("Please run calibration first: python Scripts/calibrate_projection.py")
        if not skip_check:
            response = input("Run calibration now? (y/n): ").strip().lower()
            if response == 'y':
                run_calibration_tool()
                return True
            else:
                logger.info("Exiting. Please calibrate before running pipeline.")
                return False
        return False
    
    # Load and display calibration
    with open(config_path, 'r') as f:
        calib = json.load(f)
    
    logger.info("\n" + "="*50)
    logger.info("CALIBRATION CHECK")
    logger.info("="*50)
    logger.info(f"Calibration file: {config_path}")
    logger.info(f"  Offset Position: {calib.get('offset_pos', [0,0,0])}")
    logger.info(f"  Offset Rotation: {calib.get('offset_rot_euler', [0,0,0])}")
    logger.info(f"  FOV: {calib.get('fov', 100)}")
    logger.info("="*50 + "\n")
    
    if skip_check:
        logger.info("Skipping calibration check (--skip-calib-check)")
        return True
    
    # Interactive prompt
    while True:
        response = input("Proceed with this calibration? (y/n/recalibrate): ").strip().lower()
        if response == 'y':
            logger.info("Calibration accepted. Proceeding...\n")
            return True
        elif response == 'n':
            logger.info("Calibration rejected. Exiting.")
            logger.info("Run 'python Scripts/calibrate_projection.py' to adjust calibration.")
            return False
        elif response == 'recalibrate':
            run_calibration_tool()
            return True
        else:
            print("Please enter 'y', 'n', or 'recalibrate'")

def run_calibration_tool():
    """Launch interactive calibration tool."""
    logger.info("Launching calibration tool...")
    calib_script = os.path.join(os.path.dirname(__file__), 'calibrate_projection.py')
    subprocess.run([sys.executable, calib_script])
    logger.info("Calibration complete. Continuing with pipeline...\n")

def run_pipeline(config_path, skip_calib_check=False, force_recalibrate=False):
    """Main pipeline orchestrator."""
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    video_path = config['video_path']
    motion_dir = config['motion_dir']
    output_dir = config['output_dir']
    calibration_path = config['calibration_path']
    timestamp_offset = config['timestamp_offset']
    options = config['options']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 0: Calibration Validation
    if not validate_calibration(calibration_path, skip_calib_check, force_recalibrate):
        return
    
    # Stage 1: Data Loading
    logger.info("Stage 1: Loading Data...")
    
    # Check for compressed version of the video
    use_compressed = options.get('use_compressed', True)
    if use_compressed:
        video_path_obj = Path(video_path)
        compressed_path = video_path_obj.with_name(f"{video_path_obj.stem}_compressed.mp4")
        if compressed_path.exists():
            logger.info(f"  Found compressed video: {compressed_path}")
            video_path = str(compressed_path)
        else:
            logger.info(f"  No compressed video found at {compressed_path}, using original.")
            
    v_loader = VideoLoader(video_path)
    m_loader = MotionLoader(motion_dir)
    
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Frames: {v_loader.total_frames}, FPS: {v_loader.fps}")
    logger.info(f"  Motion: {len(m_loader.timestamps)} poses loaded")
    
    # Stage 2: Timestamp Processing
    logger.info("\nStage 2: Timestamp Processing...")
    timestamps_ms = v_loader.extract_frame_timestamps()
    v_timestamps_sec = [t / 1000.0 for t in timestamps_ms]
    
    matcher = MotionMatcher(m_loader)
    matches = matcher.match_timestamps(v_timestamps_sec, offset_ms=timestamp_offset)
    logger.info(f"  Matched {len(matches)} video frames to motion data")
    
    # Stage 3: Pose Interpolation
    logger.info("\nStage 3: Pose Interpolation...")
    interpolator = Interpolator()
    gap_threshold = options['gap_threshold']
    
    synced_frames = []
    gaps_detected = 0
    
    for frame_idx, match in enumerate(tqdm(matches, desc="  Interpolating poses")):
        frame_data = {
            'frame_idx': frame_idx,
            'video_timestamp': match['aligned_ts'],
            'hand_pose': None,
            'camera_pose': None,
            'interpolation_weight': match['weight'],
            'in_gap': False
        }
        
        prev_data, next_data = m_loader.get_surrounding_poses(match['aligned_ts'])
        
        if prev_data and next_data:
            t1, d1 = prev_data
            t2, d2 = next_data
            
            # Check for gap
            if (t2 - t1) > gap_threshold:
                frame_data['in_gap'] = True
                gaps_detected += 1
            else:
                w = match['weight']
                
                # Interpolate hand pose
                hand_pose = interpolator.interpolate_pose(d1['pose'], d2['pose'], w)
                frame_data['hand_pose'] = hand_pose
                frame_data['source_timestamps'] = [t1, t2]
                
                # Interpolate camera pose
                le_pose = None
                if d1['left_eye'] and d2['left_eye']:
                    le_pose = interpolator.interpolate_pose(d1['left_eye'], d2['left_eye'], w)
                    
                re_pose = None
                if d1['right_eye'] and d2['right_eye']:
                    re_pose = interpolator.interpolate_pose(d1['right_eye'], d2['right_eye'], w)
                    
                if le_pose and re_pose:
                    pos = (np.array(le_pose['position']) + np.array(re_pose['position'])) / 2.0
                    camera_pose = {'position': pos.tolist(), 'rotation': le_pose['rotation']}
                    frame_data['camera_pose'] = camera_pose
                elif le_pose:
                    frame_data['camera_pose'] = le_pose
                elif re_pose:
                    frame_data['camera_pose'] = re_pose
        
        synced_frames.append(frame_data)
    
    logger.info(f"  Interpolation complete. Gaps detected: {gaps_detected}")
    
    # Stage 4: Visualization (Optional)
    if options['generate_visualization']:
        logger.info("\nStage 4: Generating Visualization...")
        
        viz_type = options['visualization_type']
        output_video_path = os.path.join(output_dir, f'viz_{viz_type}.mp4')
        
        visualizer = Visualizer(
            width=int(v_loader.width),
            height=int(v_loader.height),
            config_path=calibration_path
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, v_loader.fps, 
                              (int(v_loader.width), int(v_loader.height)))
        
        v_loader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        for frame_idx in tqdm(range(len(synced_frames)), desc=f"  Rendering {viz_type}"):
            ret, frame = v_loader.cap.read()
            if not ret:
                break
            
            frame_data = synced_frames[frame_idx]
            
            if not frame_data['in_gap'] and frame_data['hand_pose'] and frame_data['camera_pose']:
                if viz_type == 'gizmo':
                    visualizer.draw_gizmo(frame, frame_data['hand_pose'], 
                                         frame_data['camera_pose'], apply_calibration=True)
                elif viz_type == 'comparison':
                    # For comparison, we'd need raw pose too - simplified here
                    visualizer.draw_hand_point(frame, frame_data['hand_pose'], 
                                              frame_data['camera_pose'], 
                                              color=(0, 255, 255), label="SYNCED")
            
            out.write(frame)
        
        out.release()
        logger.info(f"  Visualization saved: {output_video_path}")
    
    # Stage 5: Output Generation
    logger.info("\nStage 5: Generating Outputs...")
    
    # Export synced JSON
    if options['export_synced_json']:
        synced_json_path = os.path.join(output_dir, 'synced_poses.json')
        
        # Get motion source files
        motion_files = [f for f in os.listdir(motion_dir) 
                       if f.endswith('.json') and not f.endswith('_metadata.json') 
                       and not f.endswith('_validation.json')]
        
        synced_data = {
            'metadata': {
                'video_path': video_path,
                'motion_sources': motion_files,
                'total_frames': len(synced_frames),
                'fps': v_loader.fps,
                'timestamp_offset': timestamp_offset,
                'gap_threshold': gap_threshold,
                'gaps_detected': gaps_detected,
                'processing_date': datetime.now().isoformat()
            },
            'frames': synced_frames
        }
        
        with open(synced_json_path, 'w') as f:
            json.dump(synced_data, f, indent=2)
        
        logger.info(f"  Synced data saved: {synced_json_path}")
    
    # Generate processing report
    if options['generate_report']:
        report_path = os.path.join(output_dir, 'processing_report.md')
        generate_report(report_path, config, synced_frames, gaps_detected)
        logger.info(f"  Report saved: {report_path}")
    
    logger.info("\n" + "="*50)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*50)
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*50 + "\n")
    
    v_loader.close()

def generate_report(report_path, config, synced_frames, gaps_detected):
    """Generate processing report markdown."""
    
    frames_with_pose = sum(1 for f in synced_frames if f['hand_pose'] is not None)
    frames_in_gap = sum(1 for f in synced_frames if f['in_gap'])
    
    report = f"""# VideoSync Processing Report

## Processing Information
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Configuration**: {config.get('video_path', 'N/A')}

## Input Data
- **Video**: `{config['video_path']}`
- **Motion Directory**: `{config['motion_dir']}`
- **Total Frames**: {len(synced_frames)}
- **Timestamp Offset**: {config['timestamp_offset']}

## Processing Statistics
- **Frames with Synchronized Pose**: {frames_with_pose} ({frames_with_pose/len(synced_frames)*100:.1f}%)
- **Frames in Gaps**: {frames_in_gap} ({frames_in_gap/len(synced_frames)*100:.1f}%)
- **Gap Threshold**: {config['options']['gap_threshold']}s

## Configuration
- **Visualization Type**: {config['options']['visualization_type']}
- **Generate Visualization**: {config['options']['generate_visualization']}
- **Export Synced JSON**: {config['options']['export_synced_json']}

## Output Files
- Synced poses: `synced_poses.json`
- Visualization: `viz_{config['options']['visualization_type']}.mp4` (if enabled)
- Report: `processing_report.md`

## Notes
- Frames marked as "in_gap" indicate periods where motion recording was paused
- Interpolation was skipped for gap frames to avoid artifacts
- All timestamps are in Unix epoch seconds
"""
    
    with open(report_path, 'w') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description='VideoSync Full Pipeline Orchestrator')
    parser.add_argument('--config', default='config/pipeline_config.json',
                       help='Path to pipeline configuration file')
    parser.add_argument('--skip-calib-check', action='store_true',
                       help='Skip interactive calibration validation')
    parser.add_argument('--recalibrate', action='store_true',
                       help='Force recalibration before processing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please create a config file or use default: config/pipeline_config.json")
        return
    
    run_pipeline(args.config, args.skip_calib_check, args.recalibrate)

if __name__ == "__main__":
    main()
