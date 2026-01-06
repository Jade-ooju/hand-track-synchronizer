# Hand Track Synchronizer

A pipeline for frame-accurate synchronization of egocentric video with 3D hand pose data.

## Description
This project aims to rebuild a video-motion synchronization system to- **Video Loading**: `VideoLoader` for frame extraction with precise `CAP_PROP_POS_MSEC` timestamps.
- **Motion Loading**: `MotionLoader` parses JSON logs, supporting both single files and directory-based loading (multi-session).
- **Matching**: `MotionMatcher` aligns video frames to motion data, calculating interpolation weights.
- **Cropping**: `VideoCropper` splits long recordings into session-specific clips using VFR-safe timestamp cropping.
- **Interpolation**: `Interpolator` generates smooth poses using Slerp for rotations and Lerp for positions.
- **Visualization**: `Visualizer` projects 3D poses onto 2D video with manual calibration support.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Pipeline Workflow
```bash
# Optional: Crop video into session clips
python Scripts/run_video_cropping.py

# Calibrate 3D-to-2D projection
python Scripts/calibrate_projection.py

# Generate visualization
python Scripts/run_visualization.py
```

## Documentation

- **[DATA_STRUCTURE.md](DATA_STRUCTURE.md)** - Where to place video files and motion data
- **[USAGE.md](USAGE.md)** - Complete pipeline guide with step-by-step instructions

## Usage
📖 **For detailed usage instructions, see [USAGE.md](USAGE.md)**

## Setup
1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Setup:**
    Create the data directory structure and place your files. See **[DATA_STRUCTURE.md](DATA_STRUCTURE.md)** for details.
    
    **Quick setup:**
    ```bash
    mkdir -p data/raw/your_dataset
    # Copy your video and JSON files to data/raw/your_dataset/
    ```

5.  **Configuration:**
    Modify `config/pipeline_config.json` to point to your data paths.
