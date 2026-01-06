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
git clone https://github.com/Jade-ooju/hand-track-synchronizer.git
cd hand-track-synchronizer
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

- **[USAGE.md](USAGE.md)** - Complete pipeline guide with step-by-step instructions
- **[docs/project_guidelines.md](docs/project_guidelines.md)** - Development guidelines
- **[docs/video_cropping_report.md](docs/video_cropping_report.md)** - Video cropping methodology

## Usage
📖 **For detailed usage instructions, see [USAGE.md](USAGE.md)**

## Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Jade-ooju/hand-track-synchronizer.git
    cd hand-track-synchronizer
    ```

2.  **Create and activate a virtual environment:**
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

4.  **Configuration:**
    Modify `config/config.json` to point to your data paths.
