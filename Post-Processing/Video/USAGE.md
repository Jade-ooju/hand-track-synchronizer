# VideoSync Pipeline - Usage Guide

## Overview
VideoSync synchronizes 3D hand pose data with MR video recordings, enabling precise temporal and spatial alignment for analysis and visualization.

## Quick Start

### 1. Setup
```bash
# Clone repository
git clone https://github.com/Jade-ooju/hand-track-synchronizer.git
cd hand-track-synchronizer

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Place your data in `data/raw/`:
```
data/raw/test_002/
├── MR_View.mp4                              # Video recording
├── T_Video_20251223_120927.json            # Motion data session 1
├── T_Video_20251223_120932.json            # Motion data session 2
└── ...                                      # Additional motion files
```

## Pipeline Workflow

### Recommended: Full Pipeline Orchestrator
The easiest way to process your data is using the full pipeline orchestrator:

```bash
python Scripts/run_full_pipeline.py
```

This single command will:
1. ✓ Validate calibration (interactive prompt)
2. ✓ Load video and motion data
3. ✓ Match and interpolate poses
4. ✓ Generate visualization
5. ✓ Export synced dataset JSON
6. ✓ Create processing report

**Output** (`data/synced/test_002/`):
- `synced_poses.json` - Per-frame synchronized pose data
- `viz_gizmo.mp4` - Visualization video
- `processing_report.md` - Processing summary

**Options:**
```bash
# Skip calibration prompt (use existing)
python Scripts/run_full_pipeline.py --skip-calib-check

# Force recalibration before processing
python Scripts/run_full_pipeline.py --recalibrate

# Custom config
python Scripts/run_full_pipeline.py --config my_config.json
```

---

### Manual Step-by-Step (Advanced)
For more control, run individual pipeline stages:

### Step 1: Video Cropping (Optional)
If you have a single long video with multiple motion recording sessions:

```bash
python Scripts/run_video_cropping.py
```

**What it does:**
- Splits the main video into clips aligned with each motion JSON file
- Uses calibrated timestamp offset for precise alignment
- Outputs to `data/raw/test_002/cropped/`

**Configuration:**
Edit `Scripts/run_video_cropping.py` to adjust:
- `video_path`: Path to source video
- `json_dir`: Directory containing motion JSON files
- `output_dir`: Where to save cropped clips
- `calibrated_start`: Unix timestamp for video start time

---

### Step 2: Interactive Calibration
Calibrate the 3D-to-2D projection to align hand poses with video:

```bash
python Scripts/calibrate_projection.py
```

**What it does:**
- Opens an interactive window showing the video with overlay
- Allows real-time adjustment of projection parameters
- Saves calibration to `config/calibration.json`

#### Calibration Controls

| Key | Action | Description |
|-----|--------|-------------|
| **W** | +Z offset | Move hand forward in depth |
| **S** | -Z offset | Move hand backward in depth |
| **A** | -X offset | Move hand left |
| **D** | +X offset | Move hand right |
| **Q** | -Y offset | Move hand down |
| **E** | +Y offset | Move hand up |
| **I** | +Pitch | Rotate around X-axis |
| **K** | -Pitch | Rotate around X-axis |
| **J** | +Yaw | Rotate around Y-axis |
| **L** | -Yaw | Rotate around Y-axis |
| **U** | +Roll | Rotate around Z-axis |
| **O** | -Roll | Rotate around Z-axis |
| **+** / **=** | Increase FOV | Wider field of view |
| **-** | Decrease FOV | Narrower field of view |
| **,** | Previous frames | Jump back 10 frames |
| **.** | Next frames | Jump forward 10 frames |
| **SPACE** | Save | Save current calibration |
| **R** | Reset | Reset to default values |
| **ESC** | Exit | Close calibration tool |

#### Calibration Tips
1. **Start with position offsets (W/A/S/D/Q/E)** to get the gizmo roughly aligned with the hand
2. **Use rotation offsets (I/K/J/L/U/O)** if the axes orientation doesn't match
3. **Adjust FOV (+/-)** if the hand appears at the wrong scale/distance
4. **Press SPACE frequently** to save your progress
5. **Use ,/. keys** to check alignment across different frames

**Output:**
Saves calibration parameters to:
```
config/calibration.json
```

Example calibration file:
```json
{
    "offset_pos": [-0.11, 0.0, -0.13],
    "offset_rot_euler": [0, 0, 0],
    "fov": 105.0
}
```

---

### Step 3: Generate Visualization
Create a video with synchronized hand pose overlay:

```bash
python Scripts/run_visualization.py
```

**What it does:**
- Loads video and motion data
- Matches timestamps between video frames and motion samples
- Interpolates hand poses for smooth animation
- Applies calibration from `config/calibration.json`
- Renders RGB axes (gizmo) at the hand position
- Outputs to `data/raw/test_002/viz_output.mp4`

**Features:**
- **Red axis**: X direction (right)
- **Green axis**: Y direction (up)
- **Blue axis**: Z direction (forward)
- **Yellow dot**: Hand position (origin)
- **Gap detection**: Automatically skips drawing during gaps between recording sessions

**Configuration:**
Edit `Scripts/run_visualization.py` to adjust:
- `video_path`: Source video
- `json_dir`: Motion data directory
- `output_path`: Output video path
- `calibrated_start`: Video start timestamp
- `GAP_THRESHOLD`: Max time gap for interpolation (default 0.2s)

---

### Step 4: Raw vs Synced Comparison (Optional)
Generate a comparison video showing both raw and interpolated poses:

```bash
python Scripts/compare_raw_synced.py
```

**What it does:**
- Shows **raw pose** (purple) - nearest motion timestamp, no interpolation
- Shows **synced pose** (yellow) - interpolated at exact video frame time
- Displays info panel with timestamps and metrics
- Demonstrates smoothness improvement from interpolation
- Outputs to `data/raw/test_002/comparison_output.mp4`

**Visualization:**
- **Purple circle**: Raw nearest pose (jumpy)
- **Yellow circle**: Synced interpolated pose (smooth)
- **Info panel**: Shows temporal offset and position difference
- **Legend**: Color coding explanation

**Use case:**
Use this to validate that interpolation is working correctly and to demonstrate quality improvement for presentations or debugging.

---

## Project Structure

```
VideoSync/
├── config/
│   ├── config.json              # General configuration (legacy)
│   └── calibration.json         # Projection calibration parameters
├── data/
│   └── raw/
│       └── test_002/
│           ├── MR_View.mp4      # Input video
│           ├── *.json           # Motion data files
│           ├── cropped/         # Cropped video clips
│           └── viz_output.mp4   # Visualization output
├── docs/
│   ├── project_guidelines.md
│   └── video_cropping_report.md
├── Scripts/
│   ├── run_video_cropping.py   # Step 1: Video cropping
│   ├── calibrate_projection.py # Step 2: Interactive calibration
│   └── run_visualization.py    # Step 3: Generate visualization
├── src/
│   ├── video_loader.py          # Video I/O and metadata
│   ├── motion_loader.py         # Motion data loading
│   ├── motion_matcher.py        # Timestamp alignment
│   ├── interpolator.py          # Pose interpolation (Slerp)
│   ├── visualizer.py            # 3D-to-2D projection
│   └── video_cropper.py         # Video splitting
├── tests/
│   └── ...                      # Unit tests
└── requirements.txt
```

## Troubleshooting

### Issue: Gizmo not visible
- Check that motion data timestamps overlap with video timestamps
- Verify `calibrated_start` value in scripts
- Ensure video and JSON files are loaded correctly

### Issue: Gizmo drifts from hand
- Run calibration tool (`calibrate_projection.py`)
- Adjust position/rotation offsets
- Save calibration with SPACE

### Issue: Gizmo appears between recording sessions
- Increase `GAP_THRESHOLD` in `run_visualization.py`
- Default is 0.2 seconds (200ms)

### Issue: Video cropping produces incorrect clips
- Check video start timestamp calibration
- Verify motion JSON timestamps are absolute Unix time
- Review `docs/video_cropping_report.md` for details

## Advanced Usage

### Custom Dataset
To use your own data:

1. Place video in `data/raw/your_dataset/video.mp4`
2. Place motion JSONs in `data/raw/your_dataset/*.json`
3. Update paths in scripts:
   - `run_video_cropping.py`
   - `calibrate_projection.py`
   - `run_visualization.py`
4. Run through the pipeline as described above

### Automated Pipeline (Coming Soon)
A full orchestrator script is planned to automate the entire workflow with a single command.
