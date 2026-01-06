# Data Structure Guide

This document explains where to place your video files and motion data to use the processing scripts.

## Directory Structure

When you copy the `Post-Processing/Video` folder to your repository, create the following structure:

```
YourRepository/
├── Post-Processing/
│   └── Video/                    # Copied folder
│       ├── Scripts/
│       ├── src/
│       ├── config/
│       └── ...
└── data/                         # Create this at repository root
    ├── raw/                      # Input data
    │   ├── test_002/            # Example dataset
    │   │   ├── MR_View.mp4       # Video file
    │   │   ├── T_Video_20251223_120927.json  # Motion data
    │   │   ├── T_Video_20251223_120932.json  # Motion data
    │   │   └── ...               # Additional motion JSON files
    │   └── your_dataset/         # Your own dataset
    │       ├── your_video.mp4
    │       └── *.json            # Motion data files
    └── synced/                   # Output data (created automatically)
        ├── test_002/
        │   ├── synced_poses.json
        │   ├── viz_gizmo.mp4
        │   └── processing_report.md
        └── your_dataset/
            └── ...
```

## Data Placement

### 1. Video Files
Place your video file(s) in:
```
data/raw/<dataset_name>/<video_file>.mp4
```

**Example:**
- `data/raw/test_002/MR_View.mp4`
- `data/raw/Utensil/Utensil_Retake_001.mp4`

### 2. Motion Data (JSON Files)
Place your motion tracking JSON files in the **same directory** as the video:
```
data/raw/<dataset_name>/
├── <video_file>.mp4
├── T_Video_20251223_120927.json      # Motion session 1
├── T_Video_20251223_120932.json      # Motion session 2
└── ...                                # Additional motion files
```

**Important:** 
- Motion JSON files should be in the same directory as the video
- The scripts will automatically find all `.json` files (excluding `*_metadata.json` and `*_validation.json`)
- Motion files should contain trajectory data with timestamps and poses

### 3. Output Files
Output files are automatically created in:
```
data/synced/<dataset_name>/
├── synced_poses.json              # Synchronized pose data
├── viz_gizmo.mp4                  # Visualization video
└── processing_report.md           # Processing summary
```

## Configuration

### Using Pipeline Config (Recommended)
Edit `config/pipeline_config.json` to point to your data:

```json
{
    "video_path": "data/raw/your_dataset/your_video.mp4",
    "motion_dir": "data/raw/your_dataset",
    "output_dir": "data/synced/your_dataset",
    "calibration_path": "config/calibration.json",
    "timestamp_offset": 1766488163.738,
    "options": {
        "generate_visualization": true,
        "visualization_type": "gizmo",
        "gap_threshold": 0.2,
        "export_synced_json": true,
        "generate_report": true
    }
}
```

### Using Individual Scripts
If using individual scripts (e.g., `run_visualization.py`), edit the paths directly in the script:

```python
data_root = PROJECT_ROOT / "data" / "raw" / "your_dataset"
video_path = data_root / "your_video.mp4"
json_dir = data_root
```

## Quick Start Example

1. **Create directory structure:**
   ```bash
   mkdir -p data/raw/my_dataset
   ```

2. **Copy your files:**
   ```bash
   cp your_video.mp4 data/raw/my_dataset/
   cp *.json data/raw/my_dataset/
   ```

3. **Update config:**
   Edit `config/pipeline_config.json` to use `my_dataset`

4. **Run pipeline:**
   ```bash
   cd Post-Processing/Video
   python Scripts/run_full_pipeline.py
   ```

## Notes

- All paths in config files are **relative to the repository root** (where `data/` folder is located)
- The `Post-Processing/Video` folder can be placed anywhere in your repository
- Scripts automatically resolve paths using `PROJECT_ROOT` which is calculated from the script location
- If your repository structure is different, adjust paths in config files accordingly

