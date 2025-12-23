# Video Cropping & Alignment Report

## Objective
To split a single long MR capture video (`MR_View.mp4`) into individual clips corresponding to multiple discontinuous motion recording sessions (JSON logs), ensuring frame-accurate synchronization.

## Methodology

### 1. VFR Support
Initial attempts using frame-index based slicing (`frame_idx = time * fps`) caused drift due to the Variable Frame Rate (VFR) nature of the Meta Quest recording.
**Solution**: Refactored `VideoCropper` to use `CV_CAP_PROP_POS_MSEC`. The script scans the video and extracts frames based on their precise internal timestamps, ensuring alignment regardless of frame rate fluctuations.

### 2. Timestamp Alignment & Calibration
Standard file metadata (`FileModifiedTime`) proved imprecise for determining the exact "Video Start Time" relative to the motion logs (Unix Epoch).
- **Initial Estimate**: `ModifiedTime - Duration` -> yielded ~1.5s offset error.
- **Visual Calibration**: Manual inspection showed a 4.24s discrepancy (Content for Session 1 was found at the timestamp of Session 2).
- **Final Correction**: Applied a **-4.24s correction shift** to the start timestamp.
- **Calibrated Start Time**: `1766488163.738` (Unix Timestamp).

## Results
**Input**:
-   `data/raw/test_002/MR_View.mp4` (Duration: 28.74s)
-   4 Motion JSON files (`...927.json` to `...941.json`)

**Output**:
-   Location: `data/raw/test_002/cropped_recalib/`
-   Files:
    -   `T_Video_20251223_120927.mp4` (Duration: ~2.0s)
    -   `T_Video_20251223_120932.mp4` (Duration: ~1.7s)
    -   `T_Video_20251223_120937.mp4` (Duration: ~1.75s)
    -   `T_Video_20251223_120941.mp4` (Duration: ~1.6s)

## Usage
To reproduce the cropping with the calibrated timestamp:
```bash
python Scripts/crop_recalibrated.py
```
This script uses `src/video_cropper.py` with the explicit `start_timestamp_unix=1766488163.738`.
