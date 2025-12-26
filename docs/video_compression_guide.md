# Video Compression Guide for ML Training

This document outlines the recommended settings for compressing raw video recordings to minimize storage while maintaining quality for hand tracking and object recognition.

## Recommended Settings

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| **Codec** | `H.265 (HEVC)` | 50-70% better compression than H.264 at same quality. |
| **CRF** | `32` | Balanced quality vs size. ~85% storage reduction. |
| **Preset** | `veryfast` | Fast encoding without significant quality penalty. |
| **Resolution**| `1024x1024` (Native) | Keep native resolution for best keypoint accuracy. |

## Storage Impact

Based on benchmarks of `MR_View.mp4`:
- **Raw (H.264)**: ~37.6 MB/minute
- **Compressed (H.265, CRF 32)**: ~5.8 MB/minute (~85% reduction)
- **Compressed + Scaled (512x512)**: ~3.4 MB/minute (~91% reduction)

## Usage

### Batch Optimize Dataset
Use the `Scripts/optimize_dataset.py` script to compress all raw videos in a directory:
```powershell
python Scripts/optimize_dataset.py --dir data/raw --crf 32
```

### Main Pipeline Integration
The `Scripts/run_full_pipeline.py` automatically checks for a `_compressed.mp4` version of the input video. If found, it uses it by default to speed up processing and save space.

To disable this behavior, update the `pipeline_config.json`:
```json
"options": {
    "use_compressed": false
}
```

## Quality Verification
Visual inspection confirms that at CRF 32, hand poses and object boundaries remain sharp and detectable by ML models.
