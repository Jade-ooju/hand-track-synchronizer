# Project Guidelines and Context

## Development Environment
- **Operating System**: Windows
- **Shell**: PowerShell
- **Virtual Environment**: Use `venv`.
    - Activation: `.\venv\Scripts\activate`
    - Installation: `pip install -r requirements.txt`
- **Command Chaining**: Use `;` to chain commands in PowerShell (e.g., `git add .; git commit ...`), not `&&`.

## Project Context
This project serves as the **post-processing pipeline** for the "Unity VR to ManiSkill3 Data Collection" workflow.

### High-Level Workflow
1.  **Data Collection (External)**: Users collect data using a VR setup (Unity -> ManiSkill3). This produces:
    - Egocentric video recordings (MP4).
    - 3D hand pose motion logs (JSON).
2.  **Synchronization (This Project)**:
    - This pipeline intakes the raw video and motion logs.
    - It aligns the 3D pose data to the video timestamps.
    - It handles calibration (3D to 2D projection) and timestamp offsets.
3.  **Output**:
    - A dataset of frame-accurate synchronized poses and video frames, ready for training.

### Key References
- **Master Doc**: "Unity VR → ManiSkill3 Data Collection & Retargeting (1).pdf" (High-level pipeline description).
