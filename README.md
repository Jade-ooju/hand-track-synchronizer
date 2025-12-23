# Hand Track Synchronizer

A pipeline for frame-accurate synchronization of egocentric video with 3D hand pose data.

## Description
This project aims to rebuild a video-motion synchronization system to align egocentric video frames with 3D hand pose motion data (JSON logs), creating frame-accurate synchronized datasets for training.

## Features
- Extract precise frame timestamps from video
- Match and interpolate 3D hand poses to video timelines
- Handle calibration challenges involving 3D pose projection and timestamp alignment
- Generate raw and synced pose visualizations

## Usage
(Instructions to be added)

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
