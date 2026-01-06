import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "raw" / "test_002" / "T_Video_20251223_120927.json"
try:
    with open(file_path, 'r') as f:
        d = json.load(f)
    
    if 'trajectories' in d and len(d['trajectories']) > 0:
        t = d['trajectories'][0]
        print(f"Keys in trajectory: {list(t.keys())}")
        
        if 'timestamps' in t:
            print(f"Timestamp 0: {t['timestamps'][0]} type: {type(t['timestamps'][0])}")
            print(f"Timestamp count: {len(t['timestamps'])}")
            
        if 'poses' in t:
            pose = t['poses'][0]
            print(f"Pose 0: {pose}")
            print(f"Pose type: {type(pose)}")
            print(f"Pose length: {len(pose)}")
            
        if 'relative_time' in t:
             print(f"Relative time 0: {t['relative_time'][0]}")

    else:
        print("No trajectories found.")
        
except Exception as e:
    print(f"Error: {e}")
