import cv2
import os
import sys

def analyze_video(file_path):
    print(f"Analyzing: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return None

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Could not open video - {file_path}")
        return None

    # Get Metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    
    file_size_bytes = os.path.getsize(file_path)
    bitrate_mbps = (file_size_bytes * 8) / (duration_sec * 1024 * 1024) if duration_sec > 0 else 0
    
    # Check Codec (FourCC)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    return {
        "File": os.path.basename(file_path),
        "Resolution": f"{width}x{height}",
        "FPS": f"{fps:.2f}",
        "Codec": codec,
        "Duration (s)": f"{duration_sec:.2f}",
        "Size (MB)": f"{file_size_bytes / (1024*1024):.2f}",
        "Bitrate (Mbps)": f"{bitrate_mbps:.2f}"
    }

def main():
    base_dir = r"d:\OOJU\Projects\VideoSync\Resources\RecordComparison"
    files = ["MQDH_Wired.mp4", "PhoneApp_Wireless.mp4"]
    
    results = []
    
    for f in files:
        path = os.path.join(base_dir, f)
        data = analyze_video(path)
        if data:
            results.append(data)
    
    # Print Comparison Matrix
    print("\n--- Comparison Matrix ---")
    headers = ["File", "Resolution", "FPS", "Codec", "Duration (s)", "Size (MB)", "Bitrate (Mbps)"]
    
    # Simple table print
    header_str = " | ".join(headers)
    print(header_str)
    print("-" * len(header_str))
    
    markdown_table = f"| {' | '.join(headers)} |\n| {' | '.join(['---']*len(headers))} |\n"

    for r in results:
        row = [str(r.get(h, "N/A")) for h in headers]
        print(" | ".join(row))
        markdown_table += f"| {' | '.join(row)} |\n"

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "..", "video_comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# Video Capture Method Comparison\n\n")
        f.write(markdown_table)
    
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
