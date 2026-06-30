"""
detect.py — ClearRoute Phase 2
Runs YOLO11n on a video, georeferences each detection, and writes dados_reais.json.
Run:  python detect.py
"""

import cv2
import json
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ── Input video ───────────────────────────────────────────────────────────────
# Change this to the path of your video file before running.
<<<<<<< Updated upstream
VIDEO_PATH = r"C:\Users\julir\OneDrive\Escritorio\Hackathlon\clearroute\videos\example4.mp4"
=======
VIDEO_PATH = r"C:\Users\AM\Desktop\VSCODe\clearroute\videos\example5.mp4"
>>>>>>> Stashed changes

# ── Output file ───────────────────────────────────────────────────────────────
# Each run creates a new file with a timestamp so previous results are never overwritten.
# Example: dados/dados_reais_20250626_143022.json
_timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
_video_stem = Path(VIDEO_PATH).stem  # e.g. "example2"
OUTPUT_PATH = os.path.join("data", f"dados_reais_{_timestamp}.json")

# ── Frames folder (one per run) ───────────────────────────────────────────────
# Annotated frames are named only by timestamp (frame_00m12s.jpg), so saving every
# run into a shared frames/ folder would overwrite frames from previous videos at
# the same second. Each run gets its own subfolder, keyed by video name + timestamp,
# so a JSON's frame_path always points to that video's frame — never another's.
FRAMES_DIR = os.path.join("frames", f"{_video_stem}_{_timestamp}")

# ── Route segments ────────────────────────────────────────────────────────────
# Each segment covers a time range (in seconds) along the collection route.
# The GPS coordinates are fixed per street — they'll be replaced by real GPS
# data once the vehicle has a live feed.
ROTA = [
    {"inicio": 0,   "fim": 60,  "lat": 47.6762, "lon": 9.1691, "rua": "Marktstätte"},
    {"inicio": 60,  "fim": 120, "lat": 47.6762, "lon": 9.1691, "rua": "Hussenstraße"},
    {"inicio": 120, "fim": 180, "lat": 47.6762, "lon": 9.1691, "rua": "Seeufer"},
]

# ── Detection settings ────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.25   # detections below this score are ignored
FPS_EXTRACTION = 3      # frames extracted per second of video

# ── Class filter (Trash-AI v2 — best.pt) ─────────────────────────────────────
# Key: class name as best.pt reports it → Value: German label for the JSON.
LITTER_CLASSES = {
    "Bottle-Glass": "Flasche",
    "Can-Metal":    "Dose",
    "Cardboard":    "Karton",
    "Cup":          "Becher",
    "Mask":         "Maske",
    "Needle":       "Nadel",
    "Paper":        "Papier",
    "Plastic":      "Plastik",
    "Trash -everything else-": "Müll",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def format_timestamp(total_seconds):
    """Converts an integer number of seconds to 'MM:SS' string."""
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def get_coordinates(total_seconds):
    """
    Returns (lat, lon) for the route segment that contains `total_seconds`.
    Falls back to the last segment if the video is longer than the route.
    """
    for segment in ROTA:
        if segment["inicio"] <= total_seconds < segment["fim"]:
            return segment["lat"], segment["lon"]
    # Beyond the defined route — use the last known position
    return ROTA[-1]["lat"], ROTA[-1]["lon"]


# ── Deduplication ────────────────────────────────────────────────────────────
def deduplicate(detections, time_window=3):
    """
    Removes detections of the same type that are within `time_window` seconds
    of each other — these are the same physical object seen across consecutive frames.
    Sorts by confidence descending first so the highest-confidence sighting is kept.
    """
    def ts_to_sec(ts):
        m, s = ts.split(":")
        return int(m) * 60 + int(s)

    # Highest confidence first — the first occurrence of each object wins
    sorted_dets = sorted(detections, key=lambda d: d["konfidenz"], reverse=True)
    kept = []

    for det in sorted_dets:
        det_sec = ts_to_sec(det["zeitstempel"])
        already_kept = any(
            d["typ"] == det["typ"]
            and abs(ts_to_sec(d["zeitstempel"]) - det_sec) <= time_window
            for d in kept
        )
        if not already_kept:
            kept.append(det)

    return kept


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path, frames_dir):
    """
    Reads the video 1 frame per second, runs YOLO on each frame,
    and returns a list of detection dicts ready for JSON export.
    Annotated frames are written to `frames_dir` (unique per run).
    """
    # Load the YOLO11 nano model — downloads automatically on first run (~6 MB)
    model = YOLO("models/best2.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)

    step      = fps / FPS_EXTRACTION        # gap in frames between each sample
    n_samples = int(total_frames / step)    # total frames that will be analysed

    print(f"Video loaded: {total_frames} frames @ {fps:.1f} fps ({duration_sec}s total)")
    print(f"Extracting {FPS_EXTRACTION} frames/s → {n_samples} frames to analyse\n")

    # Create this run's frames folder once before the loop starts
    os.makedirs(frames_dir, exist_ok=True)

    detections = []
    frames_processed = 0

    for i in range(n_samples):
        frame_index = int(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        current_sec = int(frame_index / fps)
        timestamp   = format_timestamp(current_sec)
        lat, lon    = get_coordinates(current_sec)

        # Run YOLO inference — verbose=False keeps the terminal clean
        results = model(frame, verbose=False, conf=MIN_CONFIDENCE)

        # Collect every litter box found in this frame first
        frame_detections = []
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]

                if class_name not in LITTER_CLASSES:
                    continue

                frame_detections.append({
                    "lat":         lat,
                    "lon":         lon,
                    "typ":         LITTER_CLASSES[class_name],
                    "konfidenz":   round(float(box.conf), 2),
                    "zeitstempel": timestamp,
                })

        # If this frame has at least one litter detection, save the annotated image
        if frame_detections:
            # "00:07" → "frame_00m07s.jpg"
            fname      = f"frame_{timestamp.replace(':', 'm')}s.jpg"
            frame_path = os.path.join(frames_dir, fname)
            # results[0].plot() draws bounding boxes on the frame (returns BGR numpy array)
            cv2.imwrite(frame_path, results[0].plot())

            # Every detection from this frame points to the same saved image
            for det in frame_detections:
                det["frame_path"] = frame_path

        detections.extend(frame_detections)
        frames_processed += 1

        # Progress report every 10 frames so you know it's running
        if frames_processed % 10 == 0:
            print(f"Frame {frames_processed} processed, {len(detections)} detections so far.")

    cap.release()
    return detections


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found → {VIDEO_PATH}")
        print("Edit the VIDEO_PATH variable at the top of detect.py and try again.")
        raise SystemExit(1)

    print("=== ClearRoute — Detection Pipeline ===\n")
    print(f"Video file : {VIDEO_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Frames dir : {FRAMES_DIR}\n")

    results = process_video(VIDEO_PATH, FRAMES_DIR)

    before = len(results)
    results = deduplicate(results)
    print(f"Deduplication: {before} → {len(results)} detections kept.")

    # Save JSON in the same format as dados_exemplo.json
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(results)} detections saved to {OUTPUT_PATH}")
