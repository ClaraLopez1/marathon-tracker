# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Marathon Tracker** is a computer vision system that automatically registers marathon runners at the finish line. It:
- Accepts a video of the finish line crossing
- Detects and tracks individual runners using YOLOv8
- Allows interactive definition of an ROI (region of interest) and finish line
- Records the exact timestamp and position when each runner crosses the line
- Outputs a table with bib numbers, crossing timestamps, and finish positions

**Context:** TP Final (Final Project) for Computer Vision · Universidad Austral (2026), by Clara Lopez & Santos Bogo.

## Architecture

### Core Flow
1. **main.py** — Central video processing loop that orchestrates the entire pipeline
   - Loads video and displays first frame
   - Calls `select_roi()` and `select_finish_line()` for interactive setup
   - Iterates through frames, detects persons, checks for line crossings
   - Tracks unique runners and logs crossings with timestamps

2. **person_detector.py** — YOLOv8-based person detection with built-in tracking
   - Uses `ultralytics.YOLO("yolov8n.pt")` (nano model for speed)
   - Performs `.track()` with `classes=[0]` (person class only)
   - Filters by confidence threshold (0.4) and ROI membership
   - Returns tracking IDs alongside bounding boxes

3. **roi_selector.py** — Interactive drag-to-define Region of Interest
   - User drags a rectangle to mark the area where runners will pass
   - Global state: `roi_points`, `drawing` (for the drag operation)
   - Returns `(x1, y1, x2, y2)` normalized tuple

4. **finish_line.py** — Interactive 2-point line definition for the finish line
   - User clicks two points to define the crossing boundary
   - Used in `main.py`'s `crosses_line()` function to detect crossings

### Key Algorithms
- **Line crossing detection** (`main.py:crosses_line()`): Geometric check if the runner's foot (bottom center of bounding box) crosses the defined finish line
- **Tracking state**: `crossed_ids` set prevents duplicate position logging; `position` counter increments per unique crossing

### Dependencies
- **OpenCV** (`cv2`): Video I/O, drawing, mouse event handling
- **YOLOv8** (`ultralytics`): Person detection and tracking
- **Python 3.8+**

### Data
- Input: `data/video.mp4` (relative path from repo root)
- No persistent output file currently; results are printed to stdout

## Common Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install opencv-python ultralytics
```

### Run the Application
```bash
python src/main.py
```
- First frame opens for ROI selection (drag to define)
- Press Enter to confirm, Esc to cancel
- Second dialog opens for finish line (click 2 points, then Enter)
- Video plays with overlay; press 'q' to quit

### Development Notes
- **Model downloads**: First run downloads `yolov8n.pt` (~6 MB) to `~/.yolo/`
- **Interactive selectors**: Use global state to track mouse events; consider refactoring to classes if expanding
- **Frame iteration**: Uses `cap.get(cv2.CAP_PROP_FPS)` to map frame count to elapsed time
- **Tracking persistence**: YOLOv8's tracking is stateful across frames; set `persist=True` to maintain IDs

## Project Files Structure
```
marathon-tracker/
├── src/
│   ├── main.py                 # Video loop, line crossing logic
│   ├── person_detector.py      # YOLOv8 detection wrapper
│   ├── roi_selector.py         # Interactive ROI definition
│   └── finish_line.py          # Interactive finish line definition
├── data/                       # (Not in repo; add your own video.mp4)
├── README.md                   # Spanish-language project description
├── requirements.txt            # (Currently empty; update as needed)
└── CLAUDE.md                   # This file
```

## Known Limitations & TODOs
- **OCR not implemented**: README mentions Tesseract OCR for bib detection, but current code uses generic YOLO person tracking (no bib reading)
- **No output persistence**: Results printed to stdout only; consider CSV export
- **Interactive setup required**: ROI and finish line must be selected each run; could cache them
- **Confidence threshold hardcoded**: `0.4` is a magic number in `person_detector.py`

## Testing & Debugging
- To test without video: mock `cv2.VideoCapture` or create a small test video
- For tracking issues: print `r.boxes.id` values in `person_detector.py` to verify persistence
- For line crossing issues: enable visualization of the geometric calculation in `crosses_line()`
