# Boxing AI Performance Analyzer

> A local Python application that analyzes boxing/fight videos using computer vision and AI — no cloud required.

---

## Features

| Feature | Description |
|---|---|
| **Fighter Detection** | Detects fighters in every frame using YOLOv8-Pose |
| **Fighter Tracking** | Assigns consistent IDs across frames with BoT-SORT |
| **Pose Estimation** | Extracts 17-keypoint skeletons for each fighter |
| **Strike Detection** | Heuristic punch classifier (jab, cross, hook, uppercut) |
| **Defense Detection** | Detects guards, slips, ducks, and clinches |
| **Movement Analysis** | Footwork patterns, ring generalship, aggression index |
| **Fight Statistics** | Per-fighter stats: punch count, accuracy, work rate |
| **Annotated Video** | Output MP4 with bounding boxes, skeletons, action labels |
| **CSV / JSON Export** | Machine-readable stats for further analysis |
| **Streamlit Dashboard** | Interactive visual report with charts and frame replay |

---

## Project Structure

```
boxing-ai-performance-analyzer/
├── data/                   # Place your input video(s) here
│   └── fight.mp4
├── outputs/                # Generated analysis files land here
├── src/                    # Core analysis modules
│   ├── __init__.py
│   ├── detector.py         # YOLO Pose wrapper
│   ├── tracker.py          # BoT-SORT integration
│   ├── pose_features.py    # Keypoint → feature extraction
│   ├── strike_detector.py  # Rule-based punch classifier
│   ├── defense_detector.py # Guard / evasion classifier
│   ├── fight_analyzer.py   # Frame aggregation & statistics
│   └── video_processor.py  # Orchestrates full pipeline
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── tests/                  # Unit tests
├── config.py               # Central configuration (paths, thresholds)
├── main.py                 # CLI entry-point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your video

Place a fight video in the `data/` folder and name it `fight.mp4`, or pass the path explicitly.

### 3. Run the analysis pipeline

```bash
python main.py
# or with a custom video path:
python main.py --video data/your_video.mp4
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Configuration

Edit [`config.py`](config.py) to adjust:

- `VIDEO_PATH` — default input video
- `MODEL_NAME` — YOLO pose model variant (`yolov8n-pose`, `yolov8m-pose`, etc.)
- `CONFIDENCE_THRESHOLD` — detection confidence
- `IOU_THRESHOLD` — NMS overlap threshold
- `TRACKER` — tracker algorithm (`botsort.yaml` or `bytetrack.yaml`)
- `OUTPUT_DIR` — where results are written
- Punch / defense / movement thresholds

---

## Output Files

After a successful run, `outputs/` contains:

| File | Description |
|---|---|
| `annotated_<video>.mp4` | Video with overlaid skeletons and labels |
| `fight_stats.csv` | Per-frame statistics |
| `fight_summary.json` | Aggregated fight summary |

---

## Technology Stack

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** — YOLOv8 Pose + BoT-SORT
- **[OpenCV](https://opencv.org/)** — Video I/O and frame annotation
- **[NumPy](https://numpy.org/)** / **[Pandas](https://pandas.pydata.org/)** — Numerical processing & stats
- **[Streamlit](https://streamlit.io/)** — Dashboard UI
- **[Plotly](https://plotly.com/)** — Interactive charts

---

## Requirements

- Python 3.10+
- Windows / macOS / Linux
- CPU inference works; GPU (CUDA) strongly recommended for real-time speed

---

## License

MIT
