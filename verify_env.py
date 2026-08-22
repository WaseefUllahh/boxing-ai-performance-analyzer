"""
verify_env.py — Environment verification script for Boxing AI Performance Analyzer.

Run with:
    python verify_env.py

Checks:
    1. Python version (must be 3.10+)
    2. All required packages importable
    3. Package versions reported
    4. NumPy / Pandas basic smoke test
    5. OpenCV video-read capability
    6. Ultralytics YOLO importable + model download skipped (just import check)
    7. PyTorch version + CUDA availability → selected device reported
    8. Streamlit importable
    9. Plotly importable
    10. Project-internal imports (config, src.*) — circular import check
    11. pathlib path integrity check (Windows-safe)
    12. outputs/ directory writable
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"
INFO = "\033[94m  INFO\033[0m"

results: list[tuple[str, str, str]] = []   # (check_name, status, detail)

def check(name: str, fn):
    """Run fn(), record result."""
    try:
        detail = fn()
        results.append((name, "PASS", detail or ""))
        print(f"{PASS}  {name:<45} {detail or ''}")
    except Exception as exc:
        results.append((name, "FAIL", str(exc)))
        print(f"{FAIL}  {name:<45} {exc}")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Boxing AI Performance Analyzer — Environment Verification")
print("=" * 70 + "\n")

# 1. Python version
def _python():
    v = sys.version_info
    assert v >= (3, 10), f"Need Python ≥ 3.10, got {v.major}.{v.minor}.{v.micro}"
    return f"{v.major}.{v.minor}.{v.micro}  ({sys.platform})"
check("Python version ≥ 3.10", _python)

# 2. numpy
def _numpy():
    import numpy as np
    a = np.array([1, 2, 3], dtype=np.float32)
    assert a.sum() == 6.0
    return f"numpy {np.__version__}"
check("numpy import + smoke test", _numpy)

# 3. pandas
def _pandas():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert len(df) == 2
    return f"pandas {pd.__version__}"
check("pandas import + smoke test", _pandas)

# 4. OpenCV
def _opencv():
    import cv2
    # verify build info exists (catches broken installs)
    info = cv2.getBuildInformation()
    assert "OpenCV" in info
    return f"opencv {cv2.__version__}"
check("opencv-python import", _opencv)

# 5. OpenCV VideoCapture (file-level test)
def _opencv_cap():
    import cv2
    cap = cv2.VideoCapture(str(ROOT / "data" / "fight.mp4"))
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        cap.release()
        assert ret, "Read failed"
        h, w = frame.shape[:2]
        return f"fight.mp4 readable — {w}x{h}"
    else:
        cap.release()
        raise RuntimeError("data/fight.mp4 not found or unreadable")
check("OpenCV VideoCapture (fight.mp4)", _opencv_cap)

# 6. Pillow
def _pillow():
    from PIL import Image
    import PIL
    return f"Pillow {PIL.__version__}"
check("Pillow import", _pillow)

# 7. tqdm
def _tqdm():
    import tqdm
    return f"tqdm {tqdm.__version__}"
check("tqdm import", _tqdm)

# 8. plotly
def _plotly():
    import plotly
    return f"plotly {plotly.__version__}"
check("plotly import", _plotly)

# 9. streamlit
def _streamlit():
    import streamlit
    return f"streamlit {streamlit.__version__}"
check("streamlit import", _streamlit)

# 10. PyTorch
def _torch():
    import torch
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if cuda_ok else "cpu"
    cuda_str = f"CUDA {torch.version.cuda}" if cuda_ok else "CPU-only"
    return f"torch {torch.__version__}  |  {cuda_str}  |  device={device}"
check("PyTorch import + device detection", _torch)

# 11. Ultralytics
def _ultralytics():
    import ultralytics
    # Just verify the package is importable — do NOT download weights here
    from ultralytics import YOLO  # noqa: F401
    return f"ultralytics {ultralytics.__version__}"
check("ultralytics import (no weight download)", _ultralytics)

# 12. Ultralytics + torch device selection
def _device_selection():
    import torch
    import ultralytics.utils.torch_utils as tu
    device = tu.select_device("")   # "" = auto
    return f"auto-selected device: {device}"
check("Ultralytics device auto-selection", _device_selection)

# 13. Project config import
def _config():
    sys.path.insert(0, str(ROOT))
    import config as cfg
    assert cfg.ROOT == ROOT
    assert cfg.OUTPUT_DIR.exists(), f"OUTPUT_DIR missing: {cfg.OUTPUT_DIR}"
    return f"ROOT={cfg.ROOT.name}  MODEL={cfg.MODEL_NAME}"
check("config.py import + path sanity", _config)

# 14. src package — circular import test
def _src_imports():
    sys.path.insert(0, str(ROOT))
    import src  # __init__.py
    from src.pose_features   import PoseFeatureExtractor
    from src.strike_detector import StrikeDetector
    from src.defense_detector import DefenseDetector
    from src.fight_analyzer  import FightAnalyzer
    # tracker + video_processor import ultralytics — test separately below
    return "pose_features, strike_detector, defense_detector, fight_analyzer OK"
check("src.* imports (light modules)", _src_imports)

# 15. src.tracker import (requires ultralytics)
def _tracker_import():
    from src.tracker import FighterTracker
    t = FighterTracker()
    assert t.model_name == "yolov8n-pose"
    return "FighterTracker instantiated (model not loaded yet)"
check("src.tracker import", _tracker_import)

# 16. src.video_processor import
def _vp_import():
    from src.video_processor import VideoProcessor
    return "VideoProcessor importable"
check("src.video_processor import", _vp_import)

# 17. pathlib path check (Windows-safe, no hard-coded slashes)
def _paths():
    import config as cfg
    paths = {
        "ROOT":        cfg.ROOT,
        "DATA_DIR":    cfg.DATA_DIR,
        "OUTPUT_DIR":  cfg.OUTPUT_DIR,
        "STATS_CSV":   cfg.STATS_CSV,
        "SUMMARY_JSON":cfg.SUMMARY_JSON,
    }
    for name, p in paths.items():
        assert isinstance(p, Path), f"{name} is not a pathlib.Path"
    return "all paths are pathlib.Path objects"
check("pathlib.Path usage (no hard-coded strings)", _paths)

# 18. outputs/ writable
def _writable():
    import config as cfg
    test_file = cfg.OUTPUT_DIR / "_verify_write_test.tmp"
    test_file.write_text("ok")
    assert test_file.read_text() == "ok"
    test_file.unlink()
    return f"{cfg.OUTPUT_DIR} is writable"
check("outputs/ directory writable", _writable)

# 19. NumPy dtype compatibility with YOLO output shapes
def _numpy_shapes():
    import numpy as np
    # Simulate a (17,3) keypoint array as YOLO returns
    kps = np.random.rand(17, 3).astype(np.float32)
    assert kps.shape == (17, 3)
    # Simulate bbox
    bbox = np.array([100, 200, 300, 400], dtype=np.int32)
    assert bbox.tolist() == [100, 200, 300, 400]
    return "dtype shapes OK"
check("NumPy dtype shapes (YOLO simulation)", _numpy_shapes)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")

print("\n" + "=" * 70)
print(f"  Results: {passed} passed  |  {failed} failed  |  {len(results)} total")
print("=" * 70)

if failed:
    print("\n  FAILED CHECKS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f"    ✗ {name}")
            print(f"      {detail}")
    sys.exit(1)
else:
    print("\n  All checks passed — environment is ready.\n")
    sys.exit(0)
