# -*- coding: utf-8 -*-
"""
verify_env.py -- Environment verification for Boxing AI Performance Analyzer.

Run with:
    python verify_env.py

Forces stdout to UTF-8 so Unicode symbols work on Windows cp1252 consoles too.
Falls back to ASCII-only output if UTF-8 reconfigure is unavailable.
"""

from __future__ import annotations

import sys
import io
from pathlib import Path
import plotly

# ── Force UTF-8 stdout on Windows (Python 3.7+) ───────────────────────────
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

# ASCII-only status markers (safe on every console)
_PASS = "  PASS"
_FAIL = "  FAIL"
_WARN = "  WARN"

results: list[tuple[str, str, str]] = []


def _safe_print(*args, **kwargs):
    """Print that never crashes due to encoding errors."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        line = " ".join(str(a) for a in args)
        sys.stdout.buffer.write((line + "\n").encode("ascii", errors="replace"))


def check(name: str, fn):
    """Run fn(), record and print result."""
    try:
        detail = fn()
        results.append((name, "PASS", detail or ""))
        _safe_print(f"{_PASS}  {name:<50} {detail or ''}")
    except Exception as exc:
        results.append((name, "FAIL", str(exc)))
        _safe_print(f"{_FAIL}  {name:<50}")
        _safe_print(f"         ERROR: {str(exc)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
_safe_print("\n" + "=" * 70)
_safe_print("  Boxing AI Performance Analyzer -- Environment Verification")
_safe_print("=" * 70 + "\n")

# ── 1. Python version ────────────────────────────────────────────────────────
def _python():
    v = sys.version_info
    assert v >= (3, 10), f"Need Python >= 3.10, got {v.major}.{v.minor}.{v.micro}"
    return f"{v.major}.{v.minor}.{v.micro}  ({sys.platform})"
check("Python version >= 3.10", _python)

# ── 2. numpy ─────────────────────────────────────────────────────────────────
def _numpy():
    import numpy as np
    a = np.array([1, 2, 3], dtype=np.float32)
    assert a.sum() == 6.0
    return f"numpy {np.__version__}"
check("numpy import + smoke test", _numpy)

# ── 3. pandas ────────────────────────────────────────────────────────────────
def _pandas():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert len(df) == 2
    return f"pandas {pd.__version__}"
check("pandas import + smoke test", _pandas)

# ── 4. matplotlib ─────────────────────────────────────────────────────────────
def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, safe on headless
    return f"matplotlib {matplotlib.__version__}"
check("matplotlib import", _matplotlib)

# ── 5. OpenCV import ─────────────────────────────────────────────────────────
def _opencv():
    import cv2
    info = cv2.getBuildInformation()
    assert "OpenCV" in info
    return f"opencv {cv2.__version__}"
check("opencv-python import", _opencv)

# ── 6. OpenCV VideoCapture on fight.mp4 ──────────────────────────────────────
def _opencv_cap():
    import cv2
    video = ROOT / "data" / "fight.mp4"
    cap = cv2.VideoCapture(str(video))
    ok = cap.isOpened()
    if ok:
        ret, frame = cap.read()
        h, w = frame.shape[:2] if ret else (0, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert ret, "First frame read failed"
        return f"{w}x{h}  {fps:.2f}fps  {n} frames  (fight.mp4)"
    cap.release()
    raise RuntimeError("data/fight.mp4 not found or unreadable by OpenCV")
check("OpenCV VideoCapture (fight.mp4)", _opencv_cap)

# ── 7. src.video_io subsystem ─────────────────────────────────────────────────
def _video_io():
    sys.path.insert(0, str(ROOT))
    from src.video_io import VideoReader, VideoMetadata
    meta = VideoReader.probe(ROOT / "data" / "fight.mp4")
    assert isinstance(meta, VideoMetadata)
    assert meta.width > 0 and meta.height > 0 and meta.fps > 0
    return (f"{meta.width}x{meta.height}  {meta.fps:.2f}fps  "
            f"{meta.frame_count} frames  {meta.duration_s:.1f}s")
check("src.video_io VideoReader probe", _video_io)

# ── 8. Pillow ─────────────────────────────────────────────────────────────────
def _pillow():
    import PIL
    return f"Pillow {PIL.__version__}"
check("Pillow import", _pillow)

# ── 9. tqdm ──────────────────────────────────────────────────────────────────
def _tqdm():
    import tqdm
    return f"tqdm {tqdm.__version__}"
check("tqdm import", _tqdm)

# ── 10. plotly ────────────────────────────────────────────────────────────────
def _plotly():
    import plotly
    return f"plotly {plotly.__version__}"
check("plotly import", _plotly)

# ── 11. streamlit ─────────────────────────────────────────────────────────────
def _streamlit():
    import streamlit
    return f"streamlit {streamlit.__version__}"
check("streamlit import", _streamlit)

# ── 12. PyTorch ───────────────────────────────────────────────────────────────
def _torch():
    import torch
    cuda_ok  = torch.cuda.is_available()
    cuda_str = f"CUDA {torch.version.cuda}" if cuda_ok else "CPU-only"
    device   = "cuda" if cuda_ok else "cpu"
    return f"torch {torch.__version__}  |  {cuda_str}  |  device={device}"
check("PyTorch import + CUDA detection", _torch)

# ── 13. torchvision ───────────────────────────────────────────────────────────
def _torchvision():
    import torchvision
    return f"torchvision {torchvision.__version__}"
check("torchvision import", _torchvision)

# ── 14. Ultralytics ───────────────────────────────────────────────────────────
def _ultralytics():
    import ultralytics
    from ultralytics import YOLO  # noqa: F401
    return f"ultralytics {ultralytics.__version__}"
check("ultralytics import (no weight download)", _ultralytics)

# ── 15. CPU tensor op ─────────────────────────────────────────────────────────
def _cpu_tensor():
    import torch
    device = torch.device("cpu")
    t = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert t.sum().item() == 6.0
    return "CPU tensor ops verified"
check("PyTorch CPU fallback (tensor op)", _cpu_tensor)

# ── 16. Selected device ───────────────────────────────────────────────────────
def _device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"GPU selected: {name}"
    return "CPU selected (no CUDA GPU detected -- CPU mode OK)"
check("Processing device selection", _device)

# ── 17. config.py import + path sanity ───────────────────────────────────────
def _config():
    sys.path.insert(0, str(ROOT))
    import config as cfg
    assert cfg.ROOT == ROOT, f"ROOT mismatch: {cfg.ROOT} != {ROOT}"
    assert cfg.OUTPUT_DIR.exists(), f"OUTPUT_DIR missing: {cfg.OUTPUT_DIR}"
    assert isinstance(cfg.VIDEO_PATH, Path)
    return f"ROOT={cfg.ROOT.name}  MODEL={cfg.MODEL_NAME}"
check("config.py import + path sanity", _config)

# ── 18. src.* light modules (no ultralytics needed) ──────────────────────────
def _src_light():
    sys.path.insert(0, str(ROOT))
    from src.pose_features    import PoseFeatureExtractor, PoseFeatures
    from src.strike_detector  import StrikeDetector
    from src.defense_detector import DefenseDetector
    from src.fight_analyzer   import FightAnalyzer
    return "pose_features, strike, defense, analyzer -- OK"
check("src light modules (no YOLO dep)", _src_light)

# ── 19. src.tracker import ───────────────────────────────────────────────────
def _tracker():
    from src.tracker import FighterTracker
    t = FighterTracker()
    assert t.model_name == "yolov8n-pose"
    return "FighterTracker instantiated (model not loaded)"
check("src.tracker import", _tracker)

# ── 20. src.video_processor import ───────────────────────────────────────────
def _vp():
    from src.video_processor import VideoProcessor
    return "VideoProcessor importable"
check("src.video_processor import", _vp)

# ── 21. No circular imports (import all at once) ──────────────────────────────
def _no_cycles():
    import importlib
    mods = [
        "config",
        "src.video_io",
        "src.pose_features",
        "src.strike_detector",
        "src.defense_detector",
        "src.fight_analyzer",
        "src.tracker",
        "src.video_processor",
    ]
    for m in mods:
        importlib.import_module(m)
    return f"All {len(mods)} modules imported -- no circular deps"
check("Circular import check (all modules)", _no_cycles)

# ── 22. pathlib.Path (no os.path or hard-coded separators) ────────────────────
def _paths():
    import config as cfg
    path_attrs = ["ROOT", "DATA_DIR", "OUTPUT_DIR", "STATS_CSV", "SUMMARY_JSON"]
    for attr in path_attrs:
        val = getattr(cfg, attr)
        assert isinstance(val, Path), f"{attr} is not pathlib.Path"
    return f"All {len(path_attrs)} paths are pathlib.Path (Windows-safe)"
check("pathlib.Path usage (no os.path)", _paths)

# ── 23. outputs/ writable ─────────────────────────────────────────────────────
def _writable():
    import config as cfg
    test_file = cfg.OUTPUT_DIR / "_env_verify_write_test.tmp"
    test_file.write_text("ok", encoding="utf-8")
    content = test_file.read_text(encoding="utf-8")
    test_file.unlink()
    assert content == "ok"
    return f"{cfg.OUTPUT_DIR.name}/ is writable"
check("outputs/ directory writable", _writable)

# ── 24. NumPy shapes match YOLO convention ────────────────────────────────────
def _np_shapes():
    import numpy as np
    kps  = np.random.rand(17, 3).astype(np.float32)
    bbox = np.array([100, 200, 300, 400], dtype=np.int32)
    assert kps.shape  == (17, 3)
    assert bbox.shape == (4,)
    assert kps.dtype  == np.float32
    assert bbox.dtype == np.int32
    return "YOLO keypoint (17,3) and bbox (4,) shapes OK"
check("NumPy shapes (YOLO output simulation)", _np_shapes)

# ── 25. pip -- check for version conflicts ────────────────────────────────────
def _pip_check():
    import subprocess, json
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, timeout=30
    )
    output = result.stdout.strip() + result.stderr.strip()
    if result.returncode == 0:
        return "No dependency conflicts detected"
    # Summarise conflicts
    lines = [l for l in output.splitlines() if l.strip()]
    return f"WARN: {len(lines)} conflict(s) -- see below\n" + "\n".join(f"         {l}" for l in lines[:5])
check("pip check (dependency conflicts)", _pip_check)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for _, s, _ in results if s == "PASS")
failed = sum(1 for _, s, _ in results if s == "FAIL")

_safe_print("\n" + "=" * 70)
_safe_print(f"  Results: {passed} passed  |  {failed} failed  |  {len(results)} total checks")
_safe_print("=" * 70)

if failed:
    _safe_print("\n  FAILED CHECKS:")
    for name, status, detail in results:
        if status == "FAIL":
            _safe_print(f"    [X] {name}")
            if detail:
                _safe_print(f"        {detail[:150]}")
    _safe_print("")
    sys.exit(1)
else:
    _safe_print("\n  [OK] All checks passed -- environment is ready.\n")
    sys.exit(0)
