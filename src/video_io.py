"""
src/video_io.py — Robust video input subsystem for Boxing AI Performance Analyzer.

Responsibilities
----------------
- Validate that a video file exists and is readable by OpenCV.
- Extract and validate video metadata (resolution, FPS, frame count, duration).
- Provide a safe, memory-efficient frame iterator (one frame at a time).
- Handle corrupted / dropped frames gracefully.
- Ensure the VideoCapture is always released (context manager).
- Fail fast with clear, actionable error messages.

Design principles
-----------------
- NEVER load the entire video into RAM — strict frame-by-frame streaming.
- Use pathlib.Path everywhere (Windows-safe, no hard-coded separators).
- Resource management via context manager (__enter__ / __exit__).
- All public errors are VideoIOError subclasses for easy catching.

Usage
-----
    from src.video_io import VideoReader, VideoMetadata

    # One-shot metadata inspection
    meta = VideoReader.probe(path)
    print(meta)

    # Frame iteration (context manager guarantees release)
    with VideoReader(path) as reader:
        for frame_idx, frame in reader.frames():
            process(frame)

    # Or use the class directly inside VideoProcessor
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class VideoIOError(RuntimeError):
    """Base class for all video I/O errors raised by this module."""


class VideoNotFoundError(VideoIOError):
    """Raised when the video file does not exist."""


class VideoOpenError(VideoIOError):
    """Raised when OpenCV cannot open the video file."""


class VideoMetadataError(VideoIOError):
    """Raised when video metadata is invalid (zero FPS, zero dimensions, etc.)."""


# ─────────────────────────────────────────────────────────────────────────────
# VideoMetadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VideoMetadata:
    """
    Immutable snapshot of a video file's properties.

    All fields are populated at open time from cv2.CAP_PROP_* queries.
    No pixel data is read to construct this object.
    """
    path:        Path
    width:       int
    height:      int
    fps:         float
    frame_count: int       # may be 0 for live streams or some containers
    duration_s:  float     # seconds; 0.0 if frame_count is unknown
    fourcc_str:  str       # four-character codec code, e.g. "mp4v"

    # ── Derived properties ────────────────────────────────────────────────
    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height)"""
        return (self.width, self.height)

    @property
    def duration_str(self) -> str:
        """Human-readable duration, e.g. '2:29.7'."""
        total = self.duration_s
        mins  = int(total // 60)
        secs  = total - mins * 60
        return f"{mins}:{secs:04.1f}"

    def __str__(self) -> str:
        lines = [
            "",
            "-" * 40,
            f"  Video     : {self.path.name}",
            f"  Path      : {self.path}",
            f"  Resolution: {self.width} x {self.height}",
            f"  FPS       : {self.fps:.3f}",
            f"  Frames    : {self.frame_count:,}",
            f"  Duration  : {self.duration_s:.1f} s  ({self.duration_str})",
            f"  Codec     : {self.fourcc_str}",
            "-" * 40,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VideoReader
# ─────────────────────────────────────────────────────────────────────────────

class VideoReader:
    """
    Safe, streaming video reader backed by cv2.VideoCapture.

    Memory model
    ------------
    Only ONE frame is held in RAM at a time.  The previous frame's ndarray
    is overwritten by the next call to cap.read().  Never accumulates frames.

    Resource management
    -------------------
    Use as a context manager:

        with VideoReader(path) as reader:
            for idx, frame in reader.frames():
                ...

    Or call open() / close() manually — close() is idempotent.

    Parameters
    ----------
    path : str | Path
        Path to the video file.
    max_consecutive_failures : int
        How many consecutive unreadable frames to tolerate before stopping.
        Useful for videos with sporadic corruption.  Default: 5.
    """

    def __init__(
        self,
        path: str | Path,
        max_consecutive_failures: int = 5,
    ) -> None:
        self.path = Path(path).resolve()
        self.max_consecutive_failures = max_consecutive_failures
        self._cap: Optional[cv2.VideoCapture] = None
        self._meta: Optional[VideoMetadata] = None

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False   # do not suppress exceptions

    # ── Open / close ──────────────────────────────────────────────────────

    def open(self) -> "VideoReader":
        """
        Open the video file, validate it, and read metadata.

        Raises
        ------
        VideoNotFoundError  — file does not exist
        VideoOpenError      — OpenCV cannot open the file
        VideoMetadataError  — metadata is invalid
        """
        # 1. File existence check
        if not self.path.exists():
            raise VideoNotFoundError(
                f"Video file not found: {self.path}\n"
                "Place a fight video in the data/ folder or pass --video <path>."
            )
        if not self.path.is_file():
            raise VideoNotFoundError(
                f"Path exists but is not a file: {self.path}"
            )

        # 2. OpenCV open — pass as str() because cv2 does not accept Path
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise VideoOpenError(
                f"OpenCV could not open: {self.path}\n"
                "Possible causes: unsupported codec, corrupted file, "
                "or missing FFmpeg backend."
            )

        # 3. Extract raw metadata
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str  = self._fourcc_to_str(fourcc_int)

        # 4. Validate metadata
        errors: list[str] = []
        if width <= 0:
            errors.append(f"Invalid frame width: {width}")
        if height <= 0:
            errors.append(f"Invalid frame height: {height}")
        if fps <= 0.0:
            errors.append(
                f"Invalid FPS: {fps}  "
                "(the container may not report FPS; try re-encoding with FFmpeg)"
            )
        if errors:
            cap.release()
            raise VideoMetadataError(
                f"Video metadata validation failed for {self.path.name}:\n"
                + "\n".join(f"  * {e}" for e in errors)
            )

        # 5. Derive duration (guard against containers with frame_count == 0)
        duration = (frame_count / fps) if frame_count > 0 else 0.0

        self._cap  = cap
        self._meta = VideoMetadata(
            path        = self.path,
            width       = width,
            height      = height,
            fps         = fps,
            frame_count = frame_count,
            duration_s  = duration,
            fourcc_str  = fourcc_str,
        )
        return self

    def close(self) -> None:
        """Release the VideoCapture.  Safe to call multiple times."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def meta(self) -> VideoMetadata:
        """Return video metadata.  Raises if not yet opened."""
        if self._meta is None:
            raise VideoIOError("VideoReader has not been opened yet. Call open() first.")
        return self._meta

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── Frame iterator ────────────────────────────────────────────────────

    def frames(
        self,
        start_frame: int = 0,
        end_frame:   Optional[int] = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_index, bgr_frame) one at a time.

        Memory model: only one frame is in RAM at any point.

        Parameters
        ----------
        start_frame : int
            0-based index of the first frame to yield.
        end_frame : int | None
            Exclusive upper bound.  None = read until end-of-video.

        Yields
        ------
        (frame_index, frame)
            frame_index : int — 0-based index from the start of the video.
            frame       : np.ndarray — BGR image, shape (H, W, 3), dtype uint8.

        Notes
        -----
        - Corrupted / unreadable frames are skipped with a warning.
        - If max_consecutive_failures consecutive frames fail, iteration stops.
        - The generator is safe to break out of early.
        """
        if not self.is_open:
            raise VideoIOError(
                "VideoReader is not open. Call open() or use as context manager."
            )

        # Seek to start frame if needed
        if start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

        frame_idx         = start_frame
        consecutive_fails = 0

        while True:
            # Check hard stop
            if end_frame is not None and frame_idx >= end_frame:
                break

            ret, frame = self._cap.read()

            if not ret:
                # Distinguish end-of-video from read failure
                current_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
                total       = self._meta.frame_count

                # Normal end-of-video
                if total > 0 and current_pos >= total:
                    break
                # Stream ended (live cam or container with unknown length)
                if not ret and frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= self.max_consecutive_failures:
                        print(
                            f"[VideoReader] {consecutive_fails} consecutive read "
                            f"failures at frame {frame_idx} — stopping iteration."
                        )
                        break
                    print(
                        f"[VideoReader] Warning: unreadable frame {frame_idx} "
                        f"(failure {consecutive_fails}/{self.max_consecutive_failures}) — skipping."
                    )
                    frame_idx += 1
                    continue

            # Validate frame dimensions match metadata
            if frame is None or frame.size == 0:
                consecutive_fails += 1
                print(f"[VideoReader] Warning: empty frame at index {frame_idx} — skipping.")
                frame_idx += 1
                if consecutive_fails >= self.max_consecutive_failures:
                    break
                continue

            # Good frame — reset failure counter
            consecutive_fails = 0
            yield frame_idx, frame
            frame_idx += 1

    # ── Class-level probe (no instance needed) ────────────────────────────

    @classmethod
    def probe(cls, path: str | Path) -> VideoMetadata:
        """
        Open, read metadata, close immediately — no frame data read.

        Ideal for quick inspection without holding a file handle.
        """
        reader = cls(path)
        reader.open()
        meta = reader.meta
        reader.close()
        return meta

    # ── Static helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fourcc_to_str(fourcc_int: int) -> str:
        """Convert integer fourcc to a human-readable 4-char string."""
        try:
            chars = [
                chr((fourcc_int >> (8 * i)) & 0xFF)
                for i in range(4)
            ]
            result = "".join(c for c in chars if c.isprintable())
            return result if result else f"0x{fourcc_int:08X}"
        except Exception:
            return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper (standalone function)
# ─────────────────────────────────────────────────────────────────────────────

def print_video_info(path: str | Path) -> VideoMetadata:
    """
    Open the video at path, print a formatted metadata table, return metadata.

    Uses ASCII-only output so it works on every Windows console encoding
    (cp1252, cp437, UTF-8, etc.).

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    VideoMetadata
    """
    import sys as _sys
    meta = VideoReader.probe(path)

    # ASCII-safe box drawing
    SEP  = "+" + "-" * 44 + "+"
    SIDE = "|"

    def _row(label: str, value: str) -> str:
        content = f"  {label:<12}: {value}"
        # pad to fixed width
        padded = f"{content:<44}"
        return f"{SIDE}{padded}{SIDE}"

    lines = [
        "",
        SEP,
        f"{SIDE}  Video Metadata Report{' ' * 23}{SIDE}",
        SEP,
        _row("File",       meta.path.name),
        _row("Resolution", f"{meta.width} x {meta.height}"),
        _row("FPS",        f"{meta.fps:.3f}"),
        _row("Frames",     f"{meta.frame_count:,}"),
        _row("Duration",   f"{meta.duration_s:.1f} s  ({meta.duration_str})"),
        _row("Codec",      meta.fourcc_str),
        SEP,
        "",
    ]
    output = "\n".join(lines)

    # Write safely regardless of console encoding
    try:
        print(output)
    except UnicodeEncodeError:
        _sys.stdout.buffer.write((output + "\n").encode("ascii", errors="replace"))

    return meta

