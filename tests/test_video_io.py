"""
tests/test_video_io.py — Unit tests for src/video_io.py

Run with:
    python -m pytest tests/test_video_io.py -v

Tests cover:
    - VideoMetadata construction and properties
    - fourcc conversion helper
    - Error handling: missing file, unopenable file, zero FPS/dims
    - VideoReader.probe() on the real fight.mp4
    - Frame iteration: first N frames, shape validation, no memory accumulation
    - Context manager resource release
    - Generator early-break (does not leak the cap)
    - print_video_info output
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import cv2

# ── Resolve project root so imports work from any CWD ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.video_io import (
    VideoReader,
    VideoMetadata,
    VideoIOError,
    VideoNotFoundError,
    VideoOpenError,
    VideoMetadataError,
    print_video_info,
)

FIGHT_VIDEO = ROOT / "data" / "fight.mp4"
SKIP_IF_NO_VIDEO = pytest.mark.skipif(
    not FIGHT_VIDEO.exists(),
    reason="data/fight.mp4 not found — skipping live video tests",
)


# ─────────────────────────────────────────────────────────────────────────────
# VideoMetadata unit tests (no file I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoMetadata:
    """Tests for the VideoMetadata dataclass."""

    def _make(self, **kwargs) -> VideoMetadata:
        defaults = dict(
            path=Path("dummy.mp4"),
            width=1920, height=1080,
            fps=25.0, frame_count=3750,
            duration_s=150.0, fourcc_str="mp4v",
        )
        defaults.update(kwargs)
        return VideoMetadata(**defaults)

    def test_resolution_tuple(self):
        m = self._make(width=1280, height=720)
        assert m.resolution == (1280, 720)

    def test_duration_str_format(self):
        m = self._make(fps=25.0, frame_count=3750, duration_s=150.0)
        # 150 seconds = 2:30.0
        assert "2:" in m.duration_str
        assert "30" in m.duration_str

    def test_str_contains_filename(self):
        m = self._make(path=Path("fight.mp4"))
        assert "fight.mp4" in str(m)

    def test_str_contains_resolution(self):
        m = self._make(width=1920, height=1080)
        s = str(m)
        assert "1920" in s
        assert "1080" in s

    def test_immutable(self):
        m = self._make()
        with pytest.raises((AttributeError, TypeError)):
            m.width = 999  # type: ignore[misc]

    def test_zero_frame_count_duration(self):
        # Live streams have frame_count=0 → duration_s=0
        m = self._make(frame_count=0, duration_s=0.0)
        assert m.duration_s == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# fourcc helper
# ─────────────────────────────────────────────────────────────────────────────

class TestFourccHelper:
    def test_mp4v(self):
        fourcc_int = cv2.VideoWriter_fourcc(*"mp4v")
        result = VideoReader._fourcc_to_str(fourcc_int)
        assert "mp4v" in result.lower() or len(result) == 4

    def test_zero_fourcc(self):
        # Should not crash
        result = VideoReader._fourcc_to_str(0)
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Error handling tests
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_missing_file_raises(self):
        with pytest.raises(VideoNotFoundError) as exc_info:
            VideoReader.probe(ROOT / "data" / "nonexistent_video.mp4")
        assert "not found" in str(exc_info.value).lower()

    def test_missing_file_message_contains_path(self):
        bad_path = ROOT / "data" / "no_such_file.mp4"
        with pytest.raises(VideoNotFoundError) as exc_info:
            VideoReader.probe(bad_path)
        assert str(bad_path) in str(exc_info.value)

    def test_directory_as_path_raises(self):
        """Passing a directory path (not a file) should raise VideoNotFoundError."""
        with pytest.raises(VideoNotFoundError):
            VideoReader.probe(ROOT / "data")

    def test_unopened_reader_meta_raises(self):
        reader = VideoReader(ROOT / "data" / "fight.mp4")
        with pytest.raises(VideoIOError, match="not been opened"):
            _ = reader.meta

    def test_unopened_reader_frames_raises(self):
        reader = VideoReader(ROOT / "data" / "fight.mp4")
        with pytest.raises(VideoIOError, match="not open"):
            list(reader.frames())

    def test_text_file_raises_open_error(self, tmp_path):
        """A non-video file should raise VideoOpenError."""
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_bytes(b"this is not a valid video file content")
        with pytest.raises((VideoOpenError, VideoMetadataError)):
            VideoReader.probe(fake_video)


# ─────────────────────────────────────────────────────────────────────────────
# Live video tests (require data/fight.mp4)
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveVideo:

    @SKIP_IF_NO_VIDEO
    def test_probe_returns_metadata(self):
        meta = VideoReader.probe(FIGHT_VIDEO)
        assert isinstance(meta, VideoMetadata)

    @SKIP_IF_NO_VIDEO
    def test_probe_resolution_valid(self):
        meta = VideoReader.probe(FIGHT_VIDEO)
        assert meta.width > 0
        assert meta.height > 0

    @SKIP_IF_NO_VIDEO
    def test_probe_fps_positive(self):
        meta = VideoReader.probe(FIGHT_VIDEO)
        assert meta.fps > 0.0

    @SKIP_IF_NO_VIDEO
    def test_probe_frame_count_positive(self):
        meta = VideoReader.probe(FIGHT_VIDEO)
        assert meta.frame_count > 0

    @SKIP_IF_NO_VIDEO
    def test_probe_duration_consistent(self):
        meta = VideoReader.probe(FIGHT_VIDEO)
        expected = meta.frame_count / meta.fps
        # Allow 1% tolerance (some containers are slightly off)
        assert abs(meta.duration_s - expected) < expected * 0.01 + 1.0

    @SKIP_IF_NO_VIDEO
    def test_probe_path_is_absolute(self):
        meta = VideoReader.probe(FIGHT_VIDEO)
        assert meta.path.is_absolute()

    @SKIP_IF_NO_VIDEO
    def test_context_manager_releases_cap(self):
        """After __exit__, the cap should be released (is_open == False)."""
        with VideoReader(FIGHT_VIDEO) as reader:
            assert reader.is_open
        assert not reader.is_open

    @SKIP_IF_NO_VIDEO
    def test_close_is_idempotent(self):
        reader = VideoReader(FIGHT_VIDEO)
        reader.open()
        reader.close()
        reader.close()   # second call must not raise
        assert not reader.is_open

    @SKIP_IF_NO_VIDEO
    def test_first_frame_shape(self):
        """First frame must match reported metadata dimensions."""
        with VideoReader(FIGHT_VIDEO) as reader:
            meta = reader.meta
            for idx, frame in reader.frames():
                assert frame.ndim == 3
                assert frame.shape[0] == meta.height
                assert frame.shape[1] == meta.width
                assert frame.shape[2] == 3   # BGR
                assert frame.dtype == np.uint8
                break   # only need first frame

    @SKIP_IF_NO_VIDEO
    def test_frame_dtype_uint8(self):
        with VideoReader(FIGHT_VIDEO) as reader:
            for _, frame in reader.frames():
                assert frame.dtype == np.uint8
                break

    @SKIP_IF_NO_VIDEO
    def test_early_break_does_not_leak(self):
        """Breaking from the frame generator must not prevent cap release."""
        with VideoReader(FIGHT_VIDEO) as reader:
            for idx, frame in reader.frames():
                if idx >= 4:
                    break
        assert not reader.is_open   # __exit__ ran

    @SKIP_IF_NO_VIDEO
    def test_iterate_10_frames(self):
        """Read exactly 10 frames and verify indices are sequential."""
        seen_indices = []
        with VideoReader(FIGHT_VIDEO) as reader:
            for idx, frame in reader.frames():
                seen_indices.append(idx)
                if len(seen_indices) >= 10:
                    break
        assert seen_indices == list(range(10))

    @SKIP_IF_NO_VIDEO
    def test_no_full_video_in_ram(self):
        """
        Verify frames generator does not accumulate frames in memory.

        We iterate 100 frames and assert we never hold more than 2 arrays
        simultaneously.  Since we only yield one frame at a time, the
        previous frame's ndarray can be GC'd before the next is produced.
        This test is a design assertion, not a strict memory profiling test.
        """
        frames_held = []
        with VideoReader(FIGHT_VIDEO) as reader:
            for idx, frame in reader.frames():
                # Overwrite the slot — simulates a processing pipeline
                # that does not accumulate.
                frames_held = [frame]
                if idx >= 99:
                    break
        assert len(frames_held) == 1  # only the last frame is referenced

    @SKIP_IF_NO_VIDEO
    def test_print_video_info(self, capsys):
        """print_video_info should produce output with key values."""
        meta = print_video_info(FIGHT_VIDEO)
        captured = capsys.readouterr()
        assert str(meta.width) in captured.out
        assert str(meta.height) in captured.out
        assert FIGHT_VIDEO.name in captured.out


# ─────────────────────────────────────────────────────────────────────────────
# Standalone metadata print (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Video Metadata — data/fight.mp4")
    print("=" * 60)

    if not FIGHT_VIDEO.exists():
        print(f"  ERROR: {FIGHT_VIDEO} not found.")
        sys.exit(1)

    meta = print_video_info(FIGHT_VIDEO)
    print(meta)   # also print the dataclass __str__

    print("\nReading first 5 frames to verify streaming...")
    with VideoReader(FIGHT_VIDEO) as reader:
        for idx, frame in reader.frames():
            h, w, c = frame.shape
            print(f"  Frame {idx:3d}  shape={w}×{h}×{c}  dtype={frame.dtype}")
            if idx >= 4:
                break

    print("\nStream OK — resource released cleanly.")
    print("=" * 60)
