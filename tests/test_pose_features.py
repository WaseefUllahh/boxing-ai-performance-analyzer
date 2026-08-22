"""
tests/test_pose_features.py — Tests for the robust pose geometry abstraction layer.
"""

import math
import numpy as np
import pytest

from src.pose_features import (
    distance,
    vector,
    magnitude,
    normalize,
    angle_between_vectors,
    angle_between_points,
    velocity,
    acceleration,
    midpoint,
    PoseFeatureExtractor,
    PoseFeatures
)

class TestGeometry:
    def test_distance(self):
        assert distance((0, 0), (3, 4)) == 5.0
        assert distance((1, 1), (1, 1)) == 0.0
        assert distance(None, (3, 4)) is None
        assert distance((0, 0), None) is None

    def test_vector(self):
        assert vector((1, 2), (4, 6)) == (3.0, 4.0)
        assert vector(None, (4, 6)) is None

    def test_magnitude(self):
        assert magnitude((3, 4)) == 5.0
        assert magnitude((0, 0)) == 0.0
        assert magnitude(None) is None

    def test_normalize(self):
        v = normalize((3, 4))
        assert v is not None
        assert math.isclose(v[0], 0.6)
        assert math.isclose(v[1], 0.8)
        
        # Zero-length vector returns None to avoid div-by-zero
        assert normalize((0, 0)) is None
        assert normalize(None) is None

    def test_angle_between_vectors(self):
        # Perpendicular
        assert math.isclose(angle_between_vectors((1, 0), (0, 1)), 90.0)
        # Same
        assert math.isclose(angle_between_vectors((1, 1), (1, 1)), 0.0, abs_tol=1e-5)
        # Opposite
        assert math.isclose(angle_between_vectors((1, 0), (-1, 0)), 180.0)
        # Handle None
        assert angle_between_vectors(None, (1, 0)) is None
        assert angle_between_vectors((0, 0), (1, 0)) is None

    def test_angle_between_points(self):
        # Right angle at origin
        assert math.isclose(angle_between_points((0, 1), (0, 0), (1, 0)), 90.0)
        assert angle_between_points(None, (0, 0), (1, 0)) is None

    def test_velocity_and_acceleration(self):
        assert velocity((10, 10), (15, 20)) == (5.0, 10.0)
        assert acceleration((2, 3), (5, 1)) == (3.0, -2.0)
        assert velocity(None, (15, 20)) is None
        assert acceleration((2, 3), None) is None

    def test_midpoint(self):
        assert midpoint((0, 0), (10, 10)) == (5.0, 5.0)
        assert midpoint(None, (10, 10)) == (10, 10)
        assert midpoint((0, 0), None) == (0, 0)
        assert midpoint(None, None) is None


class TestPoseFeatureExtractor:
    def setup_method(self):
        self.extractor = PoseFeatureExtractor()
        # Create a dummy full 17x3 keypoint array
        self.kps = np.zeros((17, 3), dtype=np.float32)
        # Set dummy confidences above threshold
        self.kps[:, 2] = 0.9

    def test_missing_shoulders_invalidates_features(self):
        # Confidence 0 for shoulders
        self.kps[5, 2] = 0.0
        self.kps[6, 2] = 0.0
        features = self.extractor.extract(self.kps, 1, 1080)
        assert not features.valid

    def test_basic_centers_and_dimensions(self):
        # Set shoulders
        self.kps[5] = [100, 200, 0.9] # L
        self.kps[6] = [150, 200, 0.9] # R
        
        # Set hips
        self.kps[11] = [110, 400, 0.9] # L
        self.kps[12] = [140, 400, 0.9] # R
        
        # Set nose
        self.kps[0] = [125, 100, 0.9]
        
        features = self.extractor.extract(self.kps, 1, 1080, bbox_center=(125, 300))
        
        assert features.valid
        assert features.shoulder_width == 50.0
        assert features.hip_width == 30.0
        assert features.shoulder_center == (125.0, 200.0)
        assert features.hip_center == (125.0, 400.0)
        assert features.head_center == (125.0, 100.0)
        assert features.body_center == (125.0, 300.0)
        assert features.bbox_center == (125, 300)
        
        # Torso lean (125-125 = 0 dx)
        assert features.torso_lean_deg == 0.0
        assert features.body_orientation == 0.0

    def test_missing_keypoints_handling(self):
        # Give only L shoulder and R shoulder
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [100, 200, 0.9]
        kps[6] = [150, 200, 0.9]
        
        features = self.extractor.extract(kps, 1, 1080)
        
        assert features.valid
        assert features.shoulder_width == 50.0
        assert features.head_center is None
        assert features.hip_center is None
        assert features.body_center is None
        assert features.left_wrist is None
        assert features.right_elbow is None
        assert features.stance_width_norm == 0.0
        assert features.left_arm_extension == 0.0
        assert features.torso_lean_deg == 0.0
        assert features.left_guard is False
