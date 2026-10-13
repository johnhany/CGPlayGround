"""Cameras for the night lake scene.

FixedCamera is the production view: an elevated eye on the bank looking
across the lake, yawed a little toward the moon so the disk, the glitter
path, and the far shore share one frame. debug_camera returns an
OrbitCamera preset for interactive inspection during development.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from water.scene import OrbitCamera

__all__ = ["FixedCamera", "debug_camera"]

# Aim yaw between frame centre and the moon azimuth so the moon lands
# left of centre while the lake keeps the rest of the width.
_VIEW_YAW = -0.30
_VIEW_DISTANCE = 30.0
_VIEW_DROP = 2.4


@dataclass
class FixedCamera:
    """Fixed elevated viewpoint; basis() matches the OrbitCamera signature.

    The eye sits 5.5 m up on the bank behind the beach, several metres
    above the local ground. The view yaws about seventeen degrees toward
    the moon and pitches down about four and a half degrees, so the
    waterline sits near the frame bottom, the lake fills the lower third
    of the frame, and the low moon stays inside the upper left.
    """

    eye: np.ndarray = field(default_factory=lambda: np.array([0.0, 6.0, 0.0], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.array([
        _VIEW_DISTANCE * math.sin(_VIEW_YAW),
        6.0 - _VIEW_DROP,
        _VIEW_DISTANCE * math.cos(_VIEW_YAW),
    ], dtype=np.float32))
    fov: float = 48.0

    def basis(self):
        forward = self.target - self.eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return tuple(v.astype(np.float32) for v in (self.eye, forward, right, up))


def debug_camera():
    """Orbit camera preset aimed at the moon side of the lake."""
    return OrbitCamera(yaw=0.12, pitch=0.18, distance=10.0,
                       target=np.array([0.0, 1.4, 8.0], dtype=np.float32), fov=48.0)
