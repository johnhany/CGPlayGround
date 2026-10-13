"""Photographic finish checks: film-grain determinism and statistics,
thin-lens depth of field blur ordering, and focus-plane placement (no GUI)."""

import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import NightLakeScene


def gradient_energy(frame, x0, x1, y0, y1):
    """Mean absolute luma gradient inside an array-coordinate region.

    Luma is normalized by its regional mean first: defocused glints average
    dimmer than pinhole glints, and the normalization keeps the metric on
    texture sharpness instead of absolute brightness.
    """
    luma = frame[x0:x1, y0:y1].mean(axis=2)
    luma = luma / (luma.mean() + 1e-6)
    gx = np.abs(np.diff(luma, axis=0)).mean()
    gy = np.abs(np.diff(luma, axis=1)).mean()
    return float(gx + gy)


class FilmGrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.camera = FixedCamera()

    def _draw(self, grain):
        scene = NightLakeScene(160, 90)
        scene.post_controls[None] = [0.12, 0.9, 0.22, grain]
        scene.draw(self.camera, 2.0)
        return scene.image.to_numpy()

    def test_same_frame_renders_bitwise_identical(self):
        first = self._draw(0.05)
        second = self._draw(0.05)
        np.testing.assert_array_equal(first, second)

    def test_grain_adds_variance_without_lifting_the_mean(self):
        clean = self._draw(0.0)
        grained = self._draw(0.08)
        self.assertTrue(np.isfinite(grained).all())
        # Measure on a bright sky patch: the grain kernel clamps at zero,
        # and the dark lower half of this frame holds enough near-black
        # pixels that clamping can mask the added variance globally.
        clean = clean[15:75, 66:86]
        grained = grained[15:75, 66:86]
        mean_shift = abs(float(grained.mean()) - float(clean.mean()))
        self.assertLess(mean_shift, 0.01)
        self.assertGreater(float(grained.var()), 1.05 * float(clean.var()))


class DepthOfFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.camera = FixedCamera()

    def _draw(self, enabled, focus=14.0, aperture=0.25, samples=8.0):
        scene = NightLakeScene(160, 90)
        # Rain off: streak splats ignore depth of field by design, and
        # their identical gradient would dilute the blur measurement.
        # Grain off for the same reason; both are covered separately.
        scene.water_controls[None] = [1.0, 1.0, 0.0, 0.06]
        scene.camera_controls[None] = [1.0 if enabled else 0.0, aperture,
                                       focus, samples]
        finish = list(scene.post_controls[None])
        finish[3] = 0.0
        scene.post_controls[None] = finish
        scene.draw(self.camera, 2.0)
        return scene.image.to_numpy()

    def test_focus_plane_sits_on_the_waterline_band(self):
        sharp = self._draw(False)
        blurred = self._draw(True)
        # Array coordinates: x 30-70, y 58-70 hold the near water (14-25
        # m) at the 14 m focus plane; x 100-118, y 36-44 hold the far-water
        # moon glitter column at roughly 40-80 m.
        near_sharp = gradient_energy(sharp, 30, 70, 58, 70)
        near_dof = gradient_energy(blurred, 30, 70, 58, 70)
        far_sharp = gradient_energy(sharp, 100, 118, 36, 44)
        far_dof = gradient_energy(blurred, 100, 118, 36, 44)
        self.assertGreater(near_dof, 0.7 * near_sharp)
        self.assertLess(far_dof, 0.8 * far_sharp)

    def test_focus_shift_moves_sharpness_to_the_background(self):
        near_focus = self._draw(True, focus=6.0)
        far_focus = self._draw(True, focus=60.0)
        # The far-bank rise and its silhouette trees sit at 60-90 m; the
        # 60 m focus plane sharpens them while the 6 m plane blurs them.
        near_energy = gradient_energy(near_focus, 98, 124, 44, 56)
        far_energy = gradient_energy(far_focus, 98, 124, 44, 56)
        self.assertGreater(far_energy, 1.1 * near_energy)


if __name__ == "__main__":
    unittest.main()
