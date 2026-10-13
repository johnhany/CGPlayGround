"""Stage 1 checks for the night lake scene: finite frame, night brightness
range, and a moon disk brighter than the zenith sky (no GUI required)."""

import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import NightLakeScene
from night_lake.sky import MOON_DIRECTION


class NightLakeSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)

    def test_frame_is_finite_and_dim_but_not_black(self):
        self.scene.draw(FixedCamera())
        frame = self.scene.image.to_numpy()
        self.assertTrue(np.isfinite(frame).all())
        self.assertGreaterEqual(frame.min(), 0.0)
        self.assertLessEqual(frame.max(), 1.0)
        mean = frame.mean()
        self.assertGreater(mean, 0.02)
        self.assertLess(mean, 0.35)
        # A night sky still carries visible structure (stars, bands, moon).
        self.assertGreater(frame.std(), 0.01)

    def test_moon_region_is_brighter_than_zenith(self):
        camera = FixedCamera()
        eye, forward, right, up = camera.basis()
        scale = float(np.tan(np.radians(camera.fov / 2)))
        width, height = self.scene.width, self.scene.height
        # Rebuild the per-pixel ray grid on the CPU to locate the moon.
        px = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * width / height * scale
        py = (2.0 * (np.arange(height) + 0.5) / height - 1.0) * scale
        grid_x, grid_y = np.meshgrid(px, py, indexing="ij")
        directions = (forward[None, None, :] + right[None, None, :] * grid_x[..., None]
                      + up[None, None, :] * grid_y[..., None])
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        alignment = directions @ MOON_DIRECTION
        moon_x, moon_y = np.unravel_index(np.argmax(alignment), alignment.shape)
        self.assertGreater(alignment.max(), 0.995, "moon direction should be inside the frame")
        self.scene.draw(camera)
        frame = self.scene.image.to_numpy()
        moon_patch = frame[max(0, moon_x - 3):moon_x + 4, max(0, moon_y - 3):moon_y + 4]
        zenith_x = np.argmax(directions[..., 1].max(axis=1))
        zenith_y = np.argmax(directions[..., 1].max(axis=0))
        zenith_patch = frame[max(0, zenith_x - 3):zenith_x + 4, max(0, zenith_y - 3):zenith_y + 4]
        self.assertGreater(moon_patch.mean(), zenith_patch.mean() + 0.25)

    def test_frame_is_deterministic(self):
        camera = FixedCamera()
        # A paused rain field keeps the water height field static, so two
        # draws at the same clock must match exactly.
        controls = tuple(self.scene.water_controls[None])
        self.scene.water_controls[None] = [controls[0], 0.0, 0.0, controls[3]]
        try:
            self.scene.draw(camera)
            first = self.scene.image.to_numpy()
            self.scene.draw(camera)
            np.testing.assert_array_equal(first, self.scene.image.to_numpy())
        finally:
            self.scene.water_controls[None] = controls


class CompositionTests(unittest.TestCase):
    """Frame-layout acceptance for the recomposed elevated view: the lake
    dominates the lower half, the moon disk sits inside the frame, and
    trees frame both sides (no GUI required)."""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)

    def test_lake_covers_a_third_to_half_the_frame(self):
        self.scene.draw(FixedCamera(), 2.0)
        obj = self.scene.object_id.to_numpy()
        water = float((obj == 1).mean())
        self.assertGreater(water, 0.33)
        self.assertLess(water, 0.50)

    def test_waterline_sits_near_the_frame_bottom(self):
        self.scene.draw(FixedCamera(), 2.0)
        obj = self.scene.object_id.to_numpy()
        # Array row 0 is the frame bottom; the beach band there must carry
        # visible mud, not open water reaching under the camera.
        bottom = obj[:, :int(self.scene.height * 0.2)]
        self.assertGreater(float((bottom == 2).mean()), 0.10)

    def test_trees_frame_both_sides(self):
        self.scene.draw(FixedCamera(), 2.0)
        obj = self.scene.object_id.to_numpy()
        width = self.scene.width
        margin = int(width * 0.2)
        left = float((obj[:margin] == 4).mean())
        right = float((obj[-margin:] == 4).mean())
        self.assertGreater(left, 0.005)
        self.assertGreater(right, 0.005)

    def test_moon_disk_is_inside_the_frame(self):
        camera = FixedCamera()
        eye, forward, right, up = camera.basis()
        scale = float(np.tan(np.radians(camera.fov / 2)))
        width, height = self.scene.width, self.scene.height
        px = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * width / height * scale
        py = (2.0 * (np.arange(height) + 0.5) / height - 1.0) * scale
        grid_x, grid_y = np.meshgrid(px, py, indexing="ij")
        directions = (forward[None, None, :] + right[None, None, :] * grid_x[..., None]
                      + up[None, None, :] * grid_y[..., None])
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        alignment = directions @ MOON_DIRECTION
        self.assertGreater(float(alignment.max()), 0.995,
                           "moon direction should be inside the frame")


if __name__ == "__main__":
    unittest.main()
