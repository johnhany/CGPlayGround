"""Numerical and rendering checks for the courtyard demo (no GUI required)."""

from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import taichi as ti

from water.demo_01 import OrbitCamera, PoolRenderer, main, parse_args


@ti.kernel
def wave_sample(renderer: ti.template(), x: ti.f32, z: ti.f32, clock: ti.f32,
                amplitude: ti.f32) -> ti.types.vector(3, ti.f32):
    return renderer.water(x, z, clock, amplitude, 7.0)


@ti.kernel
def bevel_sample(renderer: ti.template(), origin: ti.types.vector(3, ti.f32),
                 direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    t, n = renderer.rounded_box_hit(origin, direction, ti.Vector([0.0, 0.0, 0.0]),
                                  ti.Vector([1.0, 1.0, 1.0]), 0.2)
    return ti.Vector([t, n.x, n.y, n.z])


@ti.kernel
def leaf_sample(renderer: ti.template(), index: ti.i32, origin: ti.types.vector(3, ti.f32),
                direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    t, n = renderer.ellipsoid_hit(origin, direction, index)
    return ti.Vector([t, n.x, n.y, n.z])


class CameraTests(unittest.TestCase):
    def test_presets_have_orthonormal_basis_and_stay_above_water(self):
        camera = OrbitCamera()
        for preset in ("overview", "waterline", "top"):
            camera.preset(preset)
            eye, forward, right, up = camera.basis()
            basis = np.stack([forward, right, up])
            np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-6)
            self.assertGreater(eye[1], 0.1)

    def test_camera_limits_remain_finite(self):
        camera = OrbitCamera()
        camera.orbit(100, 100)
        camera.zoom(-100)
        camera.pan(100, -100)
        self.assertTrue(np.isfinite(camera.basis()).all())
        self.assertGreater(camera.basis()[0][1], 0.1)
        self.assertLessEqual(abs(camera.target[0]), 3.5)
        self.assertLessEqual(abs(camera.target[2]), 2.5)

    def test_invalid_render_sizes_are_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["--width", "0"])
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(["--samples", "3"])


class StartupTests(unittest.TestCase):
    def test_first_frame_finishes_before_window_creation(self):
        # The OS event loop must never be held up by cold render compilation
        # after a native window has already appeared.
        events = []
        renderer = MagicMock()
        renderer.draw.side_effect = lambda *args: events.append("render")
        window = MagicMock()
        window.running = True
        window.get_canvas.return_value.set_image.side_effect = lambda *args: events.append("image")
        window.show.side_effect = lambda: events.append("show")

        def create_window(*args, **kwargs):
            events.append("create_window")
            return window

        with patch("water.demo_01.ti.init"), \
             patch("water.demo_01.ti.sync", side_effect=lambda: events.append("sync")), \
             patch("water.demo_01.PoolRenderer", return_value=renderer), \
             patch("water.demo_01.ti.ui.Window", side_effect=create_window), \
             patch("builtins.print"):
            main(["--window-frames", "1"])
        self.assertEqual(events, ["render", "sync", "create_window", "image", "show"])


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = PoolRenderer(64, 40)

    def test_beveled_box_intersections_and_normals(self):
        renderer = self.renderer
        sample = np.array(bevel_sample(renderer, [0.0, 0.0, 3.0], [0.0, 0.0, -1.0]))
        np.testing.assert_allclose(sample, [2.0, 0.0, 0.0, 1.0], atol=0.001)
        inside = np.array(bevel_sample(renderer, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
        self.assertAlmostEqual(inside[0], 1.0, places=3)
        corner = np.array(bevel_sample(renderer, [0.9, 0.9, 3.0], [0.0, 0.0, -1.0]))
        point = np.array([0.9, 0.9, 3.0 - corner[0]])
        distance = np.linalg.norm(np.maximum(np.abs(point) - 0.8, 0.0)) - 0.2
        self.assertLess(abs(distance), 0.001)
        self.assertGreater(corner[1], 0.4)
        self.assertAlmostEqual(np.linalg.norm(corner[1:]), 1.0, places=3)
        miss = bevel_sample(renderer, [1.0, 1.0, 3.0], [0.0, 0.0, -1.0])
        self.assertGreater(miss[0], 1e5)

    def test_oriented_foliage_ray_intersection(self):
        renderer = self.renderer
        index = 5
        center = np.array(renderer.leaf_center[index])
        rotation = np.array(renderer.leaf_rotation[index])
        radius = renderer.leaf_radius[index][0]
        axis = rotation[:, 0]
        sample = np.array(leaf_sample(renderer, index, center + axis * radius * 3, -axis))
        self.assertAlmostEqual(sample[0], 2 * radius, places=4)
        np.testing.assert_allclose(sample[1:], axis, atol=1e-4)

    def test_foliage_groups_enclose_all_geometry(self):
        renderer = self.renderer
        for group in range(renderer.groups):
            center = np.array(renderer.group_center[group])
            extent = np.array(renderer.group_extent[group])
            start, end = renderer.group_range[group]
            for index in range(start, end):
                object_center = np.array(renderer.leaf_center[index])
                bounds = np.abs(np.array(renderer.leaf_rotation[index])) @ np.array(renderer.leaf_radius[index])
                self.assertTrue((object_center - bounds >= center - extent - 1e-5).all())
                self.assertTrue((object_center + bounds <= center + extent + 1e-5).all())

    def test_spatial_aa_matches_high_resolution_linear_average(self):
        # Match a 2x2 subpixel grid to independently rendered high-resolution rays.
        low = self.renderer
        high = PoolRenderer(128, 80, samples=1)
        camera = OrbitCamera()
        low.draw(camera, 2.0)
        high.draw(camera, 2.0)
        fine = high.image.to_numpy()
        averaged = fine.reshape(64, 2, 40, 2, 3).mean(axis=(1, 3))
        # Tone mapping follows the radiance average, so exact display-space
        # equality is neither expected nor correct; bound the reconstruction error.
        error = np.mean(np.abs(low.image.to_numpy() - averaged))
        self.assertLess(error, 0.025)
        low.samples = 1
        low.draw(camera, 2.0)
        self.assertGreater(np.mean(np.abs(low.image.to_numpy() - averaged)), error)
        low.samples = 4

    def test_wave_gradient_matches_finite_differences(self):
        epsilon = 0.001
        for x, z in [(0.1, 0.4), (-0.85, 0.25), (2.95, 1.95)]:
            sample = np.array(wave_sample(self.renderer, x, z, 2.0, 0.035))
            dx = (wave_sample(self.renderer, x + epsilon, z, 2.0, 0.035)[0]
                  - wave_sample(self.renderer, x - epsilon, z, 2.0, 0.035)[0]) / (2 * epsilon)
            dz = (wave_sample(self.renderer, x, z + epsilon, 2.0, 0.035)[0]
                  - wave_sample(self.renderer, x, z - epsilon, 2.0, 0.035)[0]) / (2 * epsilon)
            np.testing.assert_allclose(sample[1:], [dx, dz], atol=2e-4)

    def test_zero_amplitude_and_pool_edges_are_flat(self):
        np.testing.assert_allclose(wave_sample(self.renderer, 0.4, 0.2, 2.0, 0.0), 0.0)
        for x, z in [(3.2, 0.0), (0.0, 2.15), (-3.2, -1.0)]:
            np.testing.assert_allclose(wave_sample(self.renderer, x, z, 2.0, 0.035), 0.0, atol=1e-6)

    def test_presets_produce_finite_nonblank_frames(self):
        camera = OrbitCamera()
        for preset in ("overview", "waterline", "top"):
            camera.preset(preset)
            self.renderer.draw(camera, 2.0)
            frame = self.renderer.image.to_numpy()
            self.assertTrue(np.isfinite(frame).all(), preset)
            self.assertGreater(frame.std(), 0.04, preset)
            self.assertGreaterEqual(frame.min(), 0.0)
            self.assertLessEqual(frame.max(), 1.0)

    def test_wave_motion_changes_the_render(self):
        camera = OrbitCamera()
        camera.preset("top")
        self.renderer.draw(camera, 0.0)
        first = self.renderer.image.to_numpy()
        self.renderer.draw(camera, 1.0)
        second = self.renderer.image.to_numpy()
        self.assertGreater(np.mean(np.abs(first - second)), 0.002)
        # A paused scene is deterministic.
        self.renderer.draw(camera, 1.0)
        np.testing.assert_array_equal(second, self.renderer.image.to_numpy())
        # A flat water surface has no animated refractive caustics.
        self.renderer.draw(camera, 0.0, amplitude=0.0)
        flat = self.renderer.image.to_numpy()
        self.renderer.draw(camera, 1.0, amplitude=0.0)
        np.testing.assert_array_equal(flat, self.renderer.image.to_numpy())


if __name__ == "__main__":
    unittest.main()
