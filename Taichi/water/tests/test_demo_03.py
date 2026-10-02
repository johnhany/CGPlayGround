"""Spectrum, intersection, and rendering checks for the sunset sea demo."""

from contextlib import redirect_stderr
from io import StringIO
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import taichi as ti

from water.demo_03 import (CROSS_COUNT, FOG_DISTANCE, GRAVITY, OCEAN_PRESETS,
                           WAVE_COUNT, GerstnerSpectrum, SunsetOceanRenderer,
                           apply_preset, main, parse_args)
from water.demo_01 import OrbitCamera


@ti.kernel
def surface_probe(renderer: ti.template(), x: ti.f32, z: ti.f32, t: ti.f32) -> ti.types.vector(4, ti.f32):
    y, n = renderer.spectrum.surface(x, z, t)
    return ti.Vector([y, n.x, n.y, n.z])


@ti.kernel
def hit_probes(renderer: ti.template(), eye: ti.types.vector(3, ti.f32),
               directions: ti.types.ndarray(dtype=ti.f32, ndim=2),
               result: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(directions.shape[0]):
        d = ti.Vector([directions[i, 0], directions[i, 1], directions[i, 2]])
        t, n, _ = renderer.water_hit(eye, d, 1.0)
        result[i, 0], result[i, 1] = t, n.y


def numpy_surface(table, x, z, t):
    env = 1.0 + float((table["env_amp"] * np.sin(table["env_k"] @ np.array([x, z])
                                                  + table["env_phase"])).sum())
    dx, dz = table["dir"][:, 0], table["dir"][:, 1]
    f = table["k"] * (dx * x + dz * z) - table["omega"] * t + table["phase"]
    a, q = table["amp"], table["steep"]
    y = float((a * env * np.sin(f)).sum())
    px = x + env * float((q * a * dx * np.cos(f)).sum())
    pz = z + env * float((q * a * dz * np.cos(f)).sum())
    return y, px, pz


def numpy_normal(table, x, z, t, eps=1e-4):
    def point(x0, z0):
        y, px, pz = numpy_surface(table, x0, z0, t)
        return np.array([px, y, pz])
    dpx = (point(x + eps, z) - point(x - eps, z)) / (2 * eps)
    dpz = (point(x, z + eps) - point(x, z - eps)) / (2 * eps)
    normal = np.cross(dpz, dpx)
    return normal / np.linalg.norm(normal)


class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.spectrum = GerstnerSpectrum(seed=17)

    def test_dispersion_uses_deep_water_relation(self):
        table = self.spectrum.table
        np.testing.assert_allclose(table["omega"] ** 2, GRAVITY * table["k"], rtol=1e-6)

    def test_steepness_sum_respects_constraint(self):
        table = self.spectrum.table
        total = float((table["steep"] * table["k"] * table["amp"]).sum())
        self.assertLessEqual(total, 0.6 + 1e-6)
        self.assertGreater(total, 0.55)

    def test_amplitudes_normalize_to_wave_height(self):
        table = self.spectrum.table
        self.assertEqual(len(table["amp"]), WAVE_COUNT)
        self.assertTrue((table["amp"] > 0).all())
        self.assertAlmostEqual(float(table["amp"].sum()), 1.3, places=6)
        self.assertAlmostEqual(float(self.spectrum.amp_total[None]), 1.3, places=6)

    def test_rebuild_is_deterministic_and_sensitive(self):
        self.spectrum.rebuild(8.0, 35.0, 1.3, 0.6, 0.8)
        first = self.spectrum.omega.to_numpy().copy()
        self.spectrum.rebuild(8.0, 35.0, 1.3, 0.6, 0.8)
        np.testing.assert_array_equal(self.spectrum.omega.to_numpy(), first)
        self.spectrum.rebuild(16.0, 120.0, 1.4, 0.8, 2.0)
        self.assertGreater(float(np.abs(self.spectrum.omega.to_numpy() - first).sum()), 1.0)
        self.spectrum.rebuild(8.0, 35.0, 1.3, 0.6, 0.8)

    def test_rebuild_rejects_out_of_range_inputs(self):
        for params in [(8.0, 35.0, 0.85, 1.2, 1.6), (30.0, 35.0, 0.85, 0.5, 1.6)]:
            with self.assertRaises(ValueError):
                self.spectrum.rebuild(*params)

    def test_wavelengths_are_irregular_and_cross_swell_differs(self):
        table = self.spectrum.table
        ratios = table["k"][1:-CROSS_COUNT] / table["k"][:-CROSS_COUNT - 1]
        self.assertGreater(float(ratios.max() - ratios.min()), 0.05)
        wind = np.array([math.cos(math.radians(35.0)), math.sin(math.radians(35.0))])
        alignment = np.abs(table["dir"][-CROSS_COUNT:] @ wind)
        self.assertLess(float(alignment.max()), 0.7)
        envelope_k = np.linalg.norm(table["env_k"], axis=1)
        self.assertTrue((envelope_k < table["k"].min() * 0.2).all())


class SurfaceKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SunsetOceanRenderer(64, 40, samples=1)

    def test_height_matches_numpy_reference(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            x, z = rng.uniform(-30, 30, 2)
            probe = np.array(surface_probe(self.renderer, float(x), float(z), 2.5))
            expected, _, _ = numpy_surface(self.renderer.spectrum.table, x, z, 2.5)
            self.assertAlmostEqual(probe[0], expected, places=5)

    def test_normal_is_unit_upward_and_matches_finite_differences(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            x, z = rng.uniform(-30, 30, 2)
            probe = np.array(surface_probe(self.renderer, float(x), float(z), 4.0))
            normal = probe[1:]
            self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=5)
            self.assertGreater(normal[1], 0.3)
            reference = numpy_normal(self.renderer.spectrum.table, x, z, 4.0)
            np.testing.assert_allclose(normal, reference, atol=5e-3)

    def test_steep_crests_tilt_further_than_troughs(self):
        # With steepness the sharpest slopes exceed the pure sine slope k A.
        renderer = self.renderer
        renderer.spectrum.rebuild(10.0, 35.0, 1.2, 0.9, 1.6)
        max_tilt = 0.0
        for i in range(400):
            x, z = np.random.default_rng(i).uniform(-40, 40, 2)
            probe = np.array(surface_probe(renderer, float(x), float(z), 0.37))
            max_tilt = max(max_tilt, math.sqrt(probe[1] ** 2 + probe[3] ** 2) / probe[2])
        renderer.spectrum.rebuild(8.0, 35.0, 1.3, 0.6, 0.8)
        self.assertGreater(max_tilt, 0.28)


class IntersectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SunsetOceanRenderer(64, 40, samples=1)

    def test_downward_ray_hits_the_surface(self):
        result = np.zeros((1, 2), np.float32)
        hit_probes(self.renderer, np.array([1.3, 5.0, -0.7], np.float32),
                   np.array([[0.0, -1.0, 0.0]], np.float32), result)
        t, ny = float(result[0, 0]), float(result[0, 1])
        expected, _, _ = numpy_surface(self.renderer.spectrum.table, 1.3, -0.7, 1.0)
        self.assertAlmostEqual(5.0 - t, expected, delta=2e-3)
        self.assertGreater(ny, 0.3)

    def test_upward_and_horizontal_rays_miss(self):
        directions = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                               [0.3, 0.05, -0.2]], np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        result = np.zeros((3, 2), np.float32)
        hit_probes(self.renderer, np.array([0.0, 2.0, 0.0], np.float32), directions, result)
        self.assertTrue((result[:, 0] > 1e5).all())

    def test_grazing_hit_matches_dense_reference(self):
        eye = np.array([0.0, 0.9, 8.0], np.float32)
        rng = np.random.default_rng(11)
        directions = []
        while len(directions) < 12:
            d = np.array([rng.uniform(-0.4, 0.4), -rng.uniform(0.04, 0.20), -1.0])
            d /= np.linalg.norm(d)
            directions.append(d)
        directions = np.array(directions, np.float32)
        result = np.zeros((len(directions), 2), np.float32)
        hit_probes(self.renderer, eye, directions, result)
        table = self.renderer.spectrum.table
        for d, row in zip(directions, result):
            ts = np.linspace(0.0, 40.0, 4000)
            points = eye + ts[:, None] * d
            heights = np.array([numpy_surface(table, p[0], p[2], 1.0)[0] for p in points])
            residual = points[:, 1] - heights
            crossings = np.flatnonzero((residual[:-1] > 0) & (residual[1:] <= 0))
            self.assertGreater(len(crossings), 0)
            k = crossings[0]
            reference = ts[k] + (ts[k + 1] - ts[k]) * residual[k] / (residual[k] - residual[k + 1])
            self.assertAlmostEqual(float(row[0]), reference, delta=0.01)
            self.assertGreater(float(row[1]), 0.0)

    def test_hit_point_residual_is_small(self):
        rng = np.random.default_rng(13)
        directions = rng.normal(0, 1, (8, 3)).astype(np.float32)
        directions[:, 1] = -np.abs(directions[:, 1]) - 0.1
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        result = np.zeros((8, 2), np.float32)
        eye = np.array([0.5, 1.2, 0.5], np.float32)
        hit_probes(self.renderer, eye, directions, result)
        table = self.renderer.spectrum.table
        for d, row in zip(directions, result):
            p = eye + float(row[0]) * d
            height, _, _ = numpy_surface(table, p[0], p[2], 1.0)
            self.assertAlmostEqual(p[1], height, delta=5e-3)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SunsetOceanRenderer(64, 40, samples=1)

    def test_presets_produce_finite_nonblank_frames(self):
        camera = OrbitCamera()
        for lighting in (0.0, 1.0):
            for preset in ("overview", "waterline", "top"):
                apply_preset(camera, preset)
                self.renderer.draw(camera, 1.0, lighting=lighting)
                frame = self.renderer.image.to_numpy()
                self.assertTrue(np.isfinite(frame).all(), (preset, lighting))
                self.assertGreater(frame.std(), 0.04, (preset, lighting))
                self.assertGreaterEqual(frame.min(), 0.0)
                self.assertLessEqual(frame.max(), 1.0)

    def test_paused_state_is_deterministic(self):
        camera = OrbitCamera()
        apply_preset(camera, "overview")
        self.renderer.draw(camera, 2.0, lighting=0.0)
        first = self.renderer.image.to_numpy()
        self.renderer.draw(camera, 2.0, lighting=0.0)
        np.testing.assert_array_equal(first, self.renderer.image.to_numpy())
        self.renderer.draw(camera, 2.5, lighting=0.0)
        self.assertGreater(np.abs(self.renderer.image.to_numpy() - first).max(), 0.005)

    def test_wind_rebuild_changes_the_frame(self):
        camera = OrbitCamera()
        apply_preset(camera, "waterline")
        self.renderer.spectrum.rebuild(2.0, 35.0, 0.3, 0.3, 1.2)
        self.renderer.draw(camera, 1.0, lighting=0.0)
        calm = self.renderer.image.to_numpy()
        self.renderer.spectrum.rebuild(18.0, 200.0, 1.6, 0.8, 2.0)
        self.renderer.draw(camera, 1.0, lighting=0.0)
        storm = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(storm - calm).mean(), 0.01)
        self.renderer.spectrum.rebuild(8.0, 35.0, 1.3, 0.6, 0.8)

    def test_day_and_sunset_differ(self):
        camera = OrbitCamera()
        apply_preset(camera, "overview")
        self.renderer.draw(camera, 1.0, lighting=0.0)
        sunset = self.renderer.image.to_numpy()
        self.renderer.draw(camera, 1.0, lighting=1.0)
        day = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(day - sunset).mean(), 0.02)

    def test_apply_preset_sets_camera_fields(self):
        camera = OrbitCamera()
        for name, (yaw, pitch, distance) in OCEAN_PRESETS.items():
            apply_preset(camera, name)
            self.assertAlmostEqual(camera.yaw, yaw)
            self.assertAlmostEqual(camera.pitch, pitch)
            self.assertAlmostEqual(camera.distance, distance)

    def test_argument_validation(self):
        for argv in (["--wind", "30"], ["--wind", "-1"], ["--steepness", "1.5"],
                     ["--wave-height", "0.01"], ["--wave-scale", "5.0"],
                     ["--wind-dir", "nan"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


class StartupTests(unittest.TestCase):
    def test_first_frame_finishes_before_window_creation(self):
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

        with patch("water.demo_03.ti.init"), \
             patch("water.demo_03.ti.sync", side_effect=lambda: events.append("sync")), \
             patch("water.demo_03.SunsetOceanRenderer", return_value=renderer), \
             patch("water.demo_03.ti.ui.Window", side_effect=create_window), \
             patch("builtins.print"):
            main(["--window-frames", "1", "--time", "0"])
        self.assertEqual(events, ["render", "sync", "create_window", "image", "show"])


if __name__ == "__main__":
    unittest.main()
