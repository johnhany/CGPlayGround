"""Simulation, rain, picking, and rendering checks for the rain pond demo."""

from contextlib import redirect_stderr
from io import StringIO
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import taichi as ti

from water.demo_02 import (CELL_X, MAX_DROPS, SIM_DT, SIM_NX, SIM_NZ,
                           RainPondRenderer, RainSystem, WaveSimulation,
                           WAVE_SPEED_MAX, main, parse_args, water_pick)
from water.demo_01 import OrbitCamera


@ti.kernel
def wave_probe(sim: ti.template(), x: ti.f32, z: ti.f32) -> ti.types.vector(3, ti.f32):
    return sim.sample(x, z)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.sim = WaveSimulation()

    def setUp(self):
        self.sim.clear()
        self.sim.steps = 0

    def energy(self):
        return float((self.sim.h.to_numpy() ** 2).sum())

    def test_flat_stays_flat_and_stable_under_cfl(self):
        # Zero initial data must remain exactly zero.
        self.sim.advance(1.3, 0.998, 120)
        np.testing.assert_array_equal(self.sim.h.to_numpy(), 0.0)
        # The panel wave speed range keeps the scheme inside the CFL limit.
        self.assertLess(WAVE_SPEED_MAX * SIM_DT / CELL_X, 1.0 / math.sqrt(2.0) - 0.05)

    def test_injection_propagates_and_reaches_the_wall(self):
        self.sim.inject(0.0, 0.0, 0.08, 0.06)
        self.assertGreater(self.energy(), 0.0)
        # The wave front must travel outward and reach the far wall region.
        for _ in range(400):
            self.sim.advance(1.3, 0.9999, 1)
        field = np.abs(self.sim.h.to_numpy())
        near_wall = field[0:4, :].max()
        self.assertGreater(near_wall, 1e-5)
        self.assertTrue(np.isfinite(field).all())

    def test_damping_decays_energy(self):
        self.sim.inject(0.0, 0.0, 0.08, 0.06)
        self.sim.advance(1.3, 0.998, 30)
        early = self.energy()
        self.sim.advance(1.3, 0.998, 600)
        late = self.energy()
        self.assertLess(late, early * 0.5)

    def test_noise_remains_bounded_over_long_runs(self):
        rng = np.random.default_rng(3)
        self.sim.h.from_numpy((rng.normal(0, 0.01, (SIM_NX, SIM_NZ))).astype(np.float32))
        self.sim.h_prev.from_numpy(self.sim.h.to_numpy() * 0.9)
        for _ in range(60):
            self.sim.advance(1.8, 0.9999, 10)
        field = self.sim.h.to_numpy()
        self.assertTrue(np.isfinite(field).all())
        self.assertLess(np.abs(field).max(), 1.0)

    def test_bilinear_sample_matches_numpy(self):
        rng = np.random.default_rng(5)
        pattern = rng.normal(0, 0.02, (SIM_NX, SIM_NZ)).astype(np.float32)
        self.sim.h.from_numpy(pattern)
        interior = [(0.0, 0.0), (1.234, -0.777), (-2.91, 1.93)]
        for x, z in interior:
            u = (x + 3.2) / CELL_X - 0.5
            v = (z + 2.15) / CELL_X - 0.5
            expected = self._numpy_bilinear(pattern, u, v)
            sample = np.array(wave_probe(self.sim, x, z))
            self.assertAlmostEqual(sample[0], expected, places=5)
            # Gradient equals finite differences of the sampled height.
            eps = 0.0005
            dx = (np.array(wave_probe(self.sim, x + eps, z))[0]
                  - np.array(wave_probe(self.sim, x - eps, z))[0]) / (2 * eps)
            dz = (np.array(wave_probe(self.sim, x, z + eps))[0]
                  - np.array(wave_probe(self.sim, x, z - eps))[0]) / (2 * eps)
            np.testing.assert_allclose(sample[1:], [dx, dz], rtol=5e-3, atol=2e-3)
        # Near the boundary the sample clamps, which matches numpy exactly.
        for x, z in [(3.19, -2.14), (-3.199, 2.149)]:
            u = (x + 3.2) / CELL_X - 0.5
            v = (z + 2.15) / CELL_X - 0.5
            sample = np.array(wave_probe(self.sim, x, z))
            self.assertAlmostEqual(sample[0], self._numpy_bilinear(pattern, u, v), places=5)

    @staticmethod
    def _numpy_bilinear(field, u, v):
        u = min(max(u, 0.0), SIM_NX - 1.001)
        v = min(max(v, 0.0), SIM_NZ - 1.001)
        i0, j0 = int(u), int(v)
        fu, fv = u - i0, v - j0
        h00, h10 = field[i0, j0], field[i0 + 1, j0]
        h01, h11 = field[i0, j0 + 1], field[i0 + 1, j0 + 1]
        return float((h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv)


class RainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.sim = WaveSimulation()
        cls.rain = RainSystem(cls.sim, seed=17)
        cls.rain.reset()

    def test_rain_is_deterministic_and_contained(self):
        reference = None
        for run in range(2):
            self.sim.clear()
            self.rain.reset()
            for _ in range(240):
                self.rain.update(1.0 / 60.0, 0.55, 220, 1.0)
                self.sim.advance(1.3, 0.998, 2)
            positions = self.rain.pos.to_numpy()
            field = self.sim.h.to_numpy()
            self.assertTrue(np.isfinite(positions).all())
            self.assertTrue(np.isfinite(field).all())
            self.assertGreater(np.abs(field).max(), 1e-5)
            self.assertGreaterEqual(positions[:, 1].min(), 0.0)
            self.assertLessEqual(positions[:, 1].max(), 3.3)
            self.assertLessEqual(np.abs(positions[:, 0]).max(), 3.2)
            self.assertLessEqual(np.abs(positions[:, 2]).max(), 2.15)
            if reference is None:
                reference = (positions, field)
            else:
                np.testing.assert_allclose(positions, reference[0], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose(field, reference[1], rtol=1e-4, atol=1e-7)

    def test_zero_active_drops_leave_the_surface_calm(self):
        self.sim.clear()
        self.rain.reset()
        self.rain.update(1.0 / 60.0, 0.55, 0, 1.0)
        self.sim.advance(1.3, 0.998, 10)
        np.testing.assert_array_equal(self.sim.h.to_numpy(), 0.0)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = RainPondRenderer(64, 40)
        cls.renderer.rain.reset()
        cls.renderer.rain.count[None] = 160
        for _ in range(120):
            cls.renderer.rain.update(1.0 / 60.0, 0.55, 160, 1.0)
            cls.renderer.sim.advance(1.3, 0.998, 2)

    def test_presets_produce_finite_nonblank_frames(self):
        camera = OrbitCamera()
        for preset in ("overview", "waterline", "top"):
            camera.preset(preset)
            self.renderer.draw(camera, 1.0)
            frame = self.renderer.image.to_numpy()
            self.assertTrue(np.isfinite(frame).all(), preset)
            self.assertGreater(frame.std(), 0.04, preset)
            self.assertGreaterEqual(frame.min(), 0.0)
            self.assertLessEqual(frame.max(), 1.0)

    def test_paused_state_is_deterministic(self):
        camera = OrbitCamera()
        camera.preset("top")
        self.renderer.sim.inject(0.0, 0.0, 0.08, 0.06)
        for _ in range(10):
            self.renderer.sim.advance(1.3, 0.998, 2)
        self.renderer.draw(camera, 1.0)
        first = self.renderer.image.to_numpy()
        self.renderer.draw(camera, 1.0)
        np.testing.assert_array_equal(first, self.renderer.image.to_numpy())
        # Advancing the simulation changes the picture; compare the pool area.
        self.renderer.sim.advance(1.3, 0.998, 90)
        self.renderer.draw(camera, 1.0)
        updated = self.renderer.image.to_numpy()
        center = np.mean(np.abs(first[16:48, 10:30] - updated[16:48, 10:30]))
        self.assertGreater(center, 0.0015)

    def test_caustics_follow_the_simulation(self):
        renderer = self.renderer
        controls = renderer.water_controls[None]
        controls[1] = 0.0  # detail normals off: a calm surface focuses uniformly
        renderer.water_controls[None] = controls
        try:
            renderer.sim.clear()
            renderer.trace_caustics(0.0, 1.0, 0)
            flat = renderer.caustic_raw.to_numpy()[16:-16, 16:-16]
            sun_cosine = 0.95 / np.linalg.norm([-0.50, 0.95, -0.57])
            f0 = ((1.333 - 1) / (1.333 + 1)) ** 2
            expected = 1 - (f0 + (1 - f0) * (1 - sun_cosine) ** 5)
            self.assertAlmostEqual(flat.mean(), expected, places=3)
            self.assertLess(flat.std(), 0.001)
            renderer.sim.inject(0.0, 0.0, 0.08, 0.06)
            for _ in range(30):
                renderer.sim.advance(1.3, 0.998, 2)
            renderer.trace_caustics(0.0, 1.0, 0)
            renderer.filter_caustics()
            rippled = renderer.caustic_map.to_numpy()
            self.assertGreater(rippled.std(), 0.05)
            self.assertTrue(np.isfinite(rippled).all())
            self.assertGreaterEqual(rippled.min(), 0.0)
            self.assertLessEqual(rippled.max(), 8.0)
        finally:
            controls[1] = 1.0
            renderer.water_controls[None] = controls

    def test_water_pick_round_trips_through_the_camera(self):
        camera = OrbitCamera()
        for preset in ("overview", "waterline", "top"):
            camera.preset(preset)
            for x, z in [(0.0, 0.0), (1.2, -0.8)]:
                eye, forward, right, up = camera.basis()
                raw = np.array([x, 0.0, z]) - eye
                # Invert the perspective projection: ray = forward + right*sx + up*sy.
                scale = math.tan(math.radians(camera.fov / 2.0))
                depth = raw.dot(forward)
                horizontal = raw.dot(right) / (depth * self.renderer.width / self.renderer.height * scale)
                vertical = raw.dot(up) / (depth * scale)
                u = (horizontal + 1.0) / 2.0
                v = 1.0 - (vertical + 1.0) / 2.0
                picked = water_pick(camera, u, v, self.renderer.width, self.renderer.height)
                self.assertIsNotNone(picked, (preset, x, z))
                self.assertAlmostEqual(picked[0], x, delta=0.02)
                self.assertAlmostEqual(picked[1], z, delta=0.02)

    def test_argument_validation(self):
        for argv in (["--wave-speed", "5.0"], ["--wave-speed", "0.05"],
                     ["--damping", "1.5"], ["--damping", "0.5"],
                     ["--rain", "2.0"], ["--impact", "-1"]):
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

        with patch("water.demo_02.ti.init"), \
             patch("water.demo_02.ti.sync", side_effect=lambda: events.append("sync")), \
             patch("water.demo_02.RainPondRenderer", return_value=renderer), \
             patch("water.demo_02.ti.ui.Window", side_effect=create_window), \
             patch("builtins.print"):
            main(["--window-frames", "1", "--time", "0"])
        self.assertEqual(events, ["render", "sync", "create_window", "image", "show"])


if __name__ == "__main__":
    unittest.main()
