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
                           WAVE_SPEED_MAX, main, parse_args, simulation_budget, water_pick)
from water.demo_01 import OrbitCamera


@ti.kernel
def wave_probe(sim: ti.template(), x: ti.f32, z: ti.f32) -> ti.types.vector(3, ti.f32):
    return sim.sample(x, z)


@ti.kernel
def hit_probes(renderer: ti.template(), eye: ti.types.vector(3, ti.f32),
               directions: ti.types.ndarray(dtype=ti.f32, ndim=2),
               result: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(directions.shape[0]):
        d = ti.Vector([directions[i, 0], directions[i, 1], directions[i, 2]])
        t, n = renderer.water_hit(eye, d)
        result[i, 0], result[i, 1] = t, -d.dot(n)


@ti.kernel
def slope_probe(sim: ti.template(), x: ti.f32, z: ti.f32, footprint: ti.f32) -> ti.types.vector(2, ti.f32):
    return sim.filtered_slope(x, z, footprint)


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

    def test_stationary_uniform_height_is_preserved(self):
        self.sim.h.fill(0.01)
        self.sim.h_prev.fill(0.01)
        self.sim.advance(1.3, 0.9975, 120)
        np.testing.assert_allclose(self.sim.h.to_numpy(), 0.01, atol=1e-7)

    def test_uniform_velocity_decays_without_height_restoring_force(self):
        self.sim.h.fill(0.01)
        self.sim.h_prev.fill(0.009)
        self.sim.advance(1.3, 0.99, 60)
        expected = 0.01 + 0.001 * sum(0.99 ** k for k in range(1, 61))
        np.testing.assert_allclose(self.sim.h.to_numpy(), expected, atol=2e-6)

    def test_displacement_conserves_volume_and_initial_velocity(self):
        for x, z in [(0.0, 0.0), (3.19, 2.14)]:
            self.sim.clear()
            self.sim.inject(x, z, 0.08, 0.012)
            field = self.sim.h.to_numpy()
            self.assertLess(abs(float(field.sum())), 2e-6)
            np.testing.assert_allclose(field, self.sim.h_prev.to_numpy(), atol=1e-7)
            self.assertLess(np.abs(field).max(), 0.013)

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
        u = min(max(u, 0.0), SIM_NX - 1.0)
        v = min(max(v, 0.0), SIM_NZ - 1.0)
        i0, j0 = min(int(u), SIM_NX - 2), min(int(v), SIM_NZ - 2)
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

    def test_constructor_initializes_distributed_rain_without_manual_reset(self):
        rain = RainSystem(self.sim)
        pos = rain.pos.to_numpy()
        self.assertEqual(len(np.unique(pos, axis=0)), MAX_DROPS)
        self.assertEqual(len(np.unique(rain.slot.to_numpy())), MAX_DROPS)
        self.assertGreater(pos[:, 1].min(), 0.0)
        self.assertTrue((rain.vel.to_numpy()[:, 1] < -2.0).all())
        self.sim.clear()
        for _ in range(240):
            rain.update(SIM_DT, 0.45, 220, 1.0)
            self.sim.advance(1.3, 0.9975, 1)
        field = self.sim.h.to_numpy()
        self.assertLess(abs(float(field.mean())), 1e-6)
        self.assertGreater(np.abs(field).max(), 1e-4)
        self.assertLess(np.abs(field).max(), 0.025)

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

    def test_single_drop_injects_bounded_zero_volume_velocity(self):
        self.sim.clear()
        self.rain.reset()
        self.rain.pos[0] = [0.0, 0.001, 0.0]
        self.rain.vel[0] = [0.0, -4.0, 0.0]
        self.rain.update(SIM_DT, 0.45, 1, 1.0)
        np.testing.assert_array_equal(self.sim.h.to_numpy(), 0.0)
        velocity = -self.sim.h_prev.to_numpy() / SIM_DT
        self.assertLess(abs(float(velocity.sum())), 1e-5)
        self.assertLess(np.abs(velocity).max(), 0.5)
        self.assertGreater(np.count_nonzero(velocity), 100)
        self.sim.advance(1.3, 0.9975, 20)
        self.assertLess(np.abs(self.sim.h.to_numpy()).max(), 0.015)

    def test_single_drop_wave_front_moves_outward(self):
        self.sim.clear()
        self.rain.reset()
        self.rain.pos[0] = [0.0, 0.001, 0.0]
        self.rain.vel[0] = [0.0, -4.0, 0.0]
        self.rain.update(SIM_DT, 0.8, 1, 1)
        x = (np.arange(SIM_NX) + 0.5) * CELL_X - 3.2
        z = (np.arange(SIM_NZ) + 0.5) * CELL_X - 2.15
        distance = np.hypot(x[:, None], z[None, :])
        self.sim.advance(0.9, 0.994, 12)
        energy = self.sim.h.to_numpy() ** 2
        early = float((energy * distance).sum() / energy.sum())
        self.sim.advance(0.9, 0.994, 36)
        energy = self.sim.h.to_numpy() ** 2
        late = float((energy * distance).sum() / energy.sum())
        self.assertGreater(late, early + 0.15)

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
        # Conservative displacement starts at rest; measure a local ripple
        # rather than requiring the old height-only impulse's global motion.
        self.assertGreater(center, 0.0005)
        self.assertGreater(np.abs(first[16:48, 10:30] - updated[16:48, 10:30]).max(), 0.02)

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

    def test_flat_intersections_include_edges_and_reject_outside_pool(self):
        sim = self.renderer.sim
        sim.clear()
        sim.h.fill(0.012)
        sim.mark_dirty()
        sim.prepare_surface()
        directions = np.array([[0, -1, 0], [0.03199, -0.01, 0.02149],
                               [0.034, -0.01, 0], [0, 1, 0]], np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        result = np.zeros((4, 2), np.float32)
        hit_probes(self.renderer, np.array([0, 1.012, 0], np.float32), directions, result)
        np.testing.assert_allclose(result[:2, 0], -1 / directions[:2, 1], atol=2e-5)
        self.assertTrue((result[:2, 1] > 0).all())
        self.assertTrue((result[2:, 0] > 1e5).all())
        sim.clear()

    def test_nearest_front_intersection_matches_dense_reference_at_grazing_angles(self):
        renderer = self.renderer
        # Several crossings along shallow rays; deliberately steeper than rain.
        x = (np.arange(SIM_NX) + 0.5) * CELL_X - 3.2
        z = (np.arange(SIM_NZ) + 0.5) * CELL_X - 2.15
        h = (0.035 * np.sin(20 * x[:, None]) * np.cos(13 * z[None, :])).astype(np.float32)
        renderer.sim.h.from_numpy(h)
        renderer.sim.mark_dirty()
        renderer.sim.prepare_surface()
        camera = OrbitCamera()
        camera.preset("waterline")
        eye, forward, right, up = camera.basis()
        scale = math.tan(math.radians(camera.fov / 2))
        xx, yy = np.meshgrid(np.linspace(-0.8, 0.8, 33), np.linspace(-0.6, 0.6, 21))
        d = forward + right * xx[..., None] * 1.6 * scale + up * yy[..., None] * scale
        d = d.reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        plane_t = -eye[1] / d[:, 1]
        plane_p = eye + plane_t[:, None] * d
        select = (d[:, 1] < -0.01) & (np.abs(plane_p[:, 0]) < 2.7) & (np.abs(plane_p[:, 2]) < 1.7)
        d = d[select].astype(np.float32)
        self.assertGreater(len(d), 30)
        result = np.zeros((len(d), 2), np.float32)
        hit_probes(renderer, eye, d, result)
        for ray, actual in zip(d, result):
            lo, hi = (0.04 - eye[1]) / ray[1], (-0.04 - eye[1]) / ray[1]
            ts = np.linspace(lo, hi, 4096)
            points = eye + ts[:, None] * ray
            heights = np.array([SimulationTests._numpy_bilinear(h, (p[0] + 3.2) / CELL_X - 0.5,
                            (p[2] + 2.15) / CELL_X - 0.5) for p in points])
            residual = points[:, 1] - heights
            crossings = np.flatnonzero((residual[:-1] > 0) & (residual[1:] <= 0))
            self.assertGreater(len(crossings), 0)
            k = crossings[0]
            reference = ts[k] + (ts[k + 1] - ts[k]) * residual[k] / (residual[k] - residual[k + 1])
            self.assertAlmostEqual(float(actual[0]), reference, delta=0.0002)
            self.assertGreater(float(actual[1]), 0)
        renderer.sim.clear()

    def test_continuous_normals_and_footprint_filter(self):
        sim = self.renderer.sim
        x = (np.arange(SIM_NX) + 0.5) * CELL_X - 3.2
        pattern = np.broadcast_to((0.005 * np.sin(2 * np.pi * np.arange(SIM_NX) / 6))[:, None], (SIM_NX, SIM_NZ)).astype(np.float32)
        sim.h.from_numpy(pattern)
        sim.mark_dirty()
        sim.prepare_surface()
        boundary = x[130]
        left = np.array(slope_probe(sim, boundary - 1e-5, 0.0, 0.0))
        right = np.array(slope_probe(sim, boundary + 1e-5, 0.0, 0.0))
        self.assertLess(np.linalg.norm(left - right), 0.0002)
        near = [np.array(slope_probe(sim, v, 0.0, 0.0))[0] for v in x[100:140]]
        far = [np.array(slope_probe(sim, v, 0.0, 0.075))[0] for v in x[100:140]]
        self.assertLess(np.std(far), np.std(near) * 0.75)
        sim.clear()

    def test_caustic_cache_invalidates_clear_inject_clock_and_fine_waves(self):
        renderer = self.renderer
        camera = OrbitCamera()
        camera.preset("top")
        controls = renderer.water_controls[None]
        original = list(controls)
        controls[1] = 0
        renderer.water_controls[None] = controls
        try:
            # Keep this a cache/lighting test; other tests cover the renderer.
            calls = []
            original_trace = renderer._trace_caustics

            def record_trace(*args):
                calls.append(args)
                original_trace(*args)

            # Taichi inspects kernel attributes: ordinary functions are needed
            # here, rather than MagicMock's dynamically generated attributes.
            with patch.object(renderer, "render", new=lambda *args: None), \
                 patch.object(renderer.post, "apply", new=lambda *args: None), \
                 patch.object(renderer, "_trace_caustics", new=record_trace):
                renderer.sim.clear()
                renderer.draw(camera, 0.0)
                flat = renderer.caustic_map.to_numpy()
                renderer.draw(camera, 1.0)  # no time-dependent fine waves
                self.assertEqual(len(calls), 1)
                steps = renderer.sim.steps
                renderer.sim.inject(0, 0, 0.10, 0.012)
                renderer.draw(camera, 1.0)
                rippled = renderer.caustic_map.to_numpy()
                self.assertGreater(np.abs(rippled - flat).max(), 0.01)
                renderer.sim.clear()
                renderer.draw(camera, 1.0)
                np.testing.assert_allclose(renderer.caustic_map.to_numpy(), flat, atol=1e-5)
                self.assertEqual(renderer.sim.steps, steps)
                self.assertEqual(len(calls), 3)
                controls[1] = 1
                renderer.water_controls[None] = controls
                renderer.draw(camera, 1.0)
                first = renderer.caustic_map.to_numpy()
                renderer.draw(camera, 2.0)
                self.assertEqual(len(calls), 5)
                self.assertGreater(np.abs(renderer.caustic_map.to_numpy() - first).max(), 0.001)
        finally:
            renderer.water_controls[None] = original

    def test_argument_validation(self):
        for argv in (["--wave-speed", "5.0"], ["--wave-speed", "0.05"],
                     ["--damping", "1.5"], ["--damping", "0.5"],
                     ["--rain", "2.0"], ["--impact", "-1"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


class StartupTests(unittest.TestCase):
    def test_slow_frames_advance_real_time_instead_of_clamping_to_50ms(self):
        steps, remainder = simulation_budget(0.0, 0.1, 1.0)
        self.assertEqual(steps, 12)
        self.assertAlmostEqual(remainder, 0.0)
        steps, remainder = simulation_budget(0.0, 0.1, 2.0)
        self.assertEqual(steps, 24)
        self.assertAlmostEqual(remainder, 0.0)
        steps, remainder = simulation_budget(0.0, 10.0, 2.0)
        self.assertLessEqual(steps, 32)
        self.assertGreaterEqual(remainder, 0.0)

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
