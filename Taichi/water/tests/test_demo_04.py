"""Wake field, floating-body dynamics, and rendering checks for the harbor demo."""

from contextlib import redirect_stderr
from io import StringIO
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import taichi as ti

from water.demo_03 import OCEAN_PRESETS, apply_preset
from water.demo_04 import (BOAT_INDEX, GRAVITY, MAX_BODIES, WAKE_CELL, WAKE_DT,
                           WAKE_N, WAKE_SPEED, HarborRenderer, WakeField,
                           cursor_ray, ground_point, main, parse_args)
from water.demo_01 import OrbitCamera


@ti.kernel
def wake_probe(wake: ti.template(), x: ti.f32, z: ti.f32) -> ti.types.vector(3, ti.f32):
    return wake.sample(x, z)


@ti.kernel
def wake_velocity_probe(wake: ti.template(), x: ti.f32, z: ti.f32) -> ti.f32:
    return wake.vertical_velocity(x, z)


@ti.kernel
def foam_probe(renderer: ti.template(), p: ti.types.vector(3, ti.f32),
               clock: ti.f32, lighting: ti.f32) -> ti.types.vector(3, ti.f32):
    return renderer.foam(p, clock, lighting)


def numpy_height(table, x, z, t):
    """Host reference of the Gerstner height (envelope included, no wake)."""
    env = 1.0 + float((table["env_amp"] * np.sin(table["env_k"] @ np.array([x, z])
                                                  + table["env_phase"])).sum())
    dx, dz = table["dir"][:, 0], table["dir"][:, 1]
    f = table["k"] * (dx * x + dz * z) - table["omega"] * t + table["phase"]
    return float((table["amp"] * env * np.sin(f)).sum())


def body_state(renderer):
    pos = renderer.body_pos.to_numpy()
    vel = renderer.body_vel.to_numpy()
    return pos, vel


class WakeFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.wake = WakeField()

    def setUp(self):
        self.wake.clear()

    def test_flat_field_stays_flat(self):
        self.wake.advance(WAKE_SPEED, 0.995, 120)
        self.assertLess(float(np.abs(self.wake.h.to_numpy()).max()), 1e-6)

    def test_courant_number_is_stable(self):
        courant = WAKE_SPEED * WAKE_DT / WAKE_CELL
        self.assertLess(courant, 0.5)
        self.assertAlmostEqual(courant, 0.2, places=6)

    def test_injection_moves_no_net_water(self):
        self.wake.inject(0.0, 0.0, 0.3, 0.05)
        total = float(self.wake.h.to_numpy().sum())
        self.assertAlmostEqual(total, 0.0, delta=1e-3)

    def test_injection_propagates_and_decays(self):
        self.wake.inject(0.0, 0.0, 0.3, 0.05)
        self.wake.advance(WAKE_SPEED, 0.995, 1)
        early = np.abs(self.wake.h.to_numpy())
        self.assertGreater(early.max(), 1e-4)
        self.wake.advance(WAKE_SPEED, 0.995, 300)
        field = self.wake.h.to_numpy()
        self.assertTrue(np.isfinite(field).all())
        self.assertLess(np.abs(field).max(), 0.35 * early.max())

    def test_absorbing_boundary_eats_edge_waves(self):
        self.wake.inject(11.0, 0.0, 0.3, 0.05)
        self.wake.advance(WAKE_SPEED, 0.995, 5)
        energy0 = float((self.wake.h.to_numpy() ** 2).sum())
        self.wake.advance(WAKE_SPEED, 0.995, 300)
        energy1 = float((self.wake.h.to_numpy() ** 2).sum())
        self.assertLess(energy1, 0.3 * energy0)

    def test_sample_matches_bilinear_reference(self):
        self.wake.inject(0.37, -0.21, 0.25, 0.04)
        self.wake.advance(WAKE_SPEED, 0.995, 3)
        probe = np.array(wake_probe(self.wake, 0.41, -0.19))
        h = self.wake.h.to_numpy()
        u = (0.41 + WAKE_N * WAKE_CELL * 0.5) / WAKE_CELL - 0.5
        v = (-0.19 + WAKE_N * WAKE_CELL * 0.5) / WAKE_CELL - 0.5
        i0, j0 = int(u), int(v)
        fu, fv = u - i0, v - j0
        expected = ((h[i0, j0] * (1 - fu) + h[i0 + 1, j0] * fu) * (1 - fv)
                    + (h[i0, j0 + 1] * (1 - fu) + h[i0 + 1, j0 + 1] * fu) * fv)
        self.assertAlmostEqual(float(probe[0]), float(expected), places=5)


class BodyDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = HarborRenderer(64, 40, samples=1)

    def setUp(self):
        self.renderer.wake.clear()
        self.renderer.reset_bodies()
        self.clock = 0.0

    def step(self, count, advance=0):
        for _ in range(count):
            self.renderer.wake.advance(WAKE_SPEED, 0.995, 1)
            self.renderer.step_bodies(WAKE_DT, self.clock, advance)
            self.clock += WAKE_DT

    def test_buoy_keeps_a_small_draft_on_the_swell(self):
        # Skip the initial drop onto the surface; measure the settled ride.
        self.step(120)
        drafts = []
        for _ in range(600):
            self.renderer.wake.advance(WAKE_SPEED, 0.995, 1)
            self.renderer.step_bodies(WAKE_DT, self.clock, 0)
            self.clock += WAKE_DT
            pos = self.renderer.body_pos.to_numpy()[1]
            water = numpy_height(self.renderer.spectrum.table, pos[0], pos[2], self.clock)
            drafts.append(pos[1] - water)
        drafts = np.array(drafts)
        # The buoy rides the swell; only its draft below the local surface
        # is controlled, and that stays near the equilibrium ~6 cm.
        self.assertLess(np.abs(drafts).max(), 0.45)
        self.assertGreater(drafts.mean(), -0.35)
        self.assertLess(drafts.mean(), 0.10)

    def test_buoy_righting_moment_pulls_heel_back(self):
        # Flat-water controlled test: heel the buoy, then the restoring
        # moment must bring the roll back to zero without crossing it
        # (which would mean the torque sign is flipped and capsizes).
        self.renderer.spectrum.rebuild(8.0, 35.0, 0.05, 0.1, 0.8)
        try:
            self.step(90)
            self.renderer.body_roll[1] = 0.3
            self.step(360)
            roll = float(self.renderer.body_roll[1])
            self.assertLess(abs(roll), 0.06)
            self.assertGreater(roll, -0.06)
        finally:
            self.renderer.spectrum.rebuild(8.0, 35.0, 1.3, 0.6, 0.8)

    def test_boat_stays_afloat_and_within_speed_limit(self):
        self.renderer.set_boat_target(5.0, 5.0)
        self.step(120)
        drafts = []
        for _ in range(600):
            self.renderer.wake.advance(WAKE_SPEED, 0.995, 1)
            self.renderer.step_bodies(WAKE_DT, self.clock, 1)
            self.clock += WAKE_DT
            pos, vel = body_state(self.renderer)
            water = numpy_height(self.renderer.spectrum.table,
                                 pos[BOAT_INDEX][0], pos[BOAT_INDEX][2], self.clock)
            drafts.append(pos[BOAT_INDEX][1] - water)
        drafts = np.array(drafts)
        # Brief launches over crests are physical; the draft must stay
        # bounded and keep the hull close to the surface.
        self.assertGreater(drafts.min(), -0.55)
        self.assertLess(drafts.max(), 0.35)
        speed = float(np.hypot(vel[BOAT_INDEX][0], vel[BOAT_INDEX][2]))
        self.assertLessEqual(speed, 2.25)

    def test_boat_moves_toward_target_when_advancing(self):
        start = self.renderer.body_pos.to_numpy()[BOAT_INDEX].copy()
        self.renderer.set_boat_target(3.0, 3.0)
        self.step(240, advance=1)
        pos = self.renderer.body_pos.to_numpy()[BOAT_INDEX]
        d0 = np.hypot(start[0] - 3.0, start[2] - 3.0)
        d1 = np.hypot(pos[0] - 3.0, pos[2] - 3.0)
        self.assertLess(d1, d0 - 1.0)

    def test_boat_does_not_drift_without_advance(self):
        self.step(300, advance=0)
        pos = self.renderer.body_pos.to_numpy()[BOAT_INDEX]
        self.assertLess(np.hypot(pos[0] + 1.2, pos[2] - 1.0), 0.6)

    def test_stepping_is_deterministic(self):
        self.step(120)
        first = self.renderer.body_pos.to_numpy().copy()
        self.setUp()
        self.step(120)
        np.testing.assert_array_equal(self.renderer.body_pos.to_numpy(), first)

    def test_reset_restores_initial_layout(self):
        self.step(120, advance=1)
        self.renderer.reset_bodies()
        pos = self.renderer.body_pos.to_numpy()
        np.testing.assert_allclose(pos[BOAT_INDEX], [-1.2, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(self.renderer.body_roll[1]), 0.3, places=6)

    def test_pick_boat_hits_hull_and_misses_open_water(self):
        self.step(30)
        pos = self.renderer.body_pos.to_numpy()[BOAT_INDEX]
        hit = self.renderer.pick_boat(
            np.array([pos[0], 5.0, pos[2]], np.float32),
            np.array([0.0, -1.0, 0.0], np.float32))
        self.assertGreater(hit, 0.0)
        miss = self.renderer.pick_boat(
            np.array([0.0, 5.0, -12.0], np.float32),
            np.array([0.0, -1.0, 0.0], np.float32))
        self.assertEqual(miss, -1.0)


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = HarborRenderer(64, 40, samples=1)

    def setUp(self):
        self.renderer.wake.clear()
        self.renderer.reset_bodies()
        self.renderer.boat_forced[None] = 0

    def place_buoy_beside_boat(self, dx):
        pos = self.renderer.body_pos.to_numpy()[BOAT_INDEX].copy()
        self.renderer.body_pos[1] = np.array([pos[0] + dx, pos[1], pos[2]],
                                             dtype=np.float32)

    def pair_distance(self):
        pos = self.renderer.body_pos.to_numpy()
        return float(np.hypot(pos[BOAT_INDEX][0] - pos[1][0],
                              pos[BOAT_INDEX][2] - pos[1][2]))

    def test_overlapping_pair_is_projected_apart(self):
        self.place_buoy_beside_boat(0.7)
        before = self.pair_distance()
        self.renderer.resolve_collisions()
        after = self.pair_distance()
        self.assertGreater(after, before + 0.3)
        self.assertGreater(after, 1.2)

    def test_impulse_opposes_approach(self):
        self.place_buoy_beside_boat(-1.1)
        self.renderer.body_vel[BOAT_INDEX] = np.array([-1.2, 0.0, 0.0],
                                                      dtype=np.float32)
        self.renderer.resolve_collisions()
        vel = self.renderer.body_vel.to_numpy()
        self.assertLess(vel[1][0], -0.3)
        self.assertGreater(vel[BOAT_INDEX][0], -0.9)
        self.assertLess(vel[BOAT_INDEX][0], 0.0)

    def test_dragged_boat_shoves_float_without_slowdown(self):
        self.place_buoy_beside_boat(-1.1)
        self.renderer.body_vel[BOAT_INDEX] = np.array([-1.2, 0.0, 0.0],
                                                      dtype=np.float32)
        self.renderer.boat_forced[None] = 1
        self.renderer.resolve_collisions()
        vel = self.renderer.body_vel.to_numpy()
        np.testing.assert_allclose(vel[BOAT_INDEX], [-1.2, 0.0, 0.0], atol=1e-6)
        self.assertLess(vel[1][0], -1.0)

    def test_static_rock_pushes_buoy_out(self):
        # The foreground rock sits at (1.9, -0.12, 3.4) and pierces the
        # waterline; drop the buoy inside its footprint.
        self.renderer.body_pos[1] = np.array([1.9, 0.0, 3.7], dtype=np.float32)
        self.renderer.body_vel[1] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.renderer.resolve_collisions()
        pos = self.renderer.body_pos.to_numpy()[1]
        vel = self.renderer.body_vel.to_numpy()[1]
        self.assertGreater(pos[2], 4.1)
        self.assertGreaterEqual(vel[2], 0.0)

    def test_collisions_are_deterministic(self):
        self.place_buoy_beside_boat(0.7)
        self.renderer.resolve_collisions()
        first = self.renderer.body_pos.to_numpy().copy()
        self.setUp()
        self.place_buoy_beside_boat(0.7)
        self.renderer.resolve_collisions()
        np.testing.assert_array_equal(self.renderer.body_pos.to_numpy(), first)


class FoamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = HarborRenderer(64, 40, samples=1)

    def test_foam_rings_the_buoy_and_is_zero_far_away(self):
        self.renderer.reset_bodies()
        near = np.array(foam_probe(self.renderer, ti.Vector([-3.5 + 0.30, 0.0, 2.5]), 1.0, 0.0))
        far = np.array(foam_probe(self.renderer, ti.Vector([10.0, 0.0, 10.0]), 1.0, 0.0))
        self.assertGreater(near.sum(), 0.02)
        self.assertAlmostEqual(float(far.sum()), 0.0, places=6)

    def test_foam_fades_above_the_hull(self):
        self.renderer.reset_bodies()
        high = np.array(foam_probe(self.renderer, ti.Vector([-3.5, 2.0, 2.5]), 1.0, 0.0))
        self.assertAlmostEqual(float(high.sum()), 0.0, places=6)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = HarborRenderer(64, 40, samples=1)

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

    def test_moving_boat_changes_the_frame(self):
        camera = OrbitCamera()
        apply_preset(camera, "waterline")
        self.renderer.wake.clear()
        self.renderer.reset_bodies()
        clock = 0.0
        for _ in range(60):
            self.renderer.wake.advance(WAKE_SPEED, 0.995, 1)
            self.renderer.step_bodies(WAKE_DT, clock, 0)
            clock += WAKE_DT
        self.renderer.draw(camera, 0.5, lighting=0.0)
        before = self.renderer.image.to_numpy()
        self.renderer.set_boat_target(-4.0, -4.0)
        for _ in range(240):
            self.renderer.wake.advance(WAKE_SPEED, 0.995, 1)
            self.renderer.step_bodies(WAKE_DT, clock, 1)
            clock += WAKE_DT
        self.renderer.draw(camera, 0.5, lighting=0.0)
        after = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(after - before).mean(), 0.003)

    def test_argument_validation(self):
        for argv in (["--wind", "30"], ["--steepness", "1.5"],
                     ["--wave-height", "0.01"], ["--wave-scale", "5.0"],
                     ["--wind-dir", "nan"], ["--window-frames", "-1"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_ground_point_intersects_and_clamps(self):
        camera = OrbitCamera()
        apply_preset(camera, "top")
        center = ground_point(camera, 0.5, 0.5, 960, 600)
        self.assertIsNotNone(center)
        self.assertLessEqual(max(abs(center[0]), abs(center[1])), 14.0)
        # A ray pointing above the horizon never reaches the plane.
        apply_preset(camera, "waterline")
        eye, direction = cursor_ray(camera, 0.5, 0.0, 960, 600)
        self.assertGreater(direction[1], -1e-5)


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

        with patch("water.demo_04.ti.init"), \
             patch("water.demo_04.ti.sync", side_effect=lambda: events.append("sync")), \
             patch("water.demo_04.HarborRenderer", return_value=renderer), \
             patch("water.demo_04.ti.ui.Window", side_effect=create_window), \
             patch("builtins.print"):
            main(["--window-frames", "1", "--time", "0"])
        self.assertEqual(events, ["render", "sync", "create_window", "image", "show"])


if __name__ == "__main__":
    unittest.main()
