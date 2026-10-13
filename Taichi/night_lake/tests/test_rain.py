"""Rain stage checks: CPU advection mirror, particle conservation, landing
site behaviour of the event-driven spawn, water/land landing routing,
near-far streak brightness ordering, splash splats, and moon-disk
preservation under the rain curtain (no GUI)."""

import math
import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import PRODUCTION_EYE, NightLakeScene
from night_lake.rain import (MAX_SPLASH_DROPS, MAX_STREAKS, SPAWN_MAX_X,
                             SPAWN_MAX_Z, SPAWN_MIN_X, SPAWN_MIN_Z,
                             RainStreaks, SplashBursts)
from night_lake.terrain import terrain_height_np


def mirror_advance(streaks, dt):
    """Replay RainStreaks.advance on private copies, step for step."""
    pos = streaks.pos.copy()
    prev = streaks.prev.copy()
    alive = streaks.alive.copy()
    indices = np.nonzero(alive)[0]
    if len(indices):
        prev[indices] = pos[indices]
        pos[indices] = pos[indices] + streaks.vel[indices] * dt
        xs, zs = pos[indices, 0], pos[indices, 2]
        ground = np.maximum(terrain_height_np(xs.astype(np.float32),
                                              zs.astype(np.float32)), 0.0)
        landed = indices[pos[indices, 1] <= ground]
        alive[landed] = False
    return pos, prev, alive


class AdvectionMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_advance_matches_cpu_reference(self):
        # Hand-built state: every drop sits high enough that none lands in
        # the single mirrored step, and spawn() is never invoked, so the
        # replay must agree bit for bit with no birth interference.
        streaks = RainStreaks(64, 36)
        rng = np.random.default_rng(4)
        alive = np.zeros(MAX_STREAKS, dtype=bool)
        alive[: 37] = True
        streaks.alive = alive
        streaks.pos[:, 0] = rng.uniform(-20.0, 20.0, MAX_STREAKS)
        streaks.pos[:, 1] = rng.uniform(6.0, 11.0, MAX_STREAKS)
        streaks.pos[:, 2] = rng.uniform(5.0, 55.0, MAX_STREAKS)
        streaks.prev = streaks.pos - rng.uniform(0.1, 0.2, (MAX_STREAKS, 3))
        streaks.vel[:, 0] = rng.uniform(-0.1, 0.1, MAX_STREAKS)
        streaks.vel[:, 1] = -rng.uniform(7.5, 10.5, MAX_STREAKS)
        streaks.vel[:, 2] = rng.uniform(-0.1, 0.1, MAX_STREAKS)
        pos0, prev0, alive0 = streaks.pos.copy(), streaks.prev.copy(), streaks.alive.copy()
        spawned0 = streaks.spawned
        streaks.advance(1.0 / 60.0, 0.2)
        streaks2 = RainStreaks(64, 36)
        streaks2.pos, streaks2.prev, streaks2.alive = pos0, prev0, alive0
        streaks2.vel = streaks.vel
        pos1, prev1, alive1 = mirror_advance(streaks2, 1.0 / 60.0)
        np.testing.assert_array_equal(streaks.pos, pos1)
        np.testing.assert_array_equal(streaks.prev, prev1)
        np.testing.assert_array_equal(streaks.alive, alive1)
        self.assertEqual(streaks.spawned, spawned0)
        self.assertGreater(int(alive0.sum()), 0)


class ConservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_spawned_equals_landed_plus_alive(self):
        streaks = RainStreaks(64, 36)
        for _ in range(180):
            streaks.spawn(60.0, 0.1, 1.0 / 60.0)
            streaks.advance(1.0 / 60.0, 0.1)
            self.assertEqual(streaks.spawned,
                             streaks.landed + int(streaks.alive.sum()))
        self.assertGreater(streaks.spawned, 0)


class LandingSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_landings_stay_inside_the_birth_volume_footprint(self):
        # With zero wind a streak falls vertically, so every landing must
        # stay inside the x/z box the streaks were born in.
        streaks = RainStreaks(64, 36)
        landings = []
        for _ in range(240):
            streaks.spawn(60.0, 0.0, 1.0 / 60.0)
            landings.extend(streaks.advance(1.0 / 60.0, 0.0))
        self.assertGreater(len(landings), 0)
        points = np.asarray(landings)
        self.assertGreaterEqual(points[:, 0].min(), SPAWN_MIN_X)
        self.assertLessEqual(points[:, 0].max(), SPAWN_MAX_X)
        self.assertGreaterEqual(points[:, 1].min(), SPAWN_MIN_Z)
        self.assertLessEqual(points[:, 1].max(), SPAWN_MAX_Z)

    def test_wind_shifts_landings_downwind(self):
        # The birth volume is symmetric, so a nonzero mean landing
        # displacement along x has the sign of the wind control.
        def mean_landing_x(wind):
            streaks = RainStreaks(64, 36)
            landings = []
            for _ in range(240):
                streaks.spawn(60.0, wind, 1.0 / 60.0)
                landings.extend(streaks.advance(1.0 / 60.0, wind))
            return float(np.asarray(landings)[:, 0].mean())

        self.assertGreater(mean_landing_x(2.0), 0.05)
        self.assertLess(mean_landing_x(-2.0), -0.05)


class LandingRoutingTests(unittest.TestCase):
    """Scene-level split: water landings feed the ripple field only."""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_water_landings_inject_and_land_landings_do_not(self):
        scene = NightLakeScene(160, 90)
        # Hand-place one streak over deep water and one over the beach,
        # both a single step above their surfaces.
        scene.streaks.alive[:2] = True
        scene.streaks.pos[0] = (0.0, 0.05, 30.0)
        scene.streaks.pos[1] = (0.0, 0.05, 4.0)
        scene.streaks.prev = scene.streaks.pos.copy()
        scene.streaks.vel[:2] = (0.0, -10.0, 0.0)
        self.assertGreater(terrain_height_np(np.float32(0.0), np.float32(4.0)),
                           0.0)
        count0, _ = scene.ripples.stats()
        scene._simulate_rain(0.0, 0.0, 0.0, PRODUCTION_EYE, 1.0 / 60.0,
                             spawn=False, want_splash=False)
        count1, energy1 = scene.ripples.stats()
        self.assertEqual(count1 - count0, 1)
        self.assertGreater(energy1, 0.0)


class StreakBrightnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.eye = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cls.forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cls.right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cls.up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cls.scale = math.tan(math.radians(60.0 / 2.0))

    def _project(self, point, width, height):
        delta = point - self.eye
        depth = float(delta.dot(self.forward))
        px = 0.5 * width + delta.dot(self.right) / depth * 0.5 * height / self.scale
        py = 0.5 * height + delta.dot(self.up) / depth * 0.5 * height / self.scale
        return int(px), int(py), depth

    def test_near_streak_outshines_far_streak(self):
        width, height = 96, 54
        streaks = RainStreaks(width, height)
        # Two hand-placed streaks with identical gain: one at z = 6, one
        # at z = 60. The far one must fade without disappearing outright.
        streaks.alive[:2] = True
        streaks.pos[0] = (0.0, 3.0, 6.0)
        streaks.pos[1] = (0.0, 3.0, 60.0)
        streaks.prev[0] = streaks.pos[0] + (0.0, 0.05, 0.0)
        streaks.prev[1] = streaks.pos[1] + (0.0, 0.05, 0.0)
        streaks.gain[:2] = 1.0
        streaks.spawned = 2
        hit_depth = ti.field(ti.f32, shape=(width, height))
        hit_depth.fill(1e6)
        streaks.upload(1.0)
        streaks.splat(ti.Vector(self.eye.astype(np.float32)),
                      ti.Vector(self.forward.astype(np.float32)),
                      ti.Vector(self.right.astype(np.float32)),
                      ti.Vector(self.up.astype(np.float32)),
                      self.scale, 1.0, 140.0, hit_depth)
        accum = streaks.accum.to_numpy()
        weight = streaks.weight.to_numpy()
        self.assertGreater(weight.sum(), 0.0)
        px0, py0, _ = self._project(streaks.pos[0], width, height)
        px1, py1, _ = self._project(streaks.pos[1], width, height)
        near_brightness = accum[px0, py0].sum()
        far_brightness = accum[px1, py1].sum()
        self.assertGreater(near_brightness, 2.0 * far_brightness)
        self.assertGreater(far_brightness, 0.0)

    def test_streak_behind_the_scene_surface_is_occluded(self):
        width, height = 96, 54
        streaks = RainStreaks(width, height)
        streaks.alive[0] = True
        streaks.pos[0] = (0.0, 3.0, 6.0)
        streaks.prev[0] = streaks.pos[0] + (0.0, 0.05, 0.0)
        streaks.gain[0] = 1.0
        # A scene surface nearer than the streak along every pixel ray
        # must suppress the splat entirely.
        hit_depth = ti.field(ti.f32, shape=(width, height))
        hit_depth.fill(2.0)
        streaks.upload(1.0)
        streaks.splat(ti.Vector(self.eye.astype(np.float32)),
                      ti.Vector(self.forward.astype(np.float32)),
                      ti.Vector(self.right.astype(np.float32)),
                      ti.Vector(self.up.astype(np.float32)),
                      self.scale, 1.0, 140.0, hit_depth)
        self.assertEqual(streaks.weight.to_numpy().sum(), 0.0)


class SplashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_burst_splat_marks_the_accumulation_buffer(self):
        width, height = 96, 54
        streaks = RainStreaks(width, height)
        splash = SplashBursts()
        splash.spawn_burst(0.0, 0.0, 40.0)
        splash.advance(1.0 / 120.0)
        splash.upload()
        v3 = ti.types.vector(3, ti.f32)
        eye = ti.Vector([0.0, 1.0, 0.0])
        splash.splat(streaks.accum, streaks.weight, width, height,
                     eye, ti.Vector([0.0, 0.0, 1.0]), ti.Vector([1.0, 0.0, 0.0]),
                     ti.Vector([0.0, 1.0, 0.0]), 0.577, 1.0)
        self.assertGreater(streaks.weight.to_numpy().sum(), 0.0)

    def test_pool_capacity_is_respected(self):
        splash = SplashBursts()
        for _ in range(50):
            splash.spawn_burst(0.0, 0.0, 40.0)
        self.assertLessEqual(int(splash.alive.sum()), MAX_SPLASH_DROPS)


class RainFrameTests(unittest.TestCase):
    """Image-level checks on small frames drawn without a GUI."""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.camera = FixedCamera()

    def _fresh_scene(self):
        scene = NightLakeScene(160, 90)
        scene.water_controls[None] = [1.0, 1.0, 60.0, 0.06]
        scene.rain_controls[None] = [0.0, 0.0, 0.0, 1.0]
        scene.warm_rain(1.0)
        return scene

    def _draw(self, scene, streak=0.0, curtain=0.0):
        scene.rain_controls[None] = [streak, 0.0, curtain, 1.0]
        scene.draw(self.camera, 2.0)
        return scene.image.to_numpy()

    def test_streaks_change_the_frame(self):
        frame = self._draw(self._fresh_scene(), streak=1.0)
        reference = self._draw(self._fresh_scene(), streak=0.0)
        diff = np.abs(frame - reference).sum(axis=2)
        self.assertTrue(np.isfinite(frame).all())
        self.assertGreater(int((diff > 0.02).sum()), 50)

    def test_curtain_leaves_the_moon_disk_intact(self):
        covered = self._draw(self._fresh_scene(), curtain=1.5)
        clear = self._draw(self._fresh_scene(), curtain=0.0)
        # Taichi images are x-first; the moon sits in the upper right,
        # so high x and high y in array coordinates.
        region = (slice(90, 160), slice(45, 90), slice(None))
        moon_covered = covered[region].max(axis=(0, 1)).sum()
        moon_clear = clear[region].max(axis=(0, 1)).sum()
        self.assertGreater(moon_covered, 0.85 * moon_clear)


if __name__ == "__main__":
    unittest.main()
