"""Terrain, shallow-water solver, and shore rendering checks for demo 06."""

from contextlib import redirect_stderr
from io import StringIO
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import taichi as ti

from water.demo_06 import (MAX_BED_SLOPE, MARCH_GROWTH, MARCH_MIN_STEP,
                           SEA_DX, SEA_HALF, SEA_N, SHORE_PRESETS,
                           SHORE_PRESETS_SEA, ShallowSea, ShoreCamera,
                           ShoreRenderer, SIM_DT, TERRAIN_Y_HI, TERRAIN_Y_LO,
                           apply_preset, incident_eta, main, parse_args,
                           reef_factor, sea_parameters, shore_terrain,
                           shore_terrain_grad, terrain_march_step, water_pick)


@ti.kernel
def terrain_probe(x: ti.f32, z: ti.f32) -> ti.types.vector(3, ti.f32):
    grad = shore_terrain_grad(x, z)
    return ti.Vector([shore_terrain(x, z), grad.x, grad.y])


@ti.kernel
def reef_probe(x: ti.f32, z: ti.f32) -> ti.f32:
    return reef_factor(x, z)


@ti.kernel
def step_probe(error: ti.f32, current: ti.f32, dir_y: ti.f32, dir_h: ti.f32) -> ti.f32:
    return terrain_march_step(error, current, dir_y, dir_h)


@ti.kernel
def incident_probe(x: ti.f32, z: ti.f32, t: ti.f32,
                   amplitude: ti.f32, period: ti.f32) -> ti.f32:
    return incident_eta(x, z, t, amplitude, period)


@ti.kernel
def sample_probe(sea: ti.template(), x: ti.f32, z: ti.f32, t: ti.f32) -> ti.types.vector(4, ti.f32):
    return sea.sample(x, z, t)


@ti.kernel
def height_probe(renderer: ti.template(), x: ti.f32, z: ti.f32, t: ti.f32) -> ti.f32:
    return renderer.wave_height(x, z, t)


@ti.kernel
def surface_probe(renderer: ti.template(), x: ti.f32, z: ti.f32, t: ti.f32) -> ti.types.vector(4, ti.f32):
    y, normal = renderer.wave_surface(x, z, t)
    return ti.Vector([y, normal.x, normal.y, normal.z])


@ti.kernel
def bounds_probe(renderer: ti.template()) -> ti.types.vector(2, ti.f32):
    return renderer.wave_bounds()


@ti.kernel
def hit_probe(renderer: ti.template(), origin: ti.types.vector(3, ti.f32),
              direction: ti.types.vector(3, ti.f32), clock: ti.f32) -> ti.types.vector(4, ti.f32):
    t, n, h = renderer.water_hit(origin, direction, clock)
    return ti.Vector([t, n.x, n.y, n.z])


def numpy_terrain(grid_x, grid_z):
    bounded_x = np.clip(grid_x, -SEA_HALF, SEA_HALF)
    bounded_x = np.where(grid_x > SEA_HALF, SEA_HALF + 8 * np.tanh((grid_x - SEA_HALF) / 8), bounded_x)
    bounded_x = np.where(grid_x < -SEA_HALF, -SEA_HALF + 8 * np.tanh((grid_x + SEA_HALF) / 8), bounded_x)
    bed = (0.045 * (bounded_x - 4.0)
           + 0.16 * np.sin(0.031 * grid_z + 0.9)
           + 0.10 * np.sin(0.083 * grid_z + 3.94)
           + 0.06 * np.sin(0.017 * grid_z + 4.32))
    for cx, cz, top, sigma in ((-18.0, -10.0, 3.0, 1.6), (-6.0, 8.0, 2.2, 1.2),
                               (-30.0, 18.0, 2.3, 2.0), (20.0, -20.0, 0.9, 1.4)):
        bed = bed + top * np.exp(-((grid_x - cx) ** 2 + (grid_z - cz) ** 2) / sigma ** 2) * (1 + 0.06 * np.sin(1.8 * (grid_x - cx)) * np.sin(1.3 * (grid_z - cz)))
    return bed


def numpy_incident(x, z, t, amplitude, period, gravity=9.81, half=SEA_HALF):
    """Python mirror of demo_06.incident_eta."""
    period = max(period, 1.0)
    ramp = min(1.0, t / period)
    omega1 = 2.0 * math.pi / period
    omega2 = omega1 * 1.71
    k1 = omega1 / math.sqrt(gravity * 3.0)
    k2 = omega2 / math.sqrt(gravity * 3.0)
    kz1 = 0.105 * k1
    kz2 = -0.141 * k2
    drift = np.sin(0.045 * z + 2.0)
    a1 = amplitude * 0.72 * (1.0 + 0.20 * drift)
    a2 = amplitude * 0.28 * (1.0 - 0.50 * drift)
    phase1 = omega1 * t - kz1 * z - k1 * (x + half)
    phase2 = omega2 * t + 1.7 - kz2 * z - k2 * (x + half)
    return ramp * (a1 * np.sin(phase1) + a2 * np.sin(phase2))


def node_axis():
    return -SEA_HALF + np.arange(SEA_N) * SEA_DX


class TerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_gradient_matches_finite_differences(self):
        eps = 1.0e-3
        for x, z in ((0.0, 0.0), (-18.0, -10.0), (20.5, -19.5), (7.3, 12.1)):
            height, gx, gz = (float(v) for v in terrain_probe(x, z))
            fx = (float(terrain_probe(x + eps, z)[0]) - float(terrain_probe(x - eps, z)[0])) / (2 * eps)
            fz = (float(terrain_probe(x, z + eps)[0]) - float(terrain_probe(x, z - eps)[0])) / (2 * eps)
            self.assertAlmostEqual(gx, fx, delta=2e-3)
            self.assertAlmostEqual(gz, fz, delta=2e-3)
            self.assertTrue(math.isfinite(height))

    def test_shoreline_zero_crossing_near_x4_at_z0(self):
        crossings = []
        xs = np.arange(-4.0, 12.0, 0.25)
        heights = [float(terrain_probe(float(x), 0.0)[0]) for x in xs]
        for a, b, ha, hb in zip(xs[:-1], xs[1:], heights[:-1], heights[1:]):
            if ha == 0.0 or ha * hb < 0.0:
                crossings.append(0.5 * (a + b))
        self.assertTrue(any(3.5 < c < 4.5 for c in crossings), crossings)

    def test_reef_centers_protrude_above_still_level(self):
        for cx, cz, top, _sigma in ((-18.0, -10.0, 3.0, 1.6),
                                    (-6.0, 8.0, 2.2, 1.2),
                                    (-30.0, 18.0, 2.3, 2.0)):
            self.assertGreater(float(terrain_probe(cx, cz)[0]), 0.5)
        # The beach reef also sits above the sand around it.
        self.assertGreater(float(terrain_probe(20.0, -20.0)[0]),
                           float(terrain_probe(24.0, -20.0)[0]))
        self.assertGreaterEqual(float(reef_probe(-18.0, -10.0)), 1.0)


class TerrainMarchTests(unittest.TestCase):
    """The terrain marcher must find the first crossing, not stride over a
    reef summit and bisect the far flank (the clipped-peak artifact)."""

    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    @staticmethod
    def cpu_step(error, current, dir_y, dir_h):
        decay = -dir_y + MAX_BED_SLOPE * dir_h
        step = 0.9 * error / decay
        return max(min(step, MARCH_MIN_STEP + MARCH_GROWTH * current),
                   MARCH_MIN_STEP)

    def test_kernel_step_matches_cpu_formula(self):
        for error in (0.001, 0.1, 1.0, 5.0, 50.0):
            for current in (0.01, 1.0, 15.0, 60.0):
                for dir_y, dir_h in ((-0.95, 0.31), (-0.3, 0.95),
                                     (-0.08, 0.997), (-0.5, 0.87)):
                    got = float(step_probe(error, current, dir_y, dir_h))
                    want = self.cpu_step(error, current, dir_y, dir_h)
                    self.assertAlmostEqual(got, want, delta=1e-6,
                                           msg=(error, current, dir_y, dir_h))

    def numpy_march(self, origin, direction):
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        if abs(direction[1]) < 1e-8:
            return None
        t_hi = (TERRAIN_Y_HI - origin[1]) / direction[1]
        t_lo = (TERRAIN_Y_LO - origin[1]) / direction[1]
        near, far = max(1e-3, min(t_hi, t_lo)), max(t_hi, t_lo)
        if near > far:
            return None
        current = near
        p = origin + direction * current
        bed = float(numpy_terrain(p[0], p[2]))
        if p[1] <= bed:
            return current
        dir_h = math.hypot(direction[0], direction[2])
        hit, count = None, 0
        while hit is None and current < far and count < 160:
            step = self.cpu_step(float(p[1] - bed), current, direction[1], dir_h)
            nxt = min(current + step, far)
            p = origin + direction * nxt
            bed = float(numpy_terrain(p[0], p[2]))
            if p[1] <= bed:
                lo_t, hi_t = current, nxt
                for _ in range(10):
                    mid = 0.5 * (lo_t + hi_t)
                    pm = origin + direction * mid
                    if pm[1] <= float(numpy_terrain(pm[0], pm[2])):
                        hi_t = mid
                    else:
                        lo_t = mid
                hit = 0.5 * (lo_t + hi_t)
            else:
                current = nxt
            count += 1
        return hit

    def numpy_dense_first_crossing(self, origin, direction, step=2e-4):
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        t_hi = (TERRAIN_Y_HI - origin[1]) / direction[1]
        t_lo = (TERRAIN_Y_LO - origin[1]) / direction[1]
        near, far = max(1e-3, min(t_hi, t_lo)), max(t_hi, t_lo)
        if near > far:
            return None
        ts = np.arange(near, far, step)
        ys = origin[1] + direction[1] * ts
        beds = numpy_terrain(origin[0] + direction[0] * ts,
                             origin[2] + direction[2] * ts)
        below = np.nonzero(ys <= beds)[0]
        return None if below.size == 0 else float(ts[below[0]])

    def test_bed_slope_bound_covers_steepest_flank(self):
        worst = 0.0
        for cx, cz, _top, sigma in ((-18.0, -10.0, 3.0, 1.6),
                                    (-6.0, 8.0, 2.2, 1.2),
                                    (-30.0, 18.0, 2.3, 2.0),
                                    (20.0, -20.0, 0.9, 1.4)):
            xs = np.linspace(cx - 3.0 * sigma, cx + 3.0 * sigma, 361)
            zs = np.linspace(cz - 3.0 * sigma, cz + 3.0 * sigma, 361)
            bed = numpy_terrain(xs[:, None], zs[None, :])
            gx, gz = np.gradient(bed, xs, zs)
            worst = max(worst, float(np.max(np.hypot(gx, gz))))
        self.assertLess(worst, MAX_BED_SLOPE)

    def test_march_finds_first_reef_crossing_from_tilted_rays(self):
        # Origins chosen to stride the reef from several distances and
        # elevations; rays cut just below each summit so a far-flank
        # bisection lands metres behind the true first crossing.
        for cx, cz in ((-18.0, -10.0), (-6.0, 8.0), (-30.0, 18.0)):
            summit_y = float(numpy_terrain(cx, cz))
            target = np.array([cx, summit_y - 0.15, cz])
            for origin in ((8.0, 4.5, -26.0), (-42.0, 1.2, -12.0),
                           (-16.0, 5.5, 24.0), (6.0, 2.0, 22.0)):
                origin = np.array(origin)
                direction = target - origin
                direction /= np.linalg.norm(direction)
                marched = self.numpy_march(origin, direction)
                dense = self.numpy_dense_first_crossing(origin, direction)
                self.assertIsNotNone(dense, (cx, cz, origin.tolist()))
                self.assertIsNotNone(marched, (cx, cz, origin.tolist()))
                self.assertAlmostEqual(marched, dense, delta=5e-3)
                hit = origin + direction * marched
                self.assertLess(math.hypot(hit[0] - cx, hit[2] - cz), 1.2)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_initial_state_is_still_water(self):
        sea = ShallowSea()
        axis = node_axis()
        bed = numpy_terrain(axis[:, None], axis[None, :])
        np.testing.assert_allclose(sea.h.to_numpy(), np.maximum(-bed, 0.0), atol=1e-6)
        for field in (sea.qx, sea.qz, sea.foam, sea.wetness):
            np.testing.assert_array_equal(field.to_numpy(), np.zeros((SEA_N, SEA_N), np.float32))
        self.assertEqual(sea.time, 0.0)

    def test_twenty_seconds_of_surf_stays_bounded(self):
        sea = ShallowSea()
        sea.set_wave(0.30, 10.0)
        sea.advance(int(20.0 / SIM_DT))
        h, qx, qz = sea.h.to_numpy(), sea.qx.to_numpy(), sea.qz.to_numpy()
        for field in (h, qx, qz, sea.foam.to_numpy(), sea.wetness.to_numpy()):
            self.assertTrue(np.isfinite(field).all())
        self.assertGreaterEqual(h.min(), 0.0)
        self.assertLess(h.max(), 6.0)
        self.assertGreaterEqual(sea.foam.to_numpy().min(), 0.0)
        self.assertLessEqual(sea.foam.to_numpy().max(), 1.0)
        self.assertGreaterEqual(sea.wetness.to_numpy().min(), 0.0)
        self.assertLessEqual(sea.wetness.to_numpy().max(), 1.0)
        self.assertGreater(sea.foam.to_numpy().max(), 0.05)

    def test_waves_reach_the_beach(self):
        sea = ShallowSea()
        sea.set_wave(0.30, 10.0)
        sea.advance(int(30.0 / SIM_DT))
        axis = node_axis()
        wet_x = axis[sea.wetness.to_numpy().max(axis=1) > 0.99]
        self.assertGreater(wet_x.max(), 5.0)
        # Foam traces exist somewhere near the waterline band.
        self.assertGreater(sea.foam.to_numpy().max(), 0.1)

    def test_zero_amplitude_stays_quiet_offshore(self):
        sea = ShallowSea()
        sea.set_wave(0.0, 10.0)
        sea.advance(int(10.0 / SIM_DT))
        axis = node_axis()
        bed = numpy_terrain(axis[:, None], axis[None, :])
        eta = sea.h.to_numpy() + bed
        offshore = axis < -40.0
        self.assertLess(np.abs(eta[offshore, :]).max(), 0.02)

    def test_advance_is_deterministic(self):
        runs = []
        for _ in range(2):
            sea = ShallowSea()
            sea.set_wave(0.30, 10.0)
            sea.advance(int(4.0 / SIM_DT))
            runs.append(sea)
        first, second = runs
        np.testing.assert_array_equal(first.h.to_numpy(), second.h.to_numpy())
        np.testing.assert_array_equal(first.qx.to_numpy(), second.qx.to_numpy())
        np.testing.assert_array_equal(first.foam.to_numpy(), second.foam.to_numpy())
        np.testing.assert_array_equal(first.wetness.to_numpy(), second.wetness.to_numpy())

    def test_inject_raises_local_eta_and_is_deterministic(self):
        before, after = [], []
        for _ in range(2):
            sea = ShallowSea()
            sea.set_wave(0.30, 10.0)
            sea.advance(int(2.0 / SIM_DT))
            before.append(float(sample_probe(sea, -20.0, 0.0, sea.time)[0]))
            sea.inject(-20.0, 0.0, 0.8, 0.3)
            sea.advance(3)
            after.append(sea.h.to_numpy().copy())
        self.assertGreater(after[0].max(), before[0])
        np.testing.assert_array_equal(after[0], after[1])
        self.assertGreater(after[0].max(), max(before))

    def test_cubic_sample_matches_nodes_and_exact_gradient(self):
        sea = ShallowSea()
        sea.set_wave(0.30, 10.0)
        sea.advance(int(2.0 / SIM_DT))
        i, j = 100, 120
        x = -SEA_HALF + i * SEA_DX
        z = -SEA_HALF + j * SEA_DX
        probe = np.array(sample_probe(sea, x, z, sea.time))
        h = sea.h.to_numpy()

        axis = node_axis()
        bed = numpy_terrain(axis[:, None], axis[None, :])
        eta = h + bed
        indices = [(i,j,0.5),(i-1,j,0.125),(i+1,j,0.125),(i,j-1,0.125),(i,j+1,0.125)]
        valid = [(a,b,w) for a,b,w in indices if h[a,b] > 0.0001]
        filtered = sum(w * eta[a,b] for a,b,w in valid) / sum(w for a,b,w in valid)
        self.assertAlmostEqual(probe[3], filtered - bed[i,j], places=5)
        self.assertAlmostEqual(probe[0], filtered, places=5)
        # Cell-exact gradient: finite differences of eta inside one cell.
        px, pz = x + 0.21, z + 0.17
        eps = 0.01
        center = np.array(sample_probe(sea, px, pz, sea.time))
        fx = (float(sample_probe(sea, px + eps, pz, sea.time)[0])
              - float(sample_probe(sea, px - eps, pz, sea.time)[0])) / (2 * eps)
        fz = (float(sample_probe(sea, px, pz + eps, sea.time)[0])
              - float(sample_probe(sea, px, pz - eps, sea.time)[0])) / (2 * eps)
        self.assertAlmostEqual(center[1], fx, places=3)
        self.assertAlmostEqual(center[2], fz, places=3)

    def test_exterior_extension_matches_incident_wave(self):
        sea = ShallowSea()
        sea.set_wave(0.4, 8.0)
        t = 12.0
        for x, z in ((-70.0, 5.0), (-80.0, -30.0), (-80.0, 80.0), (-65.0, 33.0)):
            expected = numpy_incident(x, z, t, 0.4, 8.0)
            self.assertAlmostEqual(float(sample_probe(sea, x, z, t)[0]), expected, places=4)
        # The boundary plane (x = -SEA_HALF) must agree with the ghost forcing
        # inside the solver, so the far field joins the domain seamlessly.
        for z in (-40.0, 0.0, 40.0):
            self.assertAlmostEqual(float(incident_probe(-SEA_HALF, z, t, 0.4, 8.0)),
                                   numpy_incident(-SEA_HALF, z, t, 0.4, 8.0), places=5)
        # Finite and bounded by the ramped amplitude everywhere outside.
        for x, z in ((-100.0, 40.0), (-100.0, -40.0), (-64.5, 64.5)):
            value = float(sample_probe(sea, x, z, 6.0)[0])
            self.assertTrue(math.isfinite(value))
            self.assertLessEqual(abs(value), 0.4 * min(1.0, 6.0 / 8.0) + 1e-5)

    def test_incident_sea_varies_alongshore(self):
        # A single-frequency train parallel to the beach is what made the
        # old shoreline look painted on; the incident sea must drift in
        # amplitude and phase along z.
        t = 12.0
        zs = np.linspace(-60.0, 60.0, 25)
        row = np.array([numpy_incident(-SEA_HALF, float(z), t, 0.4, 8.0) for z in zs])
        self.assertGreater(row.std(), 0.03)
        self.assertGreater(row.max() - row.min(), 0.2)


class CameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_zoom_and_pan_limits(self):
        camera = ShoreCamera()
        camera.distance = 10.0
        for _ in range(40):
            camera.zoom(-1.0)
        self.assertAlmostEqual(camera.distance, 4.5)
        for _ in range(80):
            camera.zoom(1.0)
        self.assertAlmostEqual(camera.distance, 60.0)
        camera.target = np.array([0.0, 0.5, 0.0])
        for _ in range(200):
            camera.pan(0.05, 0.0)
        self.assertLessEqual(camera.target[0], 24.0)
        self.assertGreaterEqual(camera.target[1], -0.5)
        self.assertLessEqual(camera.target[1], 2.0)

    def test_presets_are_defined_and_applied(self):
        self.assertEqual(set(SHORE_PRESETS), {"overview", "waterline", "top"})
        self.assertEqual(set(SHORE_PRESETS_SEA), {"calm", "surf", "storm"})
        camera = ShoreCamera()
        apply_preset(camera, "waterline")
        np.testing.assert_allclose(camera.target, [2.0, 0.0, 0.0])
        self.assertAlmostEqual(camera.pitch, 0.12)
        self.assertAlmostEqual(camera.distance, 9.0)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = ShoreRenderer(64, 40, samples=1)
        cls.renderer.sea.set_wave(0.30, 10.0)
        cls.renderer.sea.advance(120)

    def test_presets_produce_finite_nonblank_frames(self):
        camera = ShoreCamera()
        clock = self.renderer.sea.time
        for lighting in (0.0, 1.0):
            for preset in ("overview", "waterline", "top"):
                apply_preset(camera, preset)
                self.renderer.draw(camera, clock, lighting=lighting)
                frame = self.renderer.image.to_numpy()
                self.assertTrue(np.isfinite(frame).all(), (preset, lighting))
                self.assertGreater(frame.std(), 0.04, (preset, lighting))
                self.assertGreaterEqual(frame.min(), 0.0)
                self.assertLessEqual(frame.max(), 1.0)

    def test_paused_state_is_deterministic(self):
        camera = ShoreCamera()
        apply_preset(camera, "overview")
        clock = self.renderer.sea.time
        self.renderer.draw(camera, clock, lighting=0.0)
        first = self.renderer.image.to_numpy()
        self.renderer.draw(camera, clock, lighting=0.0)
        np.testing.assert_array_equal(first, self.renderer.image.to_numpy())
        self.renderer.sea.advance(3)
        self.renderer.draw(camera, clock + 3 * SIM_DT, lighting=0.0)
        self.assertGreater(np.abs(self.renderer.image.to_numpy() - first).max(), 0.005)

    def test_splash_changes_the_frame(self):
        camera = ShoreCamera()
        apply_preset(camera, "overview")
        clock = self.renderer.sea.time
        self.renderer.draw(camera, clock, lighting=0.0)
        before = self.renderer.image.to_numpy()
        self.renderer.sea.inject(0.0, 0.0, 1.2, 0.6)
        self.renderer.sea.advance(2)
        self.renderer.draw(camera, self.renderer.sea.time, lighting=0.0)
        after = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(after - before).mean(), 0.002)
        self.assertGreater(np.abs(after - before).max(), 0.02)

    def test_foam_slider_changes_the_frame(self):
        camera = ShoreCamera()
        self.renderer.sea.advance(int(18.0 / SIM_DT))
        self.assertGreater(self.renderer.sea.foam.to_numpy().max(), 0.05)
        # Aim at the strongest foam patch that lies on open water, so the
        # slider change is visible from the surface foam term.
        foam = self.renderer.sea.foam.to_numpy()
        depth = self.renderer.sea.h.to_numpy()
        on_water = np.argwhere((foam > 0.2) & (depth > 0.05))
        self.assertGreater(len(on_water), 0)
        i, j = on_water[foam[on_water[:, 0], on_water[:, 1]].argmax()]
        axis = node_axis()
        camera.target = np.array([axis[i], 0.0, axis[j]])
        camera.yaw, camera.pitch, camera.distance = 0.5, 0.7, 8.0
        clock = self.renderer.sea.time
        self.renderer.water_controls[None] = [0.06, 0.5, 0.0, 1.0]
        self.renderer.draw(camera, clock, lighting=0.0)
        bare = self.renderer.image.to_numpy()
        self.renderer.water_controls[None] = [0.06, 0.5, 2.0, 1.0]
        self.renderer.draw(camera, clock, lighting=0.0)
        foamy = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(foamy - bare).mean(), 0.003)

    def test_wave_surface_normal_matches_finite_differences(self):
        x, z = -30.13, -20.11
        t, eps = self.renderer.sea.time, 0.01
        probe = np.array(surface_probe(self.renderer, x, z, t))
        normal = probe[1:]
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=3)
        hx = (float(height_probe(self.renderer, x + eps, z, t))
              - float(height_probe(self.renderer, x - eps, z, t))) / (2 * eps)
        hz = (float(height_probe(self.renderer, x, z + eps, t))
              - float(height_probe(self.renderer, x, z - eps, t))) / (2 * eps)
        reference = np.array([-hx, 1.0, -hz])
        reference /= np.linalg.norm(reference)
        cosine = float(np.dot(normal, reference))
        self.assertGreater(cosine, 0.9999, (normal, reference))

    def test_bounds_cover_measured_heights(self):
        renderer = self.renderer
        amp = float(renderer.sea.amplitude[None])
        bounds = np.array(bounds_probe(renderer))
        eta_min = float(renderer.sea.eta_min[None])
        eta_max = float(renderer.sea.eta_max[None])
        self.assertLessEqual(bounds[0], eta_min + 1e-3)
        self.assertGreaterEqual(bounds[1], eta_max - 1e-3)
        self.assertLessEqual(bounds[0], -1.01 * amp)
        self.assertGreaterEqual(bounds[1], 1.01 * amp)

    def test_refined_hit_matches_surface_sample(self):
        renderer = self.renderer
        clock = renderer.sea.time
        origin = np.array([-30.0, 4.0, -20.0], np.float32)
        direction = np.array([0.1, -1.0, 0.15], np.float32)
        direction /= np.linalg.norm(direction)
        hit = np.array(hit_probe(renderer, origin, direction, clock))
        self.assertLess(hit[0], 100)
        point = origin + direction * hit[0]
        reference = np.array(surface_probe(renderer, point[0], point[2], clock))
        self.assertAlmostEqual(float(point[1]), reference[0], delta=0.004)
        np.testing.assert_allclose(hit[1:], reference[1:], atol=2e-3)

    def test_reset_restores_still_water(self):
        renderer = self.renderer
        renderer.sea.advance(10)
        renderer.sea.reset()
        self.assertEqual(renderer.sea.time, 0.0)
        axis = node_axis()
        bed = numpy_terrain(axis[:, None], axis[None, :])
        np.testing.assert_allclose(renderer.sea.h.to_numpy(), np.maximum(-bed, 0.0), atol=1e-6)
        renderer.sea.set_wave(0.30, 10.0)
        renderer.sea.advance(60)

    def test_water_pick_hits_the_plane_inside_the_domain(self):
        camera = ShoreCamera()
        apply_preset(camera, "overview")
        point = water_pick(camera, 0.5, 0.55, 64, 40)
        self.assertIsNotNone(point)
        self.assertLess(abs(point[0]), SEA_HALF)
        self.assertLess(abs(point[1]), SEA_HALF)
        # Rays that leave the top of the frame never reach the plane.
        apply_preset(camera, "overview")
        self.assertIsNone(water_pick(camera, 0.5, 0.0, 64, 40))

    def test_argument_validation(self):
        for argv in (["--wave-amplitude", "0.7"], ["--wave-amplitude", "0.01"],
                     ["--wave-period", "3"], ["--wave-period", "20"],
                     ["--time", "nan"], ["--window-frames", "-1"],
                     ["--water-roughness", "0.5"], ["--samples", "3"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_sea_parameters(self):
        args = parse_args(["--sea", "storm"])
        self.assertEqual(sea_parameters(args), SHORE_PRESETS_SEA["storm"])
        args = parse_args(["--sea", "auto", "--wave-amplitude", "0.2", "--wave-period", "7.0"])
        self.assertEqual(sea_parameters(args), (0.2, 7.0))
        args = parse_args([])
        self.assertEqual(sea_parameters(args), (0.3, 10.0))


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

        with patch("water.demo_06.ti.init"), \
             patch("water.demo_06.ti.sync", side_effect=lambda: events.append("sync")), \
             patch("water.demo_06.ShoreRenderer", return_value=renderer), \
             patch("water.demo_06.ti.ui.Window", side_effect=create_window), \
             patch("builtins.print"):
            main(["--window-frames", "1", "--time", "0"])
        self.assertEqual(events, ["render", "sync", "create_window", "image", "show"])


@ti.kernel
def offset_probe(renderer: ti.template(), p: ti.types.vector(3, ti.f32)) -> ti.f32:
    return renderer.water_ray_offset(p, ti.Vector([0., 1., 0.]))


@ti.kernel
def properties_probe(renderer: ti.template(), p: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return renderer.surface_properties(p, 6, renderer.materials[6])


@ti.kernel
def floor_probe(renderer: ti.template(), p: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    color, normal = renderer.floor_material(p, 0., 0.)
    return color


@ti.kernel
def shadow_probe(renderer: ti.template(), p: ti.types.vector(3, ti.f32), d: ti.types.vector(3, ti.f32)) -> ti.f32:
    return renderer.occlusion_distance(p, d, 8.)


@ti.kernel
def detail_probe(renderer: ti.template(), x: ti.f32, z: ti.f32, footprint: ti.f32) -> ti.types.vector(2, ti.f32):
    return renderer.detail_gradient(x, z, 2., footprint)


@ti.kernel
def terrain_ray_probe(renderer: ti.template(), origin: ti.types.vector(3, ti.f32), direction: ti.types.vector(3, ti.f32)) -> ti.f32:
    return renderer.terrain_hit(origin, direction, 200.)


class ShoreConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = ShoreRenderer(64, 40, samples=1)

    def setUp(self):
        self.renderer.sea.reset()
        self.renderer.water_controls[None] = [0.06, 0.5, 1., 1.]

    def test_static_lake_has_flat_display_and_no_breaking_foam(self):
        sea = self.renderer.sea
        sea.set_wave(0., 10.)
        sea.advance(300)
        wet = sea.h.to_numpy() > 0.1
        self.assertLess(abs(sea.eta_view.to_numpy()[wet]).max(), 0.002)
        self.assertLess(sea.foam.to_numpy().max(), 0.001)
        self.assertLess(sea.strand.to_numpy().max(), 0.001)

    def test_thin_film_height_and_normal_match_the_same_surface(self):
        sea = self.renderer.sea
        axis = node_axis()
        eta = -0.043 + 0.015 * axis[:,None] - 0.004 * axis[None,:]
        sea.eta_view.from_numpy(eta.astype(np.float32))
        for x in (4.1, 4.2, 4.4):
            p = np.array(sample_probe(sea, x, 0., 0.))
            self.assertAlmostEqual(p[0], -0.043 + 0.015 * x, delta=2e-6)
            np.testing.assert_allclose(p[1:3], [.015,-.004], atol=2e-5)
            self.assertGreater(p[3], 0.)
            self.assertLess(p[3], .04)
            film = p[0] - float(terrain_probe(x, 0.)[0])
            offset = float(offset_probe(self.renderer, [x,p[0],0.]))
            self.assertLessEqual(offset, film * .151)

    def test_subcell_shoreline_retains_a_thin_water_film(self):
        sea = self.renderer.sea
        sea.eta_view.fill(.03)
        low, high = 4., 6.
        for _ in range(30):
            mid = .5 * (low + high)
            if float(terrain_probe(mid,0.)[0]) < .03: low = mid
            else: high = mid
        wet = np.array(sample_probe(sea, low - .002, 0., 0.))
        dry = np.array(sample_probe(sea, high + .002, 0., 0.))
        self.assertAlmostEqual(wet[0], .03, delta=2e-6)
        self.assertGreater(wet[3], .00001)
        self.assertLess(wet[3], .001)
        self.assertLess(dry[0], -1e5)

    def test_domain_seam_matches_height_and_derivatives(self):
        sea = self.renderer.sea
        sea.eta_view.fill(.17)
        sea.set_wave(.3, 10.)
        a = np.array(sample_probe(sea,-64.001,1.3,20.))
        b = np.array(sample_probe(sea,-63.999,1.3,20.))
        # Account for the real slope over the 2 mm gap before testing a seam.
        self.assertAlmostEqual(a[0] + .002 * a[1], b[0], delta=5e-6)
        np.testing.assert_allclose(a[1:3],b[1:3],atol=5e-5)
        for x in (-63.1,-60.7):
            p = np.array(sample_probe(sea,x,1.3,20.))
            eps = .002
            finite = (float(sample_probe(sea,x+eps,1.3,20.)[0])-float(sample_probe(sea,x-eps,1.3,20.)[0]))/(2*eps)
            self.assertAlmostEqual(p[1],finite,delta=2e-4)
        self.assertLess(float(sample_probe(sea,80.,0.,20.)[0]),-1e5)

    def test_advection_moves_mobile_foam_and_leaves_strand_on_sand(self):
        sea = self.renderer.sea
        sea.h.fill(1.);sea.qx.fill(1.);sea.qz.fill(0.)
        axis=node_axis()
        foam=np.exp(-(axis[:,None]**2+axis[None,:]**2)/4).astype(np.float32)
        sea.foam.from_numpy(foam);sea.strand.fill(.3)
        sea._begin_step();sea._advect_foam(.2)
        moved=sea.foam_adv.to_numpy()
        center=float((moved*axis[:,None]).sum()/moved.sum())
        self.assertAlmostEqual(center,.2,delta=.01)
        np.testing.assert_allclose(sea.strand.to_numpy(),.3,atol=1e-6)

    def test_cfl_substeps_cover_a_full_requested_tick(self):
        sea=self.renderer.sea
        sea.h.fill(1.);sea.qx.fill(5.);sea.qz.fill(5.)
        sea.advance(1)
        self.assertGreater(sea.internal_steps,1)
        self.assertLessEqual(sea.last_cfl,.420001)
        self.assertAlmostEqual(sea.time,SIM_DT,places=10)
        self.assertGreaterEqual(sea.h.to_numpy().min(),0.)

    def test_wet_sand_reflects_and_foam_control_reaches_deposits(self):
        r=self.renderer;p=[10.,.27,0.]
        r.sea.wet_view.fill(0.)
        dry=np.array(properties_probe(r,p))
        r.sea.wet_view.fill(1.)
        wet=np.array(properties_probe(r,p))
        self.assertLess(wet[0],dry[0]-.3)
        r.sea.strand_view.fill(.8)
        r.water_controls[None]=[.06,.5,0.,1.]
        bare=np.array(floor_probe(r,p))
        r.water_controls[None]=[.06,.5,1.,1.]
        foam=np.array(floor_probe(r,p))
        self.assertGreater(abs(foam-bare).mean(),.1)

    def test_submerged_bed_uses_water_dielectric_contrast(self):
        r = self.renderer
        r.sea.wet_view.fill(1.)
        r.sea.h_view.fill(0.)
        exposed = np.array(properties_probe(r, [10., .27, 0.]))
        r.sea.h_view.fill(.1)
        submerged = np.array(properties_probe(r, [10., .27, 0.]))
        self.assertGreater(submerged[0], exposed[0] + .3)
        self.assertAlmostEqual(submerged[2], .003, places=6)

    def test_reef_casts_a_directional_shadow(self):
        r=self.renderer
        blocked=float(shadow_probe(r,[-18.,1.,-13.],[0.,0.,1.]))
        clear=float(shadow_probe(r,[-18.,1.,-13.],[0.,0.,-1.]))
        self.assertLess(blocked,5.)
        self.assertAlmostEqual(clear,8.,places=5)

    def test_terrain_march_does_not_stall_above_reef_flanks(self):
        for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            origin = np.array([-18 + 5 * math.cos(angle), 5., -10 + 5 * math.sin(angle)], np.float32)
            target = np.array([-18., 1., -10.], np.float32)
            direction = target - origin
            direction /= np.linalg.norm(direction)
            t = float(terrain_ray_probe(self.renderer, origin, direction))
            self.assertLess(t, 20.)
            point = origin + direction * t
            bed = float(terrain_probe(point[0], point[2])[0])
            self.assertAlmostEqual(point[1], bed, delta=3e-5)

    def test_grazing_beach_rays_reach_terrain(self):
        camera = ShoreCamera()
        apply_preset(camera, "overview")
        eye, forward, right, up = camera.basis()
        scale = math.tan(math.radians(camera.fov / 2))
        for x, y in [(630, 130), (620, 135), (610, 140), (560, 100), (500, 100)]:
            direction = forward + right * ((2 * (x + .5) / 640 - 1) * 1.6 * scale) + up * ((1 - 2 * (y + .5) / 400) * scale)
            direction /= np.linalg.norm(direction)
            t = float(terrain_ray_probe(self.renderer, eye, direction))
            self.assertLess(t, 100.)
            point = eye + direction * t
            self.assertAlmostEqual(point[1], float(terrain_probe(point[0], point[2])[0]), delta=4e-5)

    def test_detail_and_sand_filter_out_below_pixel_resolution(self):
        r=self.renderer
        r.sea.h_view.fill(0.)
        np.testing.assert_allclose(detail_probe(r,0.,0.,0.),0.,atol=1e-7)
        r.sea.h_view.fill(1.)
        self.assertGreater(np.linalg.norm(detail_probe(r,0.,0.,0.)),1e-5)
        np.testing.assert_allclose(detail_probe(r,0.,0.,10.),0.,atol=1e-7)


if __name__ == "__main__":
    unittest.main()
