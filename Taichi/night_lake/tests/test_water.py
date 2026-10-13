"""Water field checks: solver CPU mirror, rain-rate linkage, analytic
gradients, and the moon glitter band in the rendered frame (no GUI)."""

import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import NightLakeScene
from night_lake.sky import MOON_DIRECTION
from night_lake.lake import (ABSORB_CELLS, DAMPING, LakeWaves, RainRipples,
                              LAKE_MAX_X, LAKE_MAX_Z, LAKE_MIN_X, LAKE_MIN_Z,
                              RIPPLE_NX, RIPPLE_NZ, SUBSTEP_DT, WAVE_SPEED,
                              CELL_X)


@ti.kernel
def ripple_sample(ripples: ti.template(), x: ti.f32, z: ti.f32) -> ti.types.vector(3, ti.f32):
    return ripples.sample(x, z)


@ti.kernel
def wave_gradient(waves: ti.template(), x: ti.f32, z: ti.f32, t: ti.f32) -> ti.types.vector(2, ti.f32):
    return waves.gradient(x, z, t)


@ti.kernel
def wave_height(waves: ti.template(), x: ti.f32, z: ti.f32, t: ti.f32) -> ti.f32:
    return waves.height(x, z, t)


def reference_step(h, h_prev, courant2, damping):
    """One leapfrog step mirroring RainRipples.step on the CPU."""
    left = np.empty_like(h)
    left[0, :], left[1:, :] = h[0, :], h[:-1, :]
    right = np.empty_like(h)
    right[-1, :], right[:-1, :] = h[-1, :], h[1:, :]
    down = np.empty_like(h)
    down[:, 0], down[:, 1:] = h[:, 0], h[:, :-1]
    up = np.empty_like(h)
    up[:, -1], up[:, :-1] = h[:, -1], h[:, 1:]
    laplacian = left + right + down + up - 4.0 * h
    i = np.arange(RIPPLE_NX)[:, None]
    j = np.arange(RIPPLE_NZ)[None, :]
    edge = np.minimum(np.minimum(i, RIPPLE_NX - 1 - i), np.minimum(j, RIPPLE_NZ - 1 - j))
    absorb = 0.90 + 0.10 * np.minimum(1.0, edge / ABSORB_CELLS)
    return h + damping * absorb * (h - h_prev) + courant2 * laplacian


class RippleSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_solver_matches_cpu_reference(self):
        ripples = RainRipples()
        rng = np.random.default_rng(5)
        # A smooth field whose previous step is nearly identical: a random
        # h / h_prev pair injects a large mean velocity that the zero mode
        # of the wave equation integrates into a slow drift, which would
        # trip the spike clamp in the kernel and break the mirror.
        h0 = (rng.standard_normal((RIPPLE_NX, RIPPLE_NZ)) * 0.002).astype(np.float32)
        h_prev0 = (h0 * 0.99 + rng.standard_normal((RIPPLE_NX, RIPPLE_NZ)) * 0.0002).astype(np.float32)
        ripples.h.from_numpy(h0)
        ripples.h_prev.from_numpy(h_prev0)
        ripples.step_n(6)
        courant2 = (WAVE_SPEED * SUBSTEP_DT / CELL_X) ** 2
        h, h_prev = h0.astype(np.float64), h_prev0.astype(np.float64)
        for _ in range(6):
            h, h_prev = reference_step(h, h_prev, courant2, DAMPING), h
        np.testing.assert_allclose(ripples.heights(), h, atol=2e-5)

    def test_rain_rate_scales_injected_energy(self):
        # Event-driven rain: the high-rate run receives three times as
        # many landing events per frame, so the deposited energy must
        # scale accordingly.
        low, high = RainRipples(), RainRipples()
        for _ in range(30):
            low_n = int(low.rng.poisson(0.5))
            high_n = int(high.rng.poisson(1.5))
            if low_n:
                positions = low.rng.uniform(
                    [LAKE_MIN_X, LAKE_MIN_Z], [LAKE_MAX_X, LAKE_MAX_Z],
                    size=(low_n, 2))
                low.inject_events(positions.astype(np.float32),
                                  low.event_strengths(low_n, 0.5))
            if high_n:
                positions = high.rng.uniform(
                    [LAKE_MIN_X, LAKE_MIN_Z], [LAKE_MAX_X, LAKE_MAX_Z],
                    size=(high_n, 2))
                high.inject_events(positions.astype(np.float32),
                                   high.event_strengths(high_n, 0.5))
        low_count, low_energy = low.stats()
        high_count, high_energy = high.stats()
        self.assertGreater(low_count, 0)
        self.assertGreater(low_energy, 0.0)
        self.assertGreater(high_energy, 1.8 * low_energy)

    def test_ripple_sample_gradient_matches_finite_differences(self):
        ripples = RainRipples()
        rng = np.random.default_rng(9)
        # Cell-interior points inside the real grid keep the finite
        # differences inside one cell, where the bilinear gradient is
        # exact.
        for _ in range(6):
            i = rng.integers(2, RIPPLE_NX - 3)
            j = rng.integers(2, RIPPLE_NZ - 3)
            x = LAKE_MIN_X + (i + 0.5 + 0.2 * (rng.random() - 0.5)) * CELL_X
            z = LAKE_MIN_Z + (j + 0.5 + 0.2 * (rng.random() - 0.5)) * CELL_X
            eps = 1e-4
            sample = np.array(ripple_sample(ripples, x, z))
            dx = (ripple_sample(ripples, x + eps, z)[0] - ripple_sample(ripples, x - eps, z)[0]) / (2 * eps)
            dz = (ripple_sample(ripples, x, z + eps)[0] - ripple_sample(ripples, x, z - eps)[0]) / (2 * eps)
            np.testing.assert_allclose(sample[1], dx, atol=1e-4)
            np.testing.assert_allclose(sample[2], dz, atol=1e-4)


class WaveGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.waves = LakeWaves()

    def test_analytic_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        eps = 1e-3
        for _ in range(6):
            x, z, t = rng.uniform(-20, 20), rng.uniform(0, 60), rng.uniform(0, 10)
            analytic = np.array(wave_gradient(self.waves, x, z, t))
            dx = (wave_height(self.waves, x + eps, z, t) - wave_height(self.waves, x - eps, z, t)) / (2 * eps)
            dz = (wave_height(self.waves, x, z + eps, t) - wave_height(self.waves, x, z - eps, t)) / (2 * eps)
            np.testing.assert_allclose(analytic, [dx, dz], atol=1e-4)


class GlitterBandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)
        cls.scene.water_controls[None] = [1.0, 1.0, 0.0, 0.06]

    def test_moon_glitter_band_is_brightest_on_the_water(self):
        camera = FixedCamera()
        scene = self.scene
        scene.draw(camera, 2.0)
        frame = scene.image.to_numpy()
        self.assertTrue(np.isfinite(frame).all())
        eye, forward, right, up = camera.basis()
        scale = float(np.tan(np.radians(camera.fov / 2)))
        width, height = scene.width, scene.height
        px = (2.0 * (np.arange(width) + 0.5) / width - 1.0) * width / height * scale
        py = (2.0 * (np.arange(height) + 0.5) / height - 1.0) * scale
        grid_x, grid_y = np.meshgrid(px, py, indexing="ij")
        directions = (forward[None, None, :] + right[None, None, :] * grid_x[..., None]
                      + up[None, None, :] * grid_y[..., None])
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        # Column whose ray azimuth best matches the moon azimuth.
        azimuth = np.arctan2(directions[..., 0], directions[..., 2])
        target = float(np.arctan2(MOON_DIRECTION[0], MOON_DIRECTION[2]))
        column_score = np.abs(azimuth - target)
        moon_x = int(np.argmin(column_score.min(axis=1)))
        rows = slice(0, int(height * 0.35))
        glitter = frame[moon_x - 10:moon_x + 11, rows, :].mean(axis=2)
        far_x = int(width * 0.12)
        other = frame[far_x - 10:far_x + 11, rows, :].mean(axis=2)
        self.assertGreater(np.percentile(glitter, 95), np.percentile(other, 95) + 0.02)


if __name__ == "__main__":
    unittest.main()
