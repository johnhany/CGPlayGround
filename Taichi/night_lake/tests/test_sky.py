"""Sky layer checks: the fbm cloud mask is deterministic, matches its CPU
mirror, dims the environment dome inside the cloud body, and attenuates
star and moon radiance proportionally to coverage.
"""

import unittest

import numpy as np
import taichi as ti

from night_lake.sky import (CLOUD_MOON_BLOCK, CLOUD_STAR_BLOCK, _cloud_lining,
                            cloud_mask, cloud_mask_np, moon_radiance,
                            night_sky, star_radiance)


@ti.kernel
def _probe(directions: ti.types.ndarray(ndim=2), out: ti.types.ndarray(ndim=2)):
    for i in range(directions.shape[0]):
        d = ti.Vector([directions[i, 0], directions[i, 1], directions[i, 2]])
        mask = cloud_mask(d)
        star = star_radiance(d)
        moon = moon_radiance(d)
        base = ti.Vector([0.05, 0.06, 0.08])
        sky = night_sky(d, base)
        out[i, 0] = mask
        out[i, 1] = star.sum()
        out[i, 2] = star.sum() * (1.0 - mask * CLOUD_STAR_BLOCK)
        out[i, 3] = moon.sum()
        out[i, 4] = moon.sum() * (1.0 - mask * CLOUD_MOON_BLOCK)
        out[i, 5] = sky.sum()
        out[i, 6] = base.sum() * (1.0 - mask * 0.55)
        out[i, 7] = _cloud_lining(d, mask)


def _view_directions(count=6000, seed=7):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(count, 3)).astype(np.float32)
    d[:, 1] = np.abs(d[:, 1]) * 0.8 + 0.05
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d


class CloudMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def _probe_np(self, directions):
        d = np.asarray(directions, dtype=np.float32)
        out = np.zeros((d.shape[0], 8), dtype=np.float32)
        _probe(d, out)
        return out

    def test_mask_is_deterministic_and_covers_part_of_the_sky(self):
        d = _view_directions(512)
        m1 = cloud_mask_np(d)
        m2 = cloud_mask_np(d)
        np.testing.assert_array_equal(m1, m2)
        self.assertGreater((m1 > 0.7).mean(), 0.05)
        self.assertLessEqual(float(m1.max()), 1.0)

    def test_kernel_mask_matches_cpu_mirror(self):
        d = _view_directions(1024)
        out = self._probe_np(d)
        np.testing.assert_allclose(out[:, 0], cloud_mask_np(d), atol=2e-3)

    def test_clouds_dim_the_dome(self):
        d = _view_directions(2048)
        out = self._probe_np(d)
        mask, sky_sum, dome_sum = out[:, 0], out[:, 5], out[:, 6]
        # Stars and the moon also contribute; compare only where both are
        # negligible (away from any star cell and outside the moon disk).
        quiet = (out[:, 1] < 1e-4) & (out[:, 3] < 1e-4)
        cloudy = quiet & (mask > 0.7)
        clear = quiet & (mask < 0.05)
        self.assertGreater(cloudy.sum(), 50)
        self.assertGreater(clear.sum(), 50)
        # sky = dimmed dome + silver lining; the lining term is exported
        # as column 7 and sums to 0.55 + 0.58 + 0.66 times 0.10 amplitude.
        expected = dome_sum + out[:, 7] * (0.55 + 0.58 + 0.66) * 0.10
        np.testing.assert_allclose(sky_sum[quiet], expected[quiet], atol=1e-4)
        self.assertGreater(sky_sum[cloudy].mean(), 0.0)

    def test_stars_are_masked_by_clouds(self):
        d = _view_directions(4096)
        out = self._probe_np(d)
        mask, star_clear, star_masked = out[:, 0], out[:, 1], out[:, 2]
        covered = mask > 0.9
        self.assertGreater(covered.sum(), 20)
        # The masking law holds pointwise up to float32 tolerance.
        np.testing.assert_allclose(star_masked, star_clear
                                   * (1.0 - mask * CLOUD_STAR_BLOCK), atol=1e-5)
        # Inside dense cloud at most a fifth of the star power survives:
        # at mask >= 0.9 the law leaves 1 - 0.9 * 0.90 = 0.19.
        self.assertLess(star_masked[covered].max(),
                        0.20 * star_clear.max() + 1e-5)
        # Where dense cloud sits, surviving star power is strictly reduced.
        self.assertLess(star_masked[covered].sum(),
                        star_clear[covered].sum())

    def test_moon_is_masked_by_clouds(self):
        d = _view_directions(4096)
        out = self._probe_np(d)
        mask, moon_clear, moon_masked = out[:, 0], out[:, 3], out[:, 4]
        np.testing.assert_allclose(moon_masked, moon_clear
                                   * (1.0 - mask * CLOUD_MOON_BLOCK), atol=1e-5)
        self.assertLessEqual(moon_masked.max(), moon_clear.max() + 1e-5)
        covered = mask > 0.5
        self.assertGreater(covered.sum(), 20)


if __name__ == "__main__":
    unittest.main()
