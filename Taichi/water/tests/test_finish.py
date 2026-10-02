"""GGX limits, refractive light transport, detail filtering, and HDR finishing."""
import math
import unittest

import numpy as np
import taichi as ti

from water.demo_01 import PoolRenderer, parse_args
from water.finish import PostProcess, water_reflection


@ti.kernel
def reflection_sample(roughness: ti.f32, index: ti.i32, cosine: ti.f32) -> ti.types.vector(4, ti.f32):
    view = ti.Vector([ti.sqrt(1.0 - cosine * cosine), cosine, 0.0])
    ray, weight = water_reflection(view, ti.Vector([0.0, 1.0, 0.0]), roughness, index, 4)
    return ti.Vector([ray.x, ray.y, ray.z, weight])


@ti.kernel
def detail_sample(renderer: ti.template(), footprint: ti.f32, amplitude: ti.f32) -> ti.types.vector(2, ti.f32):
    return renderer.detail_gradient(0.3, 0.4, 2.0, amplitude, footprint)


class FinishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = PoolRenderer(32, 32)
        cls.post = PostProcess(64, 64)

    def test_delta_reflection_and_grazing_are_bounded(self):
        for cosine in (1.0, 0.3, 0.001):
            for roughness in (0.0, 0.16, 0.45):
                for index in range(4):
                    sample = np.array(reflection_sample(roughness, index, cosine))
                    self.assertTrue(np.isfinite(sample).all())
                    self.assertAlmostEqual(np.linalg.norm(sample[:3]), 1.0, places=5)
                    self.assertGreaterEqual(sample[3], 0.0)
                    self.assertLessEqual(sample[3], 1.0)
                    if roughness == 0.0:
                        np.testing.assert_allclose(sample[:3], [-math.sqrt(1 - cosine * cosine), cosine, 0], atol=1e-5)
        rays = [np.array(reflection_sample(0.35, i, 0.7))[:3] for i in range(4)]
        self.assertGreater(np.std(rays, axis=0).max(), 0.05)

    def test_detail_filter_and_zero_amplitude(self):
        r = self.renderer
        self.assertGreater(np.linalg.norm(detail_sample(r, 0, 0.035)), 0.001)
        np.testing.assert_array_equal(detail_sample(r, 0.5, 0.035), [0, 0])
        np.testing.assert_array_equal(detail_sample(r, 0, 0), [0, 0])

    def test_flat_caustics_energy_and_moving_focus(self):
        r = self.renderer
        r.trace_caustics(2, 0, 7, 1, 0)
        flat = r.caustic_raw.to_numpy()[16:-16, 16:-16]
        sun_cosine = 0.95 / np.linalg.norm([-0.50, 0.95, -0.57])
        f0 = ((1.333 - 1) / (1.333 + 1)) ** 2
        expected = 1 - (f0 + (1 - f0) * (1 - sun_cosine) ** 5)
        self.assertAlmostEqual(flat.mean(), expected, places=3)
        self.assertLess(flat.std(), 0.001)
        r.trace_caustics(2, 0.035, 7, 1, 0)
        r.filter_caustics()
        first = r.caustic_map.to_numpy()
        self.assertGreater(first.std(), 0.15)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(first.min(), 0)
        self.assertLessEqual(first.max(), 8)
        self.assertLess(r.caustic_raw.to_numpy().sum(), 128 * 96 * 1.05)
        r.trace_caustics(3, 0.035, 7, 1, 0)
        r.filter_caustics()
        self.assertGreater(np.mean(np.abs(first - r.caustic_map.to_numpy())), 0.05)

    def test_caustics_obey_source_shadow(self):
        # A roof covering the pool must remove the refracted sunlight.
        r = self.renderer
        count, groups = r.boxes, r.groups
        center, extent, bevel = r.box_center[0], r.box_extent[0], r.box_bevel[0]
        try:
            r.boxes, r.groups = 1, 0
            r.box_center[0], r.box_extent[0], r.box_bevel[0] = [0, 2, 0], [8, .1, 8], 0
            r.trace_caustics(2, 0.035, 7, 1, 1)
            self.assertLess(r.caustic_raw.to_numpy().max(), 1e-6)
        finally:
            r.boxes, r.groups = count, groups
            r.box_center[0], r.box_extent[0], r.box_bevel[0] = center, extent, bevel

    def test_bloom_black_constant_and_hdr_impulse(self):
        p = self.post
        p.hdr.fill(0)
        p.apply(exposure=1, strength=.2, vignette=0)
        np.testing.assert_array_equal(p.image.to_numpy(), 0)
        p.hdr.fill(.1)
        p.apply(exposure=1, strength=.2, vignette=0)
        np.testing.assert_array_equal(p.bloom.to_numpy(), 0)
        source = np.zeros((64, 64, 3), dtype=np.float32)
        source[30:34, 30:34] = [12, 6, 2]
        p.hdr.from_numpy(source)
        p.apply(exposure=1, strength=0, vignette=0)
        disabled = p.image.to_numpy()
        p.apply(exposure=1, strength=.2, vignette=0)
        enabled = p.image.to_numpy()
        self.assertGreater(enabled[28, 32, 0], disabled[28, 32, 0])
        self.assertGreater(p.bloom.to_numpy().sum(), 0)
        np.testing.assert_array_equal(source, p.hdr.to_numpy())
        self.assertTrue(np.isfinite(enabled).all())
        self.assertLessEqual(enabled.max(), 1)
        self.assertGreater(enabled[28, 32, 0], enabled[28, 32, 2])

    def test_bloom_blur_preserves_constant(self):
        p = self.post
        p.bright.fill([2, 1, .5])
        p.blur(p.bright, p.horizontal, 0)
        p.blur(p.horizontal, p.bloom, 1)
        expected = np.broadcast_to([2, 1, .5], p.bloom.to_numpy().shape)
        np.testing.assert_allclose(p.bloom.to_numpy(), expected, rtol=1e-6)

    def test_roughness_input_validation(self):
        from contextlib import redirect_stderr
        from io import StringIO
        for value in ("nan", "inf", "-0.1", "0.6"):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(["--water-roughness", value])


if __name__ == "__main__":
    unittest.main()
