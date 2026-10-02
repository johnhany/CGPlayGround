"""Physical limits and lighting lookup checks, independent of a GUI."""
import math
import unittest

import numpy as np
import taichi as ti

from water.pbr import (bake_brdf_lut, convolve_environment, direction_grid,
                       fresnel_schlick, ggx_brdf)
from water.demo_01 import PoolRenderer


@ti.kernel
def fresnel_value(cosine: ti.f32) -> ti.f32:
    return fresnel_schlick(cosine, 0.04)


@ti.kernel
def brdf_value(view: ti.types.vector(3, ti.f32), light: ti.types.vector(3, ti.f32),
               roughness: ti.f32) -> ti.types.vector(3, ti.f32):
    return ggx_brdf(ti.Vector([0.5, 0.3, 0.1]), roughness, 0.0,
                    ti.Vector([0.0, 1.0, 0.0]), view, light, 0.04)


@ti.kernel
def ao_value(renderer: ti.template(), p: ti.types.vector(3, ti.f32)) -> ti.f32:
    return renderer.contact_ao(p, ti.Vector([0.0, 1.0, 0.0]))


@ti.kernel
def shadow_value(renderer: ti.template(), p: ti.types.vector(3, ti.f32)) -> ti.f32:
    return renderer.visibility(p, ti.Vector([0.0, 1.0, 0.0]), ti.Vector([0.0, 1.0, 0.0]), 1)


@ti.kernel
def environment_value(renderer: ti.template(), direction: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
    return renderer.environment.sample_map(renderer.environment.environment, direction, 1.0)


class BakingTests(unittest.TestCase):
    def test_constant_environment_preserves_energy(self):
        normals = direction_grid(8, 4)
        sampler = lambda directions: np.ones_like(directions) * [0.2, 0.5, 1.4]
        expected = np.broadcast_to([0.2, 0.5, 1.4], normals.shape)
        np.testing.assert_allclose(convolve_environment(normals, sampler=sampler), expected * math.pi, rtol=1e-6)
        for roughness in (0.0, 0.5, 1.0):
            np.testing.assert_allclose(convolve_environment(normals, roughness, sampler=sampler), expected, rtol=1e-6)

    def test_lookup_is_finite_nonnegative_and_bounded(self):
        lut = bake_brdf_lut()
        self.assertTrue(np.isfinite(lut).all())
        self.assertGreaterEqual(lut.min(), 0)
        self.assertLess(lut.max(), 1.1)


class ShadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = PoolRenderer(32, 32)

    def test_fresnel_limits(self):
        self.assertAlmostEqual(fresnel_value(1.0), 0.04, places=6)
        self.assertAlmostEqual(fresnel_value(0.0), 1.0, places=6)

    def test_brdf_reciprocity_and_backside(self):
        v = np.array([0.4, 1.0, 0.0], dtype=np.float32); v /= np.linalg.norm(v)
        l = np.array([0.0, 1.0, 0.6], dtype=np.float32); l /= np.linalg.norm(l)
        a = np.array(brdf_value(v, l, 0.45)) / l[1]
        b = np.array(brdf_value(l, v, 0.45)) / v[1]
        np.testing.assert_allclose(a, b, rtol=1e-5)
        np.testing.assert_array_equal(brdf_value(v, [0, -1, 0], 0.45), [0, 0, 0])
        self.assertTrue(np.isfinite(brdf_value(v, l, 0.0)).all())

    def test_contact_ao_and_disable(self):
        renderer = self.renderer
        controls = np.array(renderer.light_controls[None])
        try:
            open_value = ao_value(renderer, [0.0, 3.0, 0.0])
            contact_value = ao_value(renderer, [3.55, -0.19, 0.0])
            self.assertAlmostEqual(open_value, 1.0, places=5)
            self.assertLess(contact_value, open_value)
            disabled = controls.copy(); disabled[2] = 0
            renderer.light_controls[None] = disabled
            self.assertAlmostEqual(ao_value(renderer, [3.55, -0.19, 0.0]), 1.0, places=5)
        finally:
            renderer.light_controls[None] = controls

    def test_environment_seam_and_poles(self):
        r = self.renderer
        a = environment_value(r, [0.000001, 0, -1])
        b = environment_value(r, [-0.000001, 0, -1])
        np.testing.assert_allclose(a, b, atol=1e-4)  # float32 angular lookup
        for direction in ([0, 1, 0], [0, -1, 0]):
            self.assertTrue(np.isfinite(environment_value(r, direction)).all())

    def test_soft_shadow_has_partial_visibility(self):
        # Isolate a block with its projected edge exactly above the receiver.
        ti.init(arch=ti.cpu, offline_cache=False)
        r = PoolRenderer(32, 32)
        r.boxes, r.groups = 1, 0
        r.box_center[0], r.box_extent[0], r.box_bevel[0] = [0, 1, 0], [.2, .2, .2], 0
        r.light_controls[None] = [.12, 4, 0, 1]
        self.assertEqual(shadow_value(r, [0, 0, 0]), 0)
        self.assertEqual(shadow_value(r, [1, 0, 0]), 1)
        edge = shadow_value(r, [.2, 0, 0])
        self.assertGreater(edge, 0)
        self.assertLess(edge, 1)


if __name__ == "__main__":
    unittest.main()
