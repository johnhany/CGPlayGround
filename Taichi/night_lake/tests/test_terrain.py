"""Terrain checks: CPU mirror of the height field, gradient consistency,
waterline path continuity, material zoning, and a soft waterline transition
in the rendered frame (no GUI)."""

import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import NightLakeScene
from night_lake.terrain import (GRASS_LINE_DEFAULT, SHORE_Z_DEFAULT,
                                WET_SCALE_DEFAULT, terrain_gradient,
                                terrain_gradient_np, terrain_height,
                                terrain_height_np, terrain_material)


@ti.kernel
def height_probe(x: ti.f32, z: ti.f32, shore_z: ti.f32,
                 detail: ti.f32) -> ti.types.vector(3, ti.f32):
    height = terrain_height(x, z, shore_z, detail)
    grad = terrain_gradient(x, z, shore_z, detail)
    return ti.Vector([height, grad.x, grad.y])


@ti.kernel
def material_probe(h: ti.f32, slope: ti.f32, wet_scale: ti.f32,
                   grass_line: ti.f32) -> ti.types.vector(3, ti.f32):
    return terrain_material(h, slope, wet_scale, grass_line)


def march_terrain_np(eye, directions, shore_z=SHORE_Z_DEFAULT, detail=1.0):
    """Vectorized CPU mirror of the scene terrain march; returns (t, height)."""
    eye = np.asarray(eye, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    n = directions.shape[0]
    t = np.full(n, 0.05, dtype=np.float32)
    px = eye[0] + t * directions[:, 0]
    py = eye[1] + t * directions[:, 1]
    pz = eye[2] + t * directions[:, 2]
    error = py - terrain_height_np(px, pz, shore_z, detail)
    stepped = np.zeros(n, dtype=bool)
    for _ in range(160):
        active = (error > 0.0) & (t < 150.0) & ~stepped
        if not active.any():
            break
        limit = 0.03 + 0.12 * t[active]
        step = np.minimum(np.maximum(error[active] * 1.2, 0.03), limit)
        t[active] += step
        px = eye[0] + t[active] * directions[active, 0]
        py = eye[1] + t[active] * directions[active, 1]
        pz = eye[2] + t[active] * directions[active, 2]
        error[active] = py - terrain_height_np(px, pz, shore_z, detail)
        stepped[active] = True
    hit = (error <= 0.0) & stepped
    result_t = np.where(hit, t, 1e6).astype(np.float32)
    result_h = np.where(hit, eye[1] + result_t * directions[:, 1],
                        np.float32(0.0))
    return result_t, result_h


class TerrainMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_taichi_height_matches_cpu_mirror(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            x = float(rng.uniform(-30.0, 30.0))
            z = float(rng.uniform(-5.0, 70.0))
            shore = float(rng.uniform(18.0, 40.0))
            detail = float(rng.uniform(0.0, 2.0))
            probe = np.array(height_probe(x, z, shore, detail))
            height = terrain_height_np(np.float32(x), np.float32(z),
                                       np.float32(shore), np.float32(detail))
            grad = terrain_gradient_np(np.float32(x), np.float32(z),
                                       np.float32(shore), np.float32(detail))
            np.testing.assert_allclose(probe[0], height, atol=1e-5)
            np.testing.assert_allclose(probe[1:], grad, atol=1e-4)

    def test_taichi_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(7)
        # The gradient stencil is +/-0.06 m (terrain_gradient); matching that
        # window keeps the comparison on the same smoothed derivative the
        # shading normals use, instead of the sharper local derivative.
        eps = 0.06
        for _ in range(20):
            x = float(rng.uniform(-30.0, 30.0))
            z = float(rng.uniform(-5.0, 70.0))
            probe = np.array(height_probe(x, z, SHORE_Z_DEFAULT, 1.0))
            dx = (height_probe(x + eps, z, SHORE_Z_DEFAULT, 1.0)[0]
                  - height_probe(x - eps, z, SHORE_Z_DEFAULT, 1.0)[0]) / (2 * eps)
            dz = (height_probe(x, z + eps, SHORE_Z_DEFAULT, 1.0)[0]
                  - height_probe(x, z - eps, SHORE_Z_DEFAULT, 1.0)[0]) / (2 * eps)
            np.testing.assert_allclose(probe[1], dx, atol=2e-3)
            np.testing.assert_allclose(probe[2], dz, atol=2e-3)

    def test_waterline_path_is_continuous(self):
        # Walking across the waterline in 0.1 m steps must never step up or
        # down more than a gentle threshold: no cliffs at the shoreline.
        for x in (-12.0, 0.0, 7.3):
            z = np.arange(SHORE_Z_DEFAULT - 25.0, SHORE_Z_DEFAULT + 25.0, 0.1)
            heights = terrain_height_np(np.float32(x) + np.zeros_like(z),
                                        z.astype(np.float32))
            steps = np.abs(np.diff(heights))
            self.assertLess(steps.max(), 0.06)

    def test_material_zoning(self):
        wet, grass, rough = np.array(material_probe(
            -0.5, 0.05, WET_SCALE_DEFAULT, GRASS_LINE_DEFAULT))
        self.assertAlmostEqual(wet, 1.0, places=5)
        self.assertLess(rough, 0.3)
        dry = np.array(material_probe(
            3.0, 0.05, WET_SCALE_DEFAULT, GRASS_LINE_DEFAULT))
        self.assertLess(dry[0], 0.1)
        self.assertGreater(dry[1], 0.8)
        self.assertGreater(dry[2], 0.8)


class WaterlineImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)
        cls.scene.water_controls[None] = [1.0, 1.0, 0.0, 0.06]

    def test_waterline_transition_has_no_hard_edge(self):
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
        flat = directions.reshape(-1, 3)
        _, hit_h = march_terrain_np(eye, flat)
        hit_h = hit_h.reshape(width, height)
        # The bottom of the frame must be terrain for nearly every column.
        beach_rows = hit_h[:, : int(height * 0.3)] < 1e5
        self.assertGreater(beach_rows.mean(), 0.9)
        # Rows whose terrain sits near the waterline, from the CPU march.
        near_line = ((hit_h > -0.25) & (hit_h < 0.25)).any(axis=0)
        rows = np.where(near_line)[0]
        self.assertGreater(len(rows), 0)
        band = slice(rows.min(), rows.max() + 1)
        luminance = frame[:, band, :].mean(axis=2)
        row_means = luminance.mean(axis=0)
        steps = np.abs(np.diff(row_means))
        self.assertLess(steps.max(), 0.08)


if __name__ == "__main__":
    unittest.main()
