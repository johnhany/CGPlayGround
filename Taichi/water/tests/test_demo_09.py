"""PBF tank core stability, neighbor search, obstacle projection, rendering."""

import math
import unittest

import numpy as np
import taichi as ti

from water.demo_09 import (DEPTH_FAR, H_, OBSTACLE_CENTER, OBSTACLE_EXTENT,
                           TANK_PRESETS, TANK_X, TANK_Y, TANK_Z, TankCamera,
                           TankFluid, TankRenderer, apply_tank_preset,
                           poly6_value_py, spiky_gradient_py)


@ti.kernel
def kernel_poly6(distance: ti.f32, h: ti.f32) -> ti.f32:
    return poly6_value(distance, h)


@ti.kernel
def kernel_spiky(offset: ti.types.vector(3, ti.f32), h: ti.f32) -> ti.types.vector(3, ti.f32):
    return spiky_gradient(offset, h)


from water.demo_09 import poly6_value, spiky_gradient  # noqa: E402  (probe kernels above)


class KernelMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_poly6_matches_python_mirror(self):
        for distance in (0.0, 0.21, 0.55, 0.88, 1.09, 1.1, 1.7):
            got = float(kernel_poly6(distance, H_))
            want = poly6_value_py(distance, H_)
            self.assertAlmostEqual(got, want, places=6, msg=f"r={distance}")

    def test_spiky_gradient_matches_python_mirror(self):
        for offset in ([0.3, -0.4, 0.5], [-0.88, 0.0, 0.0], [0.0, 0.0, 0.0],
                       [1.2, 0.1, 0.0], [0.05, 0.06, 0.07]):
            got = np.array(kernel_spiky(ti.Vector(offset), H_), dtype=float)
            want = spiky_gradient_py(offset, H_)
            np.testing.assert_allclose(got, want, atol=1e-6, err_msg=f"r={offset}")

    def test_rest_density_matches_lattice_analytics(self):
        fluid = TankFluid(spacing=0.05)
        spacing_su = fluid.spacing
        # Interior particles have 6 axis neighbours, face-interior 5; the
        # measure kernel averages particles with more than 4 neighbours.
        nx, ny, nz = fluid.nx, fluid.ny, fluid.nz
        interior = (nx - 2) * (ny - 2) * (nz - 2)
        face = 2 * ((nx - 2) * (ny - 2) + (nx - 2) * (nz - 2) + (ny - 2) * (nz - 2))
        expected = (interior * 6 + face * 5) / (interior + face)
        expected *= poly6_value_py(spacing_su, H_)
        self.assertAlmostEqual(float(fluid.rho_rest[None]), expected, delta=0.01)


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.fluid = TankFluid(spacing=0.05)

    def setUp(self):
        self.fluid.reset()

    def domain_asserts(self, fluid):
        lu = fluid.length_unit
        pos = fluid.positions.to_numpy() * lu
        vel = fluid.velocities.to_numpy() * lu
        self.assertTrue(np.isfinite(pos).all())
        self.assertTrue(np.isfinite(vel).all())
        margin = fluid.particle_radius * lu
        self.assertGreaterEqual(pos[:, 0].min(), -1e-6)
        self.assertLessEqual(pos[:, 0].max(), TANK_X + 1e-6)
        self.assertLessEqual(pos[:, 1].max(), 1.4 + margin + 1e-6)
        self.assertGreaterEqual(pos[:, 2].min(), -1e-6)
        self.assertLessEqual(pos[:, 2].max(), TANK_Z + 1e-6)
        self.assertLess(np.linalg.norm(vel, axis=1).max(), 6.0 + 1e-6)

    def test_dam_break_stays_in_domain(self):
        fluid = self.fluid
        for _ in range(6):
            fluid.advance(1.0 / 60.0)
            self.domain_asserts(fluid)

    def test_settles_into_a_calm_pool(self):
        fluid = self.fluid
        fluid.advance(3.0)
        self.domain_asserts(fluid)
        lu = fluid.length_unit
        pos = fluid.positions.to_numpy() * lu
        vel = fluid.velocities.to_numpy() * lu
        # The column is 0.8 m tall over a 0.6x0.6 m footprint; spread over
        # the 2.0x1.0 m floor the resting depth is roughly 0.14 m.
        self.assertLess(np.percentile(pos[:, 1], 99), 0.45)
        self.assertLess(np.linalg.norm(vel, axis=1).max(), 0.8)
        self.assertLess(float(fluid.foam.to_numpy().mean()), 0.1)

    def test_repeated_runs_agree_over_one_substep(self):
        # Grid-fill slot order varies with parallel scheduling, so long runs
        # diverge chaotically; a single substep must still agree closely.
        fluid = self.fluid
        fluid.advance(1.0 / 60.0)
        reference = fluid.positions.to_numpy().copy()
        fluid.reset()
        fluid.advance(1.0 / 60.0)
        np.testing.assert_allclose(reference, fluid.positions.to_numpy(), atol=1e-4)

    def test_advance_uses_fixed_substep_budget(self):
        fluid = self.fluid
        before = int(fluid.step_counter[None])
        fluid.advance(1.0 / 60.0)
        self.assertEqual(int(fluid.step_counter[None]), before + 1)
        fluid.advance(0.5)
        expected = before + 1 + max(1, int(round(0.5 / fluid.substep_dt())))
        self.assertEqual(int(fluid.step_counter[None]), expected)

    def test_neighbor_search_matches_bruteforce(self):
        fluid = self.fluid
        rng = np.random.default_rng(9)
        pos = fluid.positions.to_numpy()
        counts = fluid.nb_count.to_numpy()
        lists = fluid.nb_list.to_numpy()
        radius = fluid.neighbor_radius
        for index in rng.choice(fluid.count, 48, replace=False):
            delta = pos - pos[index]
            distance = np.linalg.norm(delta, axis=1)
            brute = set(np.nonzero((distance < radius) & (np.arange(fluid.count) != index))[0])
            got = set(lists[index, :counts[index]].tolist())
            self.assertEqual(got, brute, f"particle {index}")

    def test_particle_inside_pillar_is_pushed_out(self):
        fluid = self.fluid
        lu = fluid.length_unit
        victim = 0
        inside = np.array(OBSTACLE_CENTER, dtype=float) / lu
        fluid.positions[victim] = inside
        fluid.old_positions[victim] = inside
        fluid.velocities[victim] = [0.0, 0.0, 0.0]
        fluid.substep(fluid.substep_dt())
        pos = np.array(fluid.positions[victim], dtype=float) * lu
        extent = np.array(OBSTACLE_EXTENT, dtype=float)
        center = np.array(OBSTACLE_CENTER, dtype=float)
        q = np.abs(pos - center)
        inside_box = bool((q < extent - 0.01).all())
        self.assertFalse(inside_box)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = TankRenderer(64, 40, samples=1, spacing=0.05)

    def setUp(self):
        self.renderer.fluid.reset()
        self.camera = TankCamera()
        apply_tank_preset(self.camera, "overview")

    def draw(self, clock=0.0):
        self.renderer.draw(self.camera, clock, clarity=1.4, lighting=1.0, exposure=1.05)
        return self.renderer.image.to_numpy()

    def test_prepare_draws_finite_frame_without_window(self):
        self.renderer.fluid.advance(0.3)
        self.renderer.prepare(self.camera, 0.3)
        frame = self.renderer.image.to_numpy()
        self.assertTrue(np.isfinite(frame).all())
        self.assertGreater(frame.std(), 0.04)

    def test_paused_render_is_stable(self):
        # Float atomic splat sums are order-dependent under parallel
        # scheduling, so re-rendering is stable within rounding, not bitwise.
        self.renderer.fluid.advance(0.4)
        first = self.draw(clock=0.4)
        second = self.draw(clock=0.4)
        self.assertLess(float(np.abs(first - second).max()), 0.02)

    def test_all_presets_render_finite_frames(self):
        self.renderer.fluid.advance(0.5)
        for name in TANK_PRESETS:
            apply_tank_preset(self.camera, name)
            frame = self.draw(clock=0.5)
            self.assertTrue(np.isfinite(frame).all(), name)
            self.assertGreater(frame.std(), 0.04, name)

    def test_surface_pixels_differ_from_background(self):
        self.renderer.fluid.advance(0.2)
        eye, forward, right, up = (ti.Vector(v, dt=ti.f32) for v in self.camera.basis())
        scale = math.tan(math.radians(self.camera.fov * 0.5))
        self.renderer.water_depth.fill(DEPTH_FAR)
        self.renderer._render_bg(eye, forward, right, up, scale, 0.2, 1.0)
        self.renderer._splat_particles(eye, forward, right, up, scale, 1.0)
        depth = self.renderer.water_depth.to_numpy()
        covered = int((depth < DEPTH_FAR).sum())
        self.assertGreater(covered, 40)
        # Shading writes water over the background on covered pixels.
        self.renderer._smooth_depth()
        self.renderer._smooth_depth_pass2()
        self.renderer._shade_water(eye, forward, right, up, scale, 1.0, 1.4)
        hdr = self.renderer.post.hdr.to_numpy()
        bg = self.renderer.bg_rgb.to_numpy()
        mask = depth < DEPTH_FAR
        self.assertGreater(float(np.abs(hdr - bg)[mask].mean()), 1e-4)

    def test_splat_respects_background_occlusion(self):
        # With a near depth everywhere, every particle is behind a solid and
        # no water may be written; with a far depth, the splat proceeds.
        eye, forward, right, up = (ti.Vector(v, dt=ti.f32) for v in self.camera.basis())
        scale = math.tan(math.radians(self.camera.fov * 0.5))
        self.renderer.water_depth.fill(DEPTH_FAR)
        self.renderer.bg_depth.fill(0.1)
        self.renderer._splat_particles(eye, forward, right, up, scale, 1.0)
        depth = self.renderer.water_depth.to_numpy()
        self.assertFalse((depth < DEPTH_FAR).any())
        self.renderer.bg_depth.fill(DEPTH_FAR)
        self.renderer._splat_particles(eye, forward, right, up, scale, 1.0)
        depth = self.renderer.water_depth.to_numpy()
        self.assertGreater(int((depth < DEPTH_FAR).sum()), 40)

    def test_reset_restores_initial_column(self):
        fluid = self.renderer.fluid
        fluid.advance(1.0)
        fluid.reset()
        lu = fluid.length_unit
        pos = fluid.positions.to_numpy() * lu
        self.assertAlmostEqual(float(pos[:, 1].max()), 0.80, delta=0.03)
        self.assertGreater(float(fluid.rho_rest[None]), 0.2)


if __name__ == "__main__":
    unittest.main()
