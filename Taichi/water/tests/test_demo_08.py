"""Ballistic droplets, splash coupling with the shallow-water grid, occlusion."""

import math
import unittest

import numpy as np
import taichi as ti

from water.demo_06 import SIM_DT
from water.demo_08 import (DROP_CAP, CROWN_CAP, GRAVITY, SPLASH_PRESETS, SplashRenderer,
                           advect_drop_cpu, apply_splash_preset)
from water.demo_07 import DiveCamera, bed_height_py


@ti.kernel
def visible_probe(renderer: ti.template(), eye: ti.types.vector(3, ti.f32),
                  position: ti.types.vector(3, ti.f32), clock: ti.f32) -> ti.f32:
    return renderer.drop_visible(eye, position, clock)


def seed_drop(renderer, index, position, velocity, life=30.0, size=0.02):
    renderer.drop_pos[index] = position
    renderer.drop_prev[index] = position
    renderer.drop_vel[index] = velocity
    renderer.drop_normal[index] = [0.0, 1.0, 0.0]
    renderer.drop_aux[index] = [0.0, life, size, 0.5]
    renderer.drop_alive[index] = 1
    renderer.drop_land_flag[index] = 0


class AdvectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SplashRenderer(32, 20, samples=1)
        cls.renderer.sea.reset()

    def test_kernel_matches_cpu_mirror(self):
        renderer = self.renderer
        renderer.reset_drops()
        cases = [([ -22.0, 5.0, 0.0], [0.8, 3.1, -0.4]),
                 ([ -20.0, 8.0, 3.0], [-1.2, 0.4, 0.9]),
                 ([ -24.0, 3.0, -6.0], [0.1, -0.3, 0.2])]
        for index, (position, velocity) in enumerate(cases):
            seed_drop(renderer, index, position, velocity)
            renderer._advect_drops(SIM_DT, 0.0)
            got_p = np.array(renderer.drop_pos[index], dtype=float)
            got_v = np.array(renderer.drop_vel[index], dtype=float)
            want_p, want_v = advect_drop_cpu(position, velocity, SIM_DT)
            np.testing.assert_allclose(got_p, want_p, atol=1e-6)
            np.testing.assert_allclose(got_v, want_v, atol=1e-6)
            self.assertEqual(int(renderer.drop_alive[index]), 1)
        renderer.reset_drops()

    def test_drop_lands_and_flags_ripple(self):
        renderer = self.renderer
        renderer.reset_drops()
        renderer.sea.reset()
        surface = float(renderer.surface_probe(-22.0, 0.0, 0.0))
        seed_drop(renderer, 0, [-22.0, surface + 0.5, 0.0], [0.0, -0.5, 0.0])
        for _ in range(40):
            if int(renderer.drop_alive[0]) == 0:
                break
            renderer._advect_drops(SIM_DT, 0.0)
        self.assertEqual(int(renderer.drop_alive[0]), 0)
        self.assertEqual(int(renderer.drop_land_flag[0]), 1)
        renderer._landed_ripples()
        renderer._apply_landing_foam()
        self.assertEqual(int(renderer.drop_land_flag[0]), 0)
        renderer.reset_drops()

    def test_dead_drop_stays_dead(self):
        renderer = self.renderer
        renderer.reset_drops()
        seed_drop(renderer, 0, [-22.0, 5.0, 0.0], [0.0, 1.0, 0.0], life=0.01)
        renderer._advect_drops(SIM_DT, 0.0)
        self.assertEqual(int(renderer.drop_alive[0]), 0)
        renderer.reset_drops()


class BurstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SplashRenderer(32, 20, samples=1)

    def setUp(self):
        self.renderer.sea.reset()
        self.renderer.reset_drops()

    def test_burst_spawns_requested_drops(self):
        renderer = self.renderer
        renderer.splash(-22.0, 0.0, 0.0, energy=1.0, drops=90)
        alive = int(renderer.drop_alive.to_numpy().sum())
        self.assertEqual(alive, 90+CROWN_CAP)
        positions = renderer.drop_pos.to_numpy()[:90]
        self.assertTrue(np.isfinite(positions).all())
        self.assertGreater(positions[:, 1].min(), -0.5)

    def test_splash_deposits_foam_ring(self):
        renderer = self.renderer
        renderer.splash(-22.0, 0.0, 0.0, energy=1.0, drops=8)
        foam = renderer.sea.foam_view.to_numpy()
        gx = int((-22.0 + 64.0) / 0.5)
        gz = int((0.0 + 64.0) / 0.5)
        near = foam[gx - 4:gx + 5, gz - 4:gz + 5]
        self.assertGreater(near.max(), 0.4)
        far = foam[10:30, 10:30]
        self.assertLess(far.max(), 0.05)

    def test_splash_raises_then_drops_smooth_the_surface(self):
        renderer = self.renderer
        before = renderer.sea.eta_view.to_numpy().copy()
        renderer.splash(-22.0, 0.0, 0.0, energy=1.2, drops=60)
        after = renderer.sea.eta_view.to_numpy()
        gx = int((-22.0 + 64.0) / 0.5)
        gz = int((0.0 + 64.0) / 0.5)
        self.assertGreater(after[gx, gz] - before[gx, gz], 0.02)
        # Let every droplet fall; landing dips must register in h.
        h_before = renderer.sea.h.to_numpy().copy()
        renderer.step_drops(400, 0.0)
        self.assertEqual(int(renderer.drop_alive.to_numpy().sum()), 0)
        h_after = renderer.sea.h.to_numpy()
        local = h_after[gx - 3:gx + 4, gz - 3:gz + 4]
        local_before = h_before[gx - 3:gx + 4, gz - 3:gz + 4]
        self.assertGreater(np.abs(local - local_before).sum(), 1e-4)


class VisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SplashRenderer(32, 20, samples=1)
        cls.renderer.sea.reset()

    def test_reef_blocks_droplet_behind_crest(self):
        renderer = self.renderer
        eye = np.array([-14.0, 1.2, -6.0], np.float32)
        behind = np.array([-20.5, 0.8, -12.0], np.float32)
        crest = float(bed_height_py(-18.0, -10.0))
        self.assertGreater(crest, 1.5)
        got = float(visible_probe(renderer, eye, behind, 0.0))
        self.assertEqual(got, 0.0)

    def test_lifted_eye_and_lifted_drop_are_visible(self):
        renderer = self.renderer
        eye = np.array([-14.0, 4.5, -6.0], np.float32)
        behind = np.array([-20.5, 0.8, -12.0], np.float32)
        self.assertEqual(float(visible_probe(renderer, eye, behind, 0.0)), 1.0)
        eye_low = np.array([-14.0, 1.2, -6.0], np.float32)
        above = np.array([-20.5, 3.5, -12.0], np.float32)
        self.assertEqual(float(visible_probe(renderer, eye_low, above, 0.0)), 1.0)


class StoneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SplashRenderer(32, 20, samples=1)
        cls.renderer.sea.reset()

    def test_stone_falls_and_reports_impact(self):
        renderer = self.renderer
        renderer.reset_drops()
        renderer.surface_probe = lambda x, z, clock: np.float32(0.0)
        renderer.stone_wait = 0.1
        self.assertIsNone(renderer.update_stone(0.2, 0.0))
        self.assertIsNotNone(renderer.stone_pos)
        center = np.array(renderer.box_center[renderer.stone_index])
        self.assertAlmostEqual(float(center[1]), 5.0, delta=1e-4)
        impact = None
        for _ in range(200):
            impact = renderer.update_stone(0.1, 0.0)
            if impact is not None:
                break
        self.assertIsNotNone(impact)
        # Free fall from 5 m reaches about sqrt(2 g h).
        self.assertAlmostEqual(impact[2], math.sqrt(2.0 * GRAVITY * 4.8), delta=0.8)
        self.assertTrue(renderer.stone_sinking)
        self.assertIsNotNone(renderer.stone_pos)
        for _ in range(8):
            renderer.update_stone(.1,0.)
        self.assertIsNone(renderer.stone_pos)
        hidden = np.array(renderer.box_extent[renderer.stone_index])
        self.assertLess(float(hidden.max()), 0.01)

    def test_stone_waits_between_drops(self):
        renderer = self.renderer
        renderer.reset_drops()
        renderer.surface_probe = lambda x, z, clock: np.float32(0.0)
        renderer.stone_wait = 5.0
        self.assertIsNone(renderer.update_stone(0.5, 0.0))
        self.assertIsNone(renderer.stone_pos)
        self.assertAlmostEqual(renderer.stone_wait, 4.5, delta=1e-6)


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = SplashRenderer(32, 20, samples=1)
        cls.renderer.sea.set_wave(0.30, 10.0)
        cls.renderer.sea.advance(120)

    def test_paused_state_is_deterministic_with_drops(self):
        renderer = self.renderer
        renderer.reset_drops()
        clock = renderer.sea.time
        renderer.splash(-22.0, 0.0, clock, energy=1.2, drops=120)
        camera = DiveCamera()
        apply_splash_preset(camera, "impact")
        renderer.draw(camera, clock, lighting=1.0)
        first = renderer.image.to_numpy()
        renderer.draw(camera, clock, lighting=1.0)
        np.testing.assert_array_equal(first, renderer.image.to_numpy())
        self.assertEqual(int(renderer.splash_accum.to_numpy().sum()), 0)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(first.min(), 0.0)
        self.assertLessEqual(first.max(), 1.0)

    def test_splat_changes_the_frame_and_respects_brightness(self):
        renderer = self.renderer
        renderer.reset_drops()
        clock = renderer.sea.time
        camera = DiveCamera()
        apply_splash_preset(camera, "impact")
        renderer.splash(-22.0, 0.0, clock, energy=1.3, drops=150)
        renderer.step_drops(5,clock)
        renderer.splash_controls[None] = [0.0, 1.0, 0.6, 0.0]
        renderer.draw(camera, clock, lighting=1.0)
        off = renderer.image.to_numpy().copy()
        renderer.splash_controls[None] = [1.5, 1.0, 0.6, 0.0]
        renderer.draw(camera, clock, lighting=1.0)
        on = renderer.image.to_numpy()
        self.assertGreater(np.abs(on - off).max(), 0.01)
        self.assertTrue(np.isfinite(on).all())

    def test_splash_preset_frames_are_finite_and_nonblank(self):
        renderer = self.renderer
        renderer.reset_drops()
        clock = renderer.sea.time
        renderer.splash(-22.0, 0.0, clock, energy=1.0, drops=100)
        camera = DiveCamera()
        for lighting in (1.0, 0.0):
            apply_splash_preset(camera, "impact")
            renderer.draw(camera, clock, lighting=lighting)
            frame = renderer.image.to_numpy()
            self.assertTrue(np.isfinite(frame).all(), lighting)
            self.assertGreater(frame.std(), 0.04, lighting)


class PresetTests(unittest.TestCase):
    def test_presets_keep_the_eye_off_the_bed(self):
        camera = DiveCamera()
        for name in SPLASH_PRESETS:
            apply_splash_preset(camera, name)
            eye = camera.basis()[0]
            self.assertGreaterEqual(float(eye[1]),
                                    bed_height_py(float(eye[0]), float(eye[2])) + 0.29,
                                    name)


@ti.kernel
def path_probe(r:ti.template(),eye:ti.types.vector(3,ti.f32),point:ti.types.vector(3,ti.f32))->ti.types.vector(4,ti.f32):
    ray,length,visible=r.optical_particle(eye,point,0.)
    return ti.Vector([ray.x,ray.y,ray.z,visible])

class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,offline_cache=False)
        cls.r=SplashRenderer(64,40,1)

    def setUp(self):
        self.r.sea.reset();self.r.reset_drops()
        self.r.splash_controls[None]=[1.,1.,0.,0.]

    def test_birth_and_first_step_keep_rising_particles(self):
        r=self.r;r.sea.advance(600);t=r.sea.time
        r.splash(-22.,0.,t,1.5,110)
        p=r.drop_pos.to_numpy()[:110]
        for point in p:
            self.assertGreater(point[1],r.surface_probe(float(point[0]),float(point[2]),t))
        r.advance(1,auto_stone=False)
        self.assertGreaterEqual(r.drop_alive.to_numpy()[:110].sum(),100)
        self.assertEqual(int(r.drop_land_flag.to_numpy().sum()),0)

    def test_ballistic_lifetime_allows_actual_return(self):
        r=self.r;r._burst_drops(-22.,0.,0.,110,1.5)
        landed=0
        for _ in range(150):
            r._advect_drops(SIM_DT,0.)
            landed+=int(r.drop_land_flag.to_numpy().sum())
            r.drop_land_flag.fill(0)
        self.assertEqual(landed,110)

    def test_projected_particle_matches_primary_camera_axis(self):
        r=self.r;c=DiveCamera();apply_splash_preset(c,'impact')
        eye,f,right,up=c.basis();scale=math.tan(math.radians(c.fov/2))
        p=eye+f*6+up*.4
        seed_drop(r,0,p,[0,0,0],size=.12)
        r._prepare_drops(eye,0.)
        r._splat_drops(eye,f,right,up,scale,0.,1.,1.4)
        a=r.splash_weight.to_numpy();x,y=np.indices(a.shape)
        self.assertGreater(a.sum(),0)
        self.assertAlmostEqual(float((x*a).sum()/a.sum()+.5),32.,delta=.4)
        self.assertAlmostEqual(float((y*a).sum()/a.sum()+.5),20+20*.4/(scale*6),delta=.4)

    def test_hdr_compositor_preserves_bright_background(self):
        r=self.r;r.post.hdr.fill(2.);r.splash_accum.fill(.2);r.splash_weight.fill(.1)
        r._composite_splash(1.)
        np.testing.assert_allclose(r.post.hdr.to_numpy(),2.,atol=1e-6)
        r.post.hdr.fill(2.);r.splash_weight.fill(0.);r._composite_splash(1.)
        np.testing.assert_array_equal(r.post.hdr.to_numpy(),2.)

    def test_underwater_transparency_and_snell_projection(self):
        r=self.r;r.prepare_water_bounds()
        self.assertEqual(visible_probe(r,[-22.,-.5,0.],[-21.,-.2,0.],0.),1.)
        got=np.array(path_probe(r,[-22.,-.5,0.],[-21.,1.,0.]))
        self.assertEqual(got[3],1.)
        self.assertLess(got[0],1/1.333)
        # The initial water ray bends closer to the surface normal.
        self.assertLess(got[0]/got[1],1/1.5)

    def test_landing_foam_accumulates_without_lost_updates(self):
        r=self.r
        land=np.zeros((DROP_CAP,3),np.float32);land[:,0]=-22;land[:,2]=.001
        r.drop_land.from_numpy(land);r.drop_land_flag.fill(1)
        r._landed_ripples();r._apply_landing_foam()
        self.assertAlmostEqual(float(r.sea.foam[84,128]),DROP_CAP*.12*.001,delta=2e-5)

    def test_speed_changes_landing_impulse(self):
        r=self.r;values=[]
        for speed in (-1.,-10.):
            r.reset_drops();seed_drop(r,0,[-22.,.001,0.],[0,speed,0],size=.03)
            r._advect_drops(SIM_DT,0.);values.append(float(r.drop_land[0].z))
        self.assertGreater(values[1],values[0]*3)

    def test_event_and_landing_pulses_preserve_water_volume(self):
        r=self.r;before=r.sea.h.to_numpy().astype(float).sum()
        r.splash(-22.,0.,0.,1.2,110)
        after=r.sea.h.to_numpy().astype(float).sum()
        self.assertAlmostEqual(before,after,delta=2e-5)
        r.step_drops(150,0.)
        self.assertAlmostEqual(before,r.sea.h.to_numpy().astype(float).sum(),delta=8e-5)

    def test_underwater_downward_pick_does_not_hit_behind_camera(self):
        r=self.r;c=DiveCamera();apply_splash_preset(c,'bed')
        self.assertIsNone(r.pick_water(c,.5,.5))

    def test_crown_expires_and_emitter_is_in_demo_views(self):
        r=self.r;r.splash(-22.,0.,0.,1.2,110);r.step_drops(7,0.)
        kind=r.drop_kind.to_numpy();alive=r.drop_alive.to_numpy();p=r.drop_pos.to_numpy()
        self.assertGreater(p[(kind==2)&(alive==1),1].max(),.5)
        for name in ('impact','top','overview'):
            c=DiveCamera();apply_splash_preset(c,name);eye,f,right,up=c.basis()
            scale=math.tan(math.radians(c.fov/2))
            for x in (-22.8,-21.2):
                for z in (-.8,.8):
                    q=np.array([x,0.,z])-eye;depth=q@f
                    self.assertGreater(depth,0)
                    self.assertLess(abs(q@right/depth),scale*1.6)
                    self.assertLess(abs(q@up/depth),scale)
        r.step_drops(20,0.)
        self.assertEqual(int(r.drop_alive.to_numpy()[kind==2].sum()),0)

    def test_fixed_substep_batch_matches_single_steps(self):
        r=self.r;r.splash(-22.,0.,0.,1.2,110);r.advance(12,auto_stone=False)
        first=r.drop_pos.to_numpy().copy();h=r.sea.h.to_numpy().copy()
        r.sea.reset();r.reset_drops();r.splash(-22.,0.,0.,1.2,110)
        for _ in range(12):r.advance(1,auto_stone=False)
        np.testing.assert_allclose(first,r.drop_pos.to_numpy(),atol=1e-5)
        np.testing.assert_allclose(h,r.sea.h.to_numpy(),atol=1e-5)


if __name__ == "__main__":
    unittest.main()
