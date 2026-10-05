"""Underwater medium, Snell window, caustics, and dive camera checks."""

import math
import unittest

import numpy as np
import taichi as ti

from water.demo_06 import SIM_DT, shore_terrain, shore_terrain_grad
from water.demo_07 import (CAUSTIC_N, CAUSTIC_DX, CAUSTIC_HALF, CAUSTIC_CENTER_X,
                           dielectric_fresnel,pack_normal,unpack_normal,
                           DIVE_PRESETS, DiveCamera, DiveRenderer,
                           IOR, apply_dive_preset, bed_height_py,
                           dive_escape, dive_escape_cpu)
from water.tests.test_demo_06 import numpy_terrain


@ti.kernel
def escape_probe(direction: ti.types.vector(3, ti.f32),
                 normal: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    tir, transmitted, _reflected = dive_escape(direction, normal)
    return ti.Vector([tir, transmitted.x, transmitted.y, transmitted.z])


@ti.kernel
def exit_probe(renderer: ti.template(), origin: ti.types.vector(3, ti.f32),
               direction: ti.types.vector(3, ti.f32), clock: ti.f32) -> ti.types.vector(4, ti.f32):
    t, normal = renderer.water_exit(origin, direction, clock)
    return ti.Vector([t, normal.x, normal.y, normal.z])


@ti.kernel
def eta_probe(renderer: ti.template(), x: ti.f32, z: ti.f32, clock: ti.f32) -> ti.f32:
    return renderer.sea.sample(x, z, clock).x


@ti.kernel
def body_probe(renderer: ti.template(), thickness: ti.f32, clarity: ti.f32,
               lighting: ti.f32) -> ti.types.vector(3, ti.f32):
    return renderer.water_body(ti.Vector([1.0, 1.0, 1.0]), thickness, clarity, 0.0,
                               ti.Vector([0.0, 1.0, 0.0]), lighting)


@ti.kernel
def density_probe(renderer: ti.template(), x: ti.f32, z: ti.f32) -> ti.f32:
    return renderer.dive_caustic_density(ti.Vector([x, 0.0, z]))


def direction_at(angle_from_vertical, azimuth=0.7):
    """Unit direction angled `angle` away from +Y."""
    s = math.sin(angle_from_vertical)
    return np.array([s * math.cos(azimuth), math.cos(angle_from_vertical),
                     s * math.sin(azimuth)], np.float32)


class EscapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_kernel_matches_cpu_mirror(self):
        normal = np.array([0.05, 0.9987, -0.02], np.float32)
        normal /= np.linalg.norm(normal)
        for angle in (0.05, 0.4, 0.7, 0.84, 0.847, 0.86):
            direction = direction_at(angle)
            got = np.array(escape_probe(direction, normal))
            tir, transmitted, _ = dive_escape_cpu(direction, normal)
            self.assertEqual(got[0], 1.0 if tir else 0.0, angle)
            np.testing.assert_allclose(got[1:], transmitted, atol=1e-5)

    def test_snell_refraction_relation(self):
        normal = np.array([0.0, 1.0, 0.0], np.float32)
        for angle in (0.1, 0.3, 0.6, 0.84):
            direction = direction_at(angle)
            got = np.array(escape_probe(direction, normal))
            self.assertEqual(got[0], 0.0, angle)
            transmitted = got[1:] / np.linalg.norm(got[1:])
            # Snell: sin(theta_air) = IOR * sin(theta_water).
            sin_air = IOR * math.sin(angle)
            cos_air = math.sqrt(max(0.0, 1.0 - sin_air * sin_air))
            self.assertAlmostEqual(float(np.dot(transmitted, normal)), cos_air, delta=1e-3)
            # The transmitted ray bends away from the vertical.
            self.assertGreater(math.acos(min(1.0, float(np.dot(transmitted, normal)))),
                               angle - 1e-3)

    def test_critical_angle_boundary(self):
        normal = np.array([0.0, 1.0, 0.0], np.float32)
        critical = math.asin(1.0 / IOR)
        for angle, expect_tir in ((critical - 0.02, 0.0), (critical + 0.02, 1.0),
                                  (1.05, 1.0), (0.2, 0.0)):
            got = escape_probe(direction_at(angle), normal)
            self.assertEqual(got[0], expect_tir, angle)


class ExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = DiveRenderer(64, 40, samples=1)
        cls.renderer.sea.reset()

    def test_exit_through_still_surface(self):
        origin = np.array([-20.0, -1.0, 0.0], np.float32)
        direction = direction_at(0.7854)
        got = np.array(exit_probe(self.renderer, origin, direction, 0.0))
        # Flat still water rests at y = 0: t = 1 / cos(45 deg).
        self.assertAlmostEqual(got[0], 1.0 / math.cos(0.7854), delta=0.05)
        self.assertGreater(got[2], 0.999, got)

    def test_downward_ray_never_exits(self):
        origin = np.array([-20.0, -1.0, 0.0], np.float32)
        direction = np.array([0.1, -0.7, 0.7], np.float32)
        direction /= np.linalg.norm(direction)
        got = exit_probe(self.renderer, origin, direction, 0.0)
        self.assertGreaterEqual(got[0], 1e5)

    def test_exit_travels_with_waves(self):
        self.renderer.sea.set_wave(0.30, 10.0)
        self.renderer.sea.advance(240)
        origin = np.array([-35.0, -2.0, -5.0], np.float32)
        direction = direction_at(0.5)
        got = np.array(exit_probe(self.renderer, origin, direction,
                                  self.renderer.sea.time))
        # t must land the sample on the local surface: p.y ~= eta there.
        point = origin + direction * got[0]
        eta = float(eta_probe(self.renderer, float(point[0]), float(point[2]),
                              self.renderer.sea.time))
        self.assertAlmostEqual(point[1], eta, delta=0.02)
        self.renderer.sea.reset()


class BodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = DiveRenderer(16, 10, samples=1)

    def test_beer_lambert_matches_closed_form(self):
        for lighting in (0.0, 1.0):
            sun = np.array([-0.50, 0.22 + 0.73 * lighting, -0.57])
            sun /= np.linalg.norm(sun)
            illumination = 0.75 + 0.25 * max(0.0, sun[1])
            coeffs = np.array([0.34, 0.12, 0.065])
            bulk = np.array([0.020, 0.150, 0.135]) * illumination
            for thickness in (0.5, 2.0, 8.0, 12.0):
                got = np.array(body_probe(self.renderer, thickness, 1.4, lighting))
                transmission = np.exp(-coeffs * thickness / 1.4)
                want = transmission + bulk * (1.0 - transmission)
                np.testing.assert_allclose(got, want, atol=1e-5)
                # Red dies first: long paths go blue-green.
                if thickness > 4.0:
                    self.assertGreater(got[1], got[0])
                    self.assertGreater(got[2], got[0])


class CausticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = DiveRenderer(16, 10, samples=1)

    def wet_mask(self):
        axis = -CAUSTIC_HALF + (np.arange(CAUSTIC_N) + 0.5) * CAUSTIC_DX
        bed = numpy_terrain((axis+CAUSTIC_CENTER_X)[:, None], axis[None, :])
        return bed < -0.05

    def test_flat_water_deposits_near_uniform_flux(self):
        self.renderer.sea.reset()
        self.renderer.trace_caustics(0.0, 1.0)
        caustic = self.renderer.dive_caustic_map.to_numpy()
        wet = self.wet_mask()
        self.assertTrue(np.isfinite(caustic).all())
        self.assertGreaterEqual(caustic.min(), 0.0)
        values = caustic[wet]
        self.assertGreater(values.mean(), 0.80)
        self.assertLess(values.mean(), 1.08)
        self.assertLess(values.max(), 2.0)

    def test_waves_focus_caustics_into_filaments(self):
        self.renderer.sea.set_wave(0.30, 10.0)
        self.renderer.sea.advance(240)
        self.renderer.trace_caustics(self.renderer.sea.time, 1.0)
        caustic = self.renderer.dive_caustic_map.to_numpy()
        wet = self.wet_mask()
        self.assertGreater(caustic[wet].max(), 1.15)
        self.assertLessEqual(caustic.max(), 6.0)
        self.assertTrue(np.isfinite(caustic).all())
        self.renderer.sea.reset()

    def test_density_matches_map_bilinear(self):
        self.renderer.trace_caustics(0.0, 1.0)
        caustic = self.renderer.dive_caustic_map.to_numpy()
        for x, z in ((-45.3, -20.7), (-22.6, 14.2), (-50.0, 30.0)):
            u = (x - CAUSTIC_CENTER_X + CAUSTIC_HALF) / CAUSTIC_DX - 0.5
            v = (z + CAUSTIC_HALF) / CAUSTIC_DX - 0.5
            ix, iy = int(math.floor(u)), int(math.floor(v))
            fx, fy = u - ix, v - iy
            want = (caustic[ix, iy] * (1 - fx) * (1 - fy)
                    + caustic[ix + 1, iy] * fx * (1 - fy)
                    + caustic[ix, iy + 1] * (1 - fx) * fy
                    + caustic[ix + 1, iy + 1] * fx * fy)
            self.assertAlmostEqual(float(density_probe(self.renderer, x, z)), want,
                                   delta=1e-4)


class CameraTests(unittest.TestCase):
    def test_bed_mirror_matches_solver_terrain(self):
        for x, z in ((0.0, 0.0), (-18.0, -10.0), (20.5, -19.5), (-44.2, 7.7)):
            self.assertAlmostEqual(bed_height_py(x, z), float(numpy_terrain(x, z)),
                                   delta=1e-6)

    def test_presets_keep_the_eye_off_the_bed(self):
        camera = DiveCamera()
        for name in DIVE_PRESETS:
            apply_dive_preset(camera, name)
            eye = camera.basis()[0]
            self.assertGreaterEqual(float(eye[1]),
                                    bed_height_py(float(eye[0]), float(eye[2])) + 0.29,
                                    name)

    def test_zero_and_small_pan_preserve_underwater_framing(self):
        for name in ('dive','window','bed'):
            camera=DiveCamera();apply_dive_preset(camera,name)
            eye=camera.basis()[0].copy()
            camera.pan(0.,0.)
            np.testing.assert_allclose(camera.basis()[0],eye,atol=1e-5)
            camera.pan(.001,.001)
            self.assertLess(np.linalg.norm(camera.basis()[0]-eye),.02)

    def test_orbit_and_zoom_clamp_extremes(self):
        camera = DiveCamera()
        apply_dive_preset(camera, "dive")
        for pitch in (-1.45, -0.8, 0.0, 0.7, 1.48):
            camera.pitch = pitch
            for distance in (4.5, 25.0, 60.0):
                camera.distance = distance
                camera.lift_above_bed()
                eye = camera.basis()[0]
                self.assertGreaterEqual(float(eye[1]),
                                        bed_height_py(float(eye[0]), float(eye[2])) + 0.29,
                                        (pitch, distance))


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = DiveRenderer(64, 40, samples=1)
        cls.renderer.sea.set_wave(0.30, 10.0)
        cls.renderer.sea.advance(120)

    def test_presets_produce_finite_nonblank_frames(self):
        camera = DiveCamera()
        clock = self.renderer.sea.time
        for lighting in (1.0, 0.0):
            for preset in DIVE_PRESETS:
                apply_dive_preset(camera, preset)
                self.renderer.draw(camera, clock, lighting=lighting)
                frame = self.renderer.image.to_numpy()
                self.assertTrue(np.isfinite(frame).all(), (preset, lighting))
                self.assertGreater(frame.std(), 0.04, (preset, lighting))
                self.assertGreaterEqual(frame.min(), 0.0)
                self.assertLessEqual(frame.max(), 1.0)

    def test_paused_state_is_deterministic(self):
        camera = DiveCamera()
        apply_dive_preset(camera, "dive")
        clock = self.renderer.sea.time
        self.renderer.draw(camera, clock, lighting=1.0)
        first = self.renderer.image.to_numpy()
        self.renderer.draw(camera, clock, lighting=1.0)
        np.testing.assert_array_equal(first, self.renderer.image.to_numpy())
        self.renderer.sea.advance(30)
        self.renderer.draw(camera, self.renderer.sea.time, lighting=1.0)
        self.assertGreater(np.abs(self.renderer.image.to_numpy() - first).max(), 0.005)

    def test_snell_window_is_brighter_than_the_deep_bed(self):
        self.renderer.sea.reset()
        self.renderer.dive_controls[None] = [1.0, 0.0, 0.0, 0.0]
        camera = DiveCamera()
        apply_dive_preset(camera, "window")
        self.renderer.draw(camera, 0.0, lighting=1.0)
        frame = self.renderer.image.to_numpy()
        f=self.renderer._dive_fresnel.to_numpy()[:,:,0]
        mode=self.renderer._dive_mode.to_numpy()[:,:,0]
        sky=(mode==1)&(f<.05)
        mirror=(mode==1)&(f>.999)
        self.assertTrue(sky.any() and mirror.any())
        self.assertGreater(frame[sky].mean(),frame[mirror].mean()+.04)
        self.renderer.sea.set_wave(0.30, 10.0)
        self.renderer.sea.advance(120)

    def test_shaft_and_caustic_sliders_change_the_frame(self):
        camera=DiveCamera()
        r=self.renderer
        clock=r.sea.time
        apply_dive_preset(camera,"bed")
        r.dive_controls[None]=[0.,0.,0.,0.]
        r.draw(camera,clock,lighting=1.);bare=r.image.to_numpy()
        r.dive_controls[None]=[1.,0.,0.,0.]
        r.draw(camera,clock,lighting=1.);caustics=r.image.to_numpy()
        self.assertGreater(np.abs(caustics-bare).max(),.01)
        apply_dive_preset(camera,"window")
        r.dive_controls[None]=[1.,0.,0.,0.]
        r.draw(camera,clock,lighting=1.);bare=r.image.to_numpy()
        r.dive_controls[None]=[1.,.8,0.,0.]
        r.draw(camera,clock,lighting=1.)
        self.assertGreater(np.abs(r.image.to_numpy()-bare).max(),.003)


@ti.kernel
def fresnel_probe(cosine:ti.f32,eta:ti.f32)->ti.f32:
    return dielectric_fresnel(cosine,eta)


@ti.kernel
def normal_roundtrip(normal:ti.types.vector(3,ti.f32))->ti.types.vector(3,ti.f32):
    return unpack_normal(pack_normal(normal))


@ti.kernel
def cached_exit_probe(r:ti.template(),origin:ti.types.vector(3,ti.f32),direction:ti.types.vector(3,ti.f32))->ti.f32:
    return r.water_exit(origin,direction,0.,600.,1)[0]


@ti.kernel
def light_color_probe(r:ti.template(),p:ti.types.vector(3,ti.f32))->ti.types.vector(3,ti.f32):
    return r.surface_sun_color(p,6,1.,ti.Vector([1.,1.,1.]))


@ti.kernel
def material_shadow_field(r:ti.template(),output:ti.template()):
    for i,j in output:
        x,z=-22.+i*.25,-14.+j*.25
        p=ti.Vector([x,shore_terrain(x,z),z])
        grad=shore_terrain_grad(x,z)
        n=ti.Vector([-grad.x,1.,-grad.y]).normalized()
        output[i,j]=r.surface_sun_visibility(p,n,6,1.,1)


class OpticalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu,offline_cache=False)
        cls.renderer=DiveRenderer(32,20,1)

    def setUp(self):
        self.renderer.sea.reset()

    def test_flat_grazing_rays_exit(self):
        r=self.renderer
        r.prepare_water_bounds()
        for elevation in (2.,5.,10.,15.,45.,89.):
            angle=math.radians(elevation)
            d=np.array([0.,math.sin(angle),math.cos(angle)],np.float32)
            want=1/math.sin(angle)
            self.assertAlmostEqual(float(exit_probe(r,[-40.,-1.,0.],d,0.)[0]),want,delta=.0002)
            self.assertAlmostEqual(float(cached_exit_probe(r,[-40.,-1.,0.],d)),want,delta=.0002)

    def test_horizontal_and_downward_rays_hit_tilted_wave(self):
        r=self.renderer
        axis=-64+np.arange(256)*.5
        eta=np.repeat((-.2*(axis+40))[:,None],256,axis=1).astype(np.float32)
        r.sea.eta_view.from_numpy(eta)
        r.sea.eta_min[None],r.sea.eta_max[None]=float(eta.min()),float(eta.max())
        r.prepare_water_bounds()
        for dy in (0.,-.05):
            d=np.array([1.,dy,0.],np.float32);d/=np.linalg.norm(d)
            expected=.1/(.2*d[0]+d[1])
            self.assertAlmostEqual(float(exit_probe(r,[-40.,-.1,0.],d,0.)[0]),expected,delta=.0002)
            self.assertAlmostEqual(float(cached_exit_probe(r,[-40.,-.1,0.],d)),expected,delta=.0002)

    def test_exact_dielectric_reflection_near_critical_angle(self):
        for angle in (0.,30.,45.,48.,48.5):
            ci=math.cos(math.radians(angle))
            ct=math.sqrt(max(0.,1-IOR*IOR*(1-ci*ci)))
            rs=(IOR*ci-ct)/(IOR*ci+ct)
            rp=(ci-IOR*ct)/(ci+IOR*ct)
            self.assertAlmostEqual(float(fresnel_probe(ci,IOR)),.5*(rs*rs+rp*rp),delta=2e-5)
        critical=math.asin(1/IOR)
        self.assertGreater(float(fresnel_probe(math.cos(critical-1e-6),IOR)),.98)
        self.assertEqual(float(fresnel_probe(math.cos(critical+1e-6),IOR)),1.)

    def test_packed_normals_preserve_full_sphere(self):
        rng=np.random.default_rng(77)
        for n in rng.normal(size=(30,3)):
            n/=np.linalg.norm(n)
            np.testing.assert_allclose(normal_roundtrip(n),n,atol=.00012)

    def test_light_attenuates_along_underwater_sun_path(self):
        r=self.renderer
        r.view_clock[None],r.view_clarity[None]=0.,1.4
        shallow=np.array(light_color_probe(r,[-40.,-.1,0.]))
        deep=np.array(light_color_probe(r,[-40.,-2.,0.]))
        self.assertTrue((deep<shallow).all())
        self.assertLess(deep[0],deep[2])

    def test_deep_water_does_not_stop_at_twelve_metres(self):
        a=np.array(body_probe(self.renderer,12.,1.4,1.))
        b=np.array(body_probe(self.renderer,40.,1.4,1.))
        self.assertTrue((b<a).all())
        self.assertGreater(np.linalg.norm(a-b),.2)

    def test_atlas_outside_is_neutral(self):
        self.renderer.dive_caustic_map.fill(4.)
        self.assertAlmostEqual(float(density_probe(self.renderer,-60.,0.)),1.,places=6)
        self.assertAlmostEqual(float(density_probe(self.renderer,15.,0.)),1.,places=6)

    def test_terrain_shadows_survive_foam_control(self):
        r=self.renderer
        output=ti.field(ti.f32,shape=(32,32))
        r.light_controls[None]=[0.,1.,0.,1.]
        r.dive_controls[None]=[0.,0.,0.,0.]
        r.water_controls[None]=[.06,.5,0.,1.]
        material_shadow_field(r,output);bare=output.to_numpy()
        r.water_controls[None]=[.06,.5,1.,1.]
        material_shadow_field(r,output);foam=output.to_numpy()
        self.assertLess(bare.min(),.5)
        self.assertGreater(bare.max(),.99)
        np.testing.assert_array_equal(bare,foam)

    def test_single_sample_packets_allocate_single_sample(self):
        self.assertEqual(self.renderer._dive_mode.shape,(32,20,1))
        self.assertEqual(self.renderer.sample_capacity,1)


class StartupTests(unittest.TestCase):
    def test_both_media_prepare_before_window_creation(self):
        from unittest.mock import MagicMock,patch
        from water.demo_07 import main
        order=[]
        renderer=MagicMock()
        renderer.prepare.side_effect=lambda *a,**k:order.append('prepare')
        window=MagicMock()
        def open_window(*a,**k):
            order.append('window')
            return window
        with patch('water.demo_07.ti.init'),patch('water.demo_07.ti.sync'),\
             patch('water.demo_07.DiveRenderer',return_value=renderer),\
             patch('water.demo_07.ti.ui.Window',side_effect=open_window):
            main(['--time','0','--window-frames','1','--width','32','--height','32'])
        self.assertEqual(order,['prepare','window'])



if __name__ == "__main__":
    unittest.main()
