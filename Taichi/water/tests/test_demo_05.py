"""Spectrum, FFT transform, LOD, and rendering checks for the wind-sea demo."""

from contextlib import redirect_stderr
from io import StringIO
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import taichi as ti

from water.demo_01 import OrbitCamera
from water.demo_03 import GRAVITY, OCEAN_PRESETS, apply_preset
from water.demo_05 import (FFTOcean, FFTLevel, LEVEL_SIZES, SEA_PRESETS,
                           TILE_LENGTH, WindOceanRenderer, main, parse_args,
                           sea_parameters)


@ti.kernel
def level_probe(level: ti.template(), x: ti.f32, z: ti.f32) -> ti.types.vector(4, ti.f32):
    height = level.sample_height(x, z)
    grad = level.sample_grad(x, z)
    foam = level.sample_foam(x, z)
    return ti.Vector([height, grad.x, grad.y, foam])


@ti.kernel
def ocean_probe(ocean: ti.template(), x: ti.f32, z: ti.f32) -> ti.types.vector(4, ti.f32):
    height = ocean.sample_height(x, z)
    grad = ocean.sample_grad_lod(x, z)
    foam = ocean.sample_foam_lod(x, z)
    return ti.Vector([height, grad.x, grad.y, foam])


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


@ti.kernel
def lod_probe(ocean: ti.template(), footprint: ti.f32,
              result: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for i, j in ti.ndrange(result.shape[0], result.shape[1]):
        g = ocean.sample_grad_lod((i + 0.25) * TILE_LENGTH / result.shape[0],
                                 (j + 0.75) * TILE_LENGTH / result.shape[1], footprint)
        result[i, j, 0], result[i, j, 1] = g.x, g.y


def numpy_idft(delta_index, n):
    """Unnormalized inverse DFT of a unit delta, matching the kernel sign."""
    grid = np.zeros((n, n), dtype=complex)
    grid[delta_index] = 1.0
    return np.fft.ifft2(grid) * n * n


class FFTTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.level = FFTLevel(16, TILE_LENGTH, 7)

    def _load_delta(self, index):
        zeros = np.zeros((16, 16), dtype=np.float32)
        for field in (self.level.ha_re, self.level.ha_im, self.level.dxa_re,
                      self.level.dxa_im, self.level.dza_re, self.level.dza_im,
                      self.level.hb_re, self.level.hb_im, self.level.dxb_re,
                      self.level.dxb_im, self.level.dzb_re, self.level.dzb_im):
            field.from_numpy(zeros)
        delta = np.zeros((16, 16), dtype=np.float32)
        delta[index] = 1.0
        self.level.ha_re.from_numpy(delta)
        self.level.debug_ifft()

    def test_dc_delta_is_constant(self):
        self._load_delta((0, 0))
        self.assertAlmostEqual(float(self.level.ha_re.to_numpy().min()), 1.0, places=4)
        self.assertAlmostEqual(float(self.level.ha_re.to_numpy().max()), 1.0, places=4)
        self.assertAlmostEqual(float(abs(self.level.ha_im.to_numpy()).max()), 0.0, places=4)

    def test_delta_transform_matches_numpy_idft(self):
        for index in ((2, 3), (5, 0), (7, 7)):
            self._load_delta(index)
            reference = numpy_idft(index, 16)
            np.testing.assert_allclose(self.level.ha_re.to_numpy(), reference.real, atol=1e-4)
            np.testing.assert_allclose(self.level.ha_im.to_numpy(), reference.imag, atol=1e-4)

    def test_linearity_via_sum_of_deltas(self):
        zeros = np.zeros((16, 16), dtype=np.float32)
        fields = (self.level.ha_re, self.level.ha_im,
                  self.level.hb_re, self.level.hb_im)

        def transform_of(delta_index):
            # The transform is in place, so every input must start from a
            # clean slate across both complex buffers.
            for field in fields:
                field.from_numpy(zeros)
            single = np.zeros((16, 16), dtype=np.float32)
            single[delta_index] = 1.0
            self.level.ha_re.from_numpy(single)
            self.level.debug_ifft()
            return (self.level.ha_re.to_numpy().copy(),
                    self.level.ha_im.to_numpy().copy())

        first_re, first_im = transform_of((1, 1))
        second_re, second_im = transform_of((2, 5))
        both = np.zeros((16, 16), dtype=np.float32)
        both[1, 1], both[2, 5] = 1.0, 1.0
        for field in fields:
            field.from_numpy(zeros)
        self.level.ha_re.from_numpy(both)
        self.level.debug_ifft()
        np.testing.assert_allclose(self.level.ha_re.to_numpy(), first_re + second_re, atol=1e-3)
        np.testing.assert_allclose(self.level.ha_im.to_numpy(), first_im + second_im, atol=1e-3)


class SpectrumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.level = FFTLevel(64, TILE_LENGTH, 23)

    def test_odd_stage_fft_matches_numpy(self):
        level = FFTLevel(32, TILE_LENGTH, 23)
        rng = np.random.default_rng(31)
        values = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(np.complex64)
        level.ha_re.from_numpy(values.real.copy())
        level.ha_im.from_numpy(values.imag.copy())
        level.debug_ifft()
        expected = np.fft.ifft2(values) * 32 ** 2
        np.testing.assert_allclose(level.ha_re.to_numpy(), expected.real, atol=5e-4)
        np.testing.assert_allclose(level.ha_im.to_numpy(), expected.imag, atol=5e-4)

    def test_dispersion_table_uses_deep_water_relation(self):
        self.level.build_spectrum(8.0, 35.0, 1.2, 0.6, False, 23)
        table = self.level.table
        np.testing.assert_allclose(table["omega"] ** 2, GRAVITY * table["k"], rtol=1e-5)

    def test_height_std_matches_significant_wave_height(self):
        for height in (0.4, 1.2, 2.6):
            self.level.build_spectrum(10.0, 35.0, height, 0.5, False, 23)
            self.level.update(0.0, 0.5)
            std = float(self.level.height.to_numpy().std())
            self.assertAlmostEqual(std, height / 4.0, delta=0.2 * height / 4.0)

    def test_surface_is_real_and_matches_explicit_idft(self):
        self.level.build_spectrum(8.0, 35.0, 1.2, 0.6, False, 23)
        self.level.set_spectra(1.7, 0.6)
        spectrum_re, spectrum_im = (self.level.ha_re.to_numpy().copy(),
                                    self.level.ha_im.to_numpy().copy())
        self.level.update(1.7, 0.6)
        height = self.level.height.to_numpy()
        n = self.level.n
        index = np.arange(n)
        mapped = np.where(index < n // 2, index, index - n)
        phase = 2j * np.pi * np.outer(mapped, index) / n
        basis = np.exp(phase)
        explicit = np.real(basis.T @ (spectrum_re + 1j * spectrum_im) @ basis)
        np.testing.assert_allclose(height, explicit, atol=1e-3)
        self.assertLess(float(abs(self.level.ha_im.to_numpy()).max()), 1e-3)

    def test_swell_prefers_long_waves_and_short_wave_slope_share_is_bounded(self):
        self.level.build_spectrum(8, 35, 1.3, 0.6, False, 23)
        k, density = self.level.table["k"], self.level.table["sigma2"]
        slope = k * k * density
        short = k > 2 * math.pi / 25
        self.assertLess(float(slope[short].sum() / slope.sum()), 0.75)
        self.level.build_spectrum(6, 35, 0.9, 0.15, True, 23)
        k, density = self.level.table["k"], self.level.table["sigma2"]
        self.assertLess(float(density[k > 2 * math.pi / 25].sum() / density.sum()), 0.1)

    def test_rebuild_is_deterministic(self):
        self.level.build_spectrum(8.0, 35.0, 1.2, 0.6, False, 23)
        self.level.update(0.4, 0.6)
        first = self.level.height.to_numpy().copy()
        self.level.build_spectrum(8.0, 35.0, 1.2, 0.6, False, 23)
        self.level.update(0.4, 0.6)
        np.testing.assert_array_equal(self.level.height.to_numpy(), first)

    def test_chop_zero_disables_foam(self):
        self.level.build_spectrum(18.0, 35.0, 2.6, 0.0, False, 23)
        self.level.update(0.8, 0.0)
        self.assertAlmostEqual(float(self.level.foam.to_numpy().max()), 0.0, places=6)


class OceanTileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.ocean = FFTOcean(seed=23)

    def test_tile_periods_match_grid(self):
        self.ocean.rebuild(8.0, 35.0, 1.3, 0.6, False)
        for x, z in ((0.0, 0.0), (13.7, -41.2), (-255.0, 255.0)):
            # The finest height remains periodic across both tile axes.
            base = float(ocean_probe(self.ocean, x, z)[0])
            self.assertAlmostEqual(base, float(ocean_probe(self.ocean, x + TILE_LENGTH, z)[0]), places=4)
            self.assertAlmostEqual(base, float(ocean_probe(self.ocean, x, z - TILE_LENGTH)[0]), places=4)

    def test_continuous_sampling_matches_quadratic_height_and_derivatives(self):
        level = self.ocean.levels[0]
        x = (np.arange(level.n) - level.n / 2) * level.cell
        z = x.copy()
        grid = (0.01 * x[:, None] ** 2 + 0.02 * z[None, :] ** 2
                + 0.015 * x[:, None] * z[None, :]).astype(np.float32)
        level.height.from_numpy(grid)
        for px, pz in [(3.37, -8.91), (0.9999, 1.4), (1.0001, 1.4)]:
            result = np.array(level_probe(level, px, pz))
            expected = [0.01 * px ** 2 + 0.02 * pz ** 2 + 0.015 * px * pz,
                        0.02 * px + 0.015 * pz, 0.04 * pz + 0.015 * px]
            np.testing.assert_allclose(result[:3], expected, atol=2e-5)
        self.ocean.rebuild(8.0, 35.0, 1.3, 0.6, False)

    def test_lods_preserve_master_fourier_phases_and_reduce_variance(self):
        self.ocean.rebuild(8, 35, 1.3, 0.6, False)
        self.ocean.update(2.0)
        master = self.ocean.levels[0]
        fine = np.fft.fft2(master.height.to_numpy()) / master.n ** 2
        previous = float(master.table["sigma2"].sum())
        for level in self.ocean.levels[1:]:
            n = level.n
            index = (np.fft.fftfreq(n) * n).astype(int) % master.n
            reference = fine[np.ix_(index, index)]
            actual = np.fft.fft2(level.height.to_numpy()) / n ** 2
            mask = (abs(actual) > 1e-5) & (abs(reference) > 1e-5)
            phases = np.angle(actual[mask] * np.conjugate(reference[mask]))
            self.assertGreater(mask.sum(), 10)
            self.assertLess(abs(phases).max(), 0.003)
            variance = float(level.table["sigma2"].sum())
            self.assertLess(variance, previous)
            previous = variance

    def test_footprint_lod_filters_without_world_origin_bands(self):
        self.ocean.rebuild(8, 35, 1.3, 0.6, False)
        first = np.array(ocean_probe(self.ocean, 47.3, 81.2))
        translated = np.array(ocean_probe(self.ocean, 47.3 + TILE_LENGTH, 81.2))
        np.testing.assert_allclose(first, translated, atol=1e-4)
        near, far = np.zeros((64, 64, 2), np.float32), np.zeros((64, 64, 2), np.float32)
        lod_probe(self.ocean, 0.0, near)
        lod_probe(self.ocean, 30.0, far)
        self.assertLess(np.mean(far ** 2), np.mean(near ** 2) * 0.5)

    def test_sample_at_grid_node_hits_cell_value(self):
        self.ocean.rebuild(8.0, 35.0, 1.3, 0.6, False)
        level = self.ocean.levels[0]
        grid = level.height.to_numpy()
        x = -TILE_LENGTH / 2 + 37 * level.cell
        z = -TILE_LENGTH / 2 + 11 * level.cell
        self.assertAlmostEqual(float(ocean_probe(self.ocean, x, z)[0]), grid[37, 11], places=4)

    def test_coarse_levels_carry_less_slope_energy(self):
        self.ocean.rebuild(12.0, 35.0, 2.0, 0.5, False)
        slopes = [float(level.grad.to_numpy().std()) for level in self.ocean.levels]
        self.assertGreater(slopes[0], slopes[1])
        self.assertGreater(slopes[1], slopes[2])
        self.assertEqual([level.n for level in self.ocean.levels], list(LEVEL_SIZES))

    def test_rebuild_rescales_amplitude(self):
        self.ocean.rebuild(3.0, 35.0, 0.35, 0.25, False)
        calm = float(self.ocean.amp_scale[None])
        self.ocean.rebuild(18.0, 35.0, 2.6, 0.9, False)
        storm = float(self.ocean.amp_scale[None])
        self.assertGreater(storm, 3.0 * calm)

    def test_gale_produces_crest_foam(self):
        self.ocean.rebuild(18.0, 35.0, 2.6, 0.9, False)
        self.ocean.update(2.0)
        finest = self.ocean.levels[0].foam.to_numpy()
        self.assertGreater(float(finest.max()), 0.1)
        self.assertLess(float((finest > 0.1).mean()), 0.2)
        self.ocean.rebuild(8.0, 35.0, 1.3, 0.6, False)
        self.ocean.update(2.0)
        self.assertLess(float(self.ocean.levels[0].foam.to_numpy().max()), 0.3)


class RendererTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.renderer = WindOceanRenderer(64, 40, samples=1)

    def test_presets_produce_finite_nonblank_frames(self):
        camera = OrbitCamera()
        for lighting in (0.0, 1.0):
            for preset in ("overview", "waterline", "top"):
                apply_preset(camera, preset)
                self.renderer.draw(camera, 1.0, lighting=lighting)
                frame = self.renderer.image.to_numpy()
                self.assertTrue(np.isfinite(frame).all(), (preset, lighting))
                self.assertGreater(frame.std(), 0.04, (preset, lighting))
                self.assertGreaterEqual(frame.min(), 0.0)
                self.assertLessEqual(frame.max(), 1.0)

    def test_paused_state_is_deterministic(self):
        camera = OrbitCamera()
        apply_preset(camera, "overview")
        self.renderer.draw(camera, 2.0, lighting=0.0)
        first = self.renderer.image.to_numpy()
        self.renderer.draw(camera, 2.0, lighting=0.0)
        np.testing.assert_array_equal(first, self.renderer.image.to_numpy())
        self.renderer.draw(camera, 2.5, lighting=0.0)
        self.assertGreater(np.abs(self.renderer.image.to_numpy() - first).max(), 0.005)

    def test_sea_rebuild_changes_the_frame(self):
        camera = OrbitCamera()
        apply_preset(camera, "waterline")
        self.renderer.refresh_ocean(3.0, 35.0, 0.35, 0.25, False)
        self.renderer.draw(camera, 1.0, lighting=0.0)
        calm = self.renderer.image.to_numpy()
        self.renderer.refresh_ocean(18.0, 35.0, 2.6, 0.9, False)
        self.renderer.draw(camera, 1.0, lighting=0.0)
        storm = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(storm - calm).mean(), 0.01)

    def test_foam_slider_changes_the_frame(self):
        camera = OrbitCamera()
        apply_preset(camera, "waterline")
        self.renderer.refresh_ocean(18.0, 35.0, 2.6, 0.9, False)
        # Coherent spectra move whitecaps relative to the old independent
        # grids. Aim at an actual crest instead of assuming one is by the pier.
        self.renderer.ocean.update(1.0)
        level = self.renderer.ocean.levels[0]
        foam = level.foam.to_numpy()
        i, j = np.unravel_index(foam.argmax(), foam.shape)
        self.assertGreater(foam[i, j], 0.1)
        camera.target = np.array([(i - level.n / 2) * level.cell,
                                  level.height.to_numpy()[i, j],
                                  (j - level.n / 2) * level.cell])
        camera.distance, camera.pitch = 12.0, 1.1
        self.renderer.water_controls[None] = [0.05, 0.5, 0.0, 1.0]
        self.renderer.draw(camera, 1.0, lighting=0.0)
        bare = self.renderer.image.to_numpy()
        self.renderer.water_controls[None] = [0.05, 0.5, 2.0, 1.0]
        self.renderer.draw(camera, 1.0, lighting=0.0)
        foamy = self.renderer.image.to_numpy()
        self.assertGreater(np.abs(foamy - bare).mean(), 0.003)

    def test_wave_surface_normal_matches_finite_differences(self):
        self.renderer.refresh_ocean(8.0, 35.0, 1.3, 0.6, False)
        x, z, t, eps = 2.0, 1.5, 0.8, 0.02
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
        self.assertGreater(cosine, 0.99999, (normal, reference))

    def test_bounds_include_extrema_and_refined_hit_normal_is_consistent(self):
        renderer = self.renderer
        renderer.refresh_ocean(8, 35, 1.3, 0.6, False)
        for clock in (0.0, 8.0):
            renderer.ocean.update(clock)
            bounds = np.array(bounds_probe(renderer))
            heights = renderer.ocean.levels[0].height.to_numpy()
            self.assertLess(bounds[0], heights.min())
            self.assertGreater(bounds[1], heights.max())
            origin = np.array([3.2, 5.0, -2.7], np.float32)
            direction = np.array([0.12, -1.0, 0.17], np.float32)
            direction /= np.linalg.norm(direction)
            hit = np.array(hit_probe(renderer, origin, direction, clock))
            self.assertLess(hit[0], 100)
            point = origin + direction * hit[0]
            reference = np.array(surface_probe(renderer, point[0], point[2], clock))
            self.assertAlmostEqual(float(point[1]), reference[0], delta=0.002)
            np.testing.assert_allclose(hit[1:], reference[1:], atol=2e-5)
        renderer._fft_clock = None

    def test_argument_validation(self):
        for argv in (["--wind", "30"], ["--wave-height", "0.01"], ["--chop", "1.5"],
                     ["--time", "nan"], ["--window-frames", "-1"], ["--wind-dir", "inf"]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def test_sea_parameters(self):
        args = parse_args(["--sea", "gale"])
        self.assertEqual(sea_parameters(args), SEA_PRESETS["gale"])
        args = parse_args(["--sea", "swell", "--swell"])
        wind, direction, height, chop, swell = sea_parameters(args)
        self.assertTrue(swell)
        self.assertEqual((wind, direction, height, chop), SEA_PRESETS["swell"][:4])
        args = parse_args(["--wind", "9", "--wind-dir", "120", "--wave-height", "1.1",
                           "--chop", "0.3", "--sea", "auto"])
        self.assertEqual(sea_parameters(args), (9.0, 120.0, 1.1, 0.3, False))


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

        with patch("water.demo_05.ti.init"), \
             patch("water.demo_05.ti.sync", side_effect=lambda: events.append("sync")), \
             patch("water.demo_05.WindOceanRenderer", return_value=renderer), \
             patch("water.demo_05.ti.ui.Window", side_effect=create_window), \
             patch("builtins.print"):
            main(["--window-frames", "1", "--time", "0"])
        self.assertEqual(events, ["render", "sync", "create_window", "image", "show"])


if __name__ == "__main__":
    unittest.main()
