"""Temporal accumulation checks: static classes settle well below the raw
jitter noise, the water surface recovers from a single-frame perturbation
within a few frames, and the accumulation state resets whenever the clock
does not advance, keeping repeated draws bit-identical."""

import math
import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import JITTER_GX, JITTER_GY, NightLakeScene


class TemporalAccumulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)
        cls.camera = FixedCamera()

    def setUp(self):
        # Every test starts from a fresh accumulation state.
        self.scene._accum_valid = False
        self.scene._last_accum_clock = None
        self.scene.reflection_gain = 1.0
        self.scene.temporal_aa = True
        controls = tuple(self.scene.water_controls[None])
        self.scene.water_controls[None] = [controls[0], 0.0, 0.0, controls[3]]

    def _draw(self, clock):
        self.scene.draw(self.camera, clock=clock)

    def _raw_jittered_frame(self, clock):
        """Render one frame through the same kernel path draw() uses, with
        the frame-counter jitter, without touching the accumulation
        buffer; returns the HDR grid."""
        scene = self.scene
        eye, forward, right, up = self.camera.basis()
        scale = math.tan(math.radians(self.camera.fov / 2))
        pixel_angle = 2.0 * scale / scene.height
        scene.post.advance_frame(clock)
        frame_idx = float(scene.post.frame[None])
        jx = frame_idx * JITTER_GX % 1.0
        jy = frame_idx * JITTER_GY % 1.0
        scene.render(eye, forward, right, up, clock, scale, 1.0,
                     pixel_angle, scene.reflection_gain, jx, jy)
        return scene.post.hdr.to_numpy().copy()

    def test_static_classes_settle_below_raw_jitter_noise(self):
        scene = self.scene
        diffs = []
        prev = None
        for i in range(26):
            self._draw(1.0 + i / 60.0)
            accum = scene.accum_color.to_numpy()
            static = (scene.object_id.to_numpy() == 0) | (scene.object_id.to_numpy() == 2)
            if prev is not None:
                change = np.abs(accum - prev).max(axis=2)[static]
                diffs.append(float(np.quantile(change, 0.99)))
            prev = accum
        # Raw jittered frames differ by this much frame to frame on the
        # same static content; the accumulated sequence must be far calmer.
        raw_a = self._raw_jittered_frame(1.5)
        raw_b = self._raw_jittered_frame(1.5 + 1.0 / 60.0)
        raw = float(np.quantile(np.abs(raw_a - raw_b).max(axis=2)[static], 0.99))
        self.assertLess(diffs[-1], 0.35 * raw)
        # Steady-state shimmer of the running mean stays in the noise floor
        # of the HDR frame, well below clipping.
        self.assertLess(diffs[-1], 0.05)

    def test_water_recovers_within_a_few_frames(self):
        scene = self.scene
        for i in range(8):
            self._draw(2.0 + i / 60.0)
        water = scene.object_id.to_numpy() == 1

        def water_change(before, after):
            # Median over pixels: the moon glint throws hundreds of HDR
            # units at a few glitter-column pixels per frame, which would
            # drown any mean-based motion estimate.
            change = np.abs(after - before).max(axis=2)[water]
            return float(np.median(change))

        baseline_prev = scene.accum_color.to_numpy().copy()
        self._draw(2.0 + 8 / 60.0)
        baseline = scene.accum_color.to_numpy()
        wave_motion = water_change(baseline_prev, baseline)
        # Perturb one frame: multiply the wave amplitude eightfold, which
        # moves every water normal, then restore it. The waves are analytic
        # in the clock, so the frames before and after match exactly.
        controls = tuple(scene.water_controls[None])
        scene.water_controls[None] = [8.0, controls[1], controls[2], controls[3]]
        self._draw(2.0 + 9 / 60.0)
        scene.water_controls[None] = controls
        shocked = scene.accum_color.to_numpy().copy()
        shock = water_change(baseline, shocked)
        self.assertGreater(shock, 4.0 * wave_motion + 1e-4)
        prev = shocked
        late = None
        for i in range(5):
            self._draw(2.0 + (10 + i) / 60.0)
            current = scene.accum_color.to_numpy()
            late = water_change(prev, current)
            prev = current
        # A 2-frame window halves the residue per frame; after five frames
        # the transient must be back near the ambient wave motion.
        self.assertLess(late, max(3.0 * wave_motion, 0.2 * shock))

    def test_repeated_clock_resets_accumulation_bitwise(self):
        self._draw(3.0)
        first = self.scene.image.to_numpy()
        # The clock does not advance: accumulation restarts, so the frame
        # must match the first draw exactly.
        self._draw(3.0)
        np.testing.assert_array_equal(first, self.scene.image.to_numpy())

    def test_toggle_off_matches_centered_first_frame(self):
        scene = self.scene
        scene.temporal_aa = False
        self._draw(4.0)
        without_taa = scene.image.to_numpy()
        scene._accum_valid = False
        scene.temporal_aa = True
        self._draw(4.0)
        with_taa = scene.image.to_numpy()
        # Both draws sample at the pixel center: the first accumulated
        # frame is a copy of the plain render, grain included.
        np.testing.assert_array_equal(without_taa, with_taa)


if __name__ == "__main__":
    unittest.main()
