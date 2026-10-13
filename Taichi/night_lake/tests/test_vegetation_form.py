"""Vegetation form checks: species parameter sets produce distinct,
reproducible crown shapes; skeleton invariants hold (branch radius decays
monotonically, trunk flare sits at the base); grass clusters stay above
the waterline, carry a withered mix, and cover more of the frame."""

import unittest

import numpy as np
import taichi as ti

from night_lake.camera import FixedCamera
from night_lake.scene import NightLakeScene
from night_lake.terrain import terrain_height_np
from night_lake.trees import TreeField


class SpeciesGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.field = TreeField()

    def test_build_is_reproducible(self):
        other = TreeField()
        np.testing.assert_array_equal(
            self.field.seg_a.to_numpy()[:self.field.count],
            other.seg_a.to_numpy()[:other.count])
        np.testing.assert_array_equal(
            self.field.seg_radius.to_numpy()[:self.field.count],
            other.seg_radius.to_numpy()[:other.count])
        self.assertEqual(self.field.leaves.count, other.leaves.count)

    def test_species_crown_shapes_differ(self):
        # Tree 0 grows from the umbrella parameter set, tree 1 from the
        # round one: the umbrella crown must spread wider than it rises,
        # the round crown taller than it spreads.
        field = self.field
        a = field.seg_a.to_numpy()[:field.count]
        b = field.seg_b.to_numpy()[:field.count]
        ranges = field.tree_range.to_numpy()[:field.tree_count]
        ratios = []
        for i in range(2):
            s, e = ranges[i]
            pts = np.concatenate([a[s:e], b[s:e]])
            horiz = float(np.hypot(pts[:, 0].ptp(), pts[:, 2].ptp()))
            vert = float(pts[:, 1].ptp())
            ratios.append(horiz / vert)
        self.assertGreater(ratios[0], 1.3 * ratios[1])

    def test_branch_radius_decays_monotonically(self):
        field = self.field
        radius = field.seg_radius.to_numpy()[:field.count]
        parent = field.seg_parent.to_numpy()[:field.count]
        for child in range(field.count):
            p = int(parent[child])
            if p >= 0:
                self.assertLessEqual(radius[child, 0], radius[p, 0] + 1e-6,
                                     f"segment {child} thicker than its parent")

    def test_trunk_flare_sits_at_the_base(self):
        field = self.field
        a = field.seg_a.to_numpy()[:field.count]
        b = field.seg_b.to_numpy()[:field.count]
        radius = field.seg_radius.to_numpy()[:field.count]
        ranges = field.tree_range.to_numpy()[:field.tree_count]
        for i in range(2):
            s, e = ranges[i]
            k = s + int(np.argmax(radius[s:e, 0]))
            ymin = float(min(a[s:e, 1].min(), b[s:e, 1].min()))
            ymax = float(max(a[s:e, 1].max(), b[s:e, 1].max()))
            low = min(float(a[k, 1]), float(b[k, 1]))
            self.assertGreater(radius[k, 0], 1.5 * float(np.median(radius[s:e, 0])))
            self.assertLess((low - ymin) / (ymax - ymin), 0.25)


class GrassFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)
        cls.camera = FixedCamera()

    def test_clusters_sit_above_the_waterline(self):
        grass = self.scene.grass
        centres = np.asarray(grass.cluster_centres if hasattr(grass, "cluster_centres")
                             else grass.cluster_centers)
        heights = terrain_height_np(centres[:, 0].astype(np.float32),
                                    centres[:, 2].astype(np.float32))
        self.assertGreater(heights.min(), 0.0)

    def test_withered_and_lush_blades_mix(self):
        dry = self.scene.grass.cards.card_dry.to_numpy()
        dry = dry[:self.scene.grass.cards.count]
        self.assertGreater((dry > 0.25).mean(), 0.05)
        self.assertGreater((dry < 0.05).mean(), 0.5)

    def test_grass_covers_more_of_the_frame(self):
        # Reworked scatter: clustered meadows at 620 clusters with wider
        # blades cover the beach band far better than the old 26 pixels.
        scene = self.scene
        scene.warm_rain(0.5)
        for i in range(3):
            scene.draw(self.camera, 2.0 + i / 60.0)
        grass_pixels = int((scene.object_id.to_numpy() == 3).sum())
        self.assertGreater(grass_pixels, 40)


if __name__ == "__main__":
    unittest.main()
