"""Stage 4 checks for vegetation: deterministic generation from the fixed
seed, structural invariants of the grass and tree fields, and image-level
presence of both framing trees and foreground grass (no GUI required)."""

import unittest

import numpy as np
import taichi as ti

import night_lake.grass as grass_module
from night_lake.camera import FixedCamera
from night_lake.grass import (GrassField, MAX_BLADES, generate_blade_atlas)
from night_lake.scene import PRODUCTION_EYE, NightLakeScene
from night_lake.terrain import terrain_height_np
from night_lake.trees import TreeField, generate_leaf_atlas


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)

    def test_grass_is_deterministic_across_instances(self):
        first = GrassField(PRODUCTION_EYE)
        second = GrassField(PRODUCTION_EYE)
        self.assertEqual(first.cards.count, second.cards.count)
        self.assertEqual(first.blade_count, second.blade_count)
        np.testing.assert_allclose(first.cards.card_center.to_numpy(),
                                   second.cards.card_center.to_numpy())
        np.testing.assert_allclose(first.blade_vertex.to_numpy(),
                                   second.blade_vertex.to_numpy())

    def test_grass_changes_with_seed(self):
        baseline = GrassField(PRODUCTION_EYE)
        grass_module.GRASS_SEED = 99
        try:
            other = GrassField(PRODUCTION_EYE)
        finally:
            grass_module.GRASS_SEED = 17
        self.assertFalse(np.allclose(baseline.cards.card_center.to_numpy()[:64],
                                     other.cards.card_center.to_numpy()[:64]))

    def test_grass_sits_above_the_waterline(self):
        field = GrassField(PRODUCTION_EYE)
        self.assertGreater(len(field.cluster_centers), 0)
        for center in field.cluster_centers:
            height = terrain_height_np(np.float32(center[0]), np.float32(center[2]))
            self.assertGreater(height, 0.0)

    def test_tree_radius_shrinks_along_parent_chains(self):
        field = TreeField()
        radii = field.seg_radius.to_numpy()
        parents = field.seg_parent.to_numpy()
        for i in range(field.count):
            parent = parents[i]
            if parent >= 0:
                self.assertLess(radii[i, 0], radii[parent, 0])

    def test_leaf_cards_are_orthogonal(self):
        field = TreeField()
        leaves = field.leaves
        us = leaves.card_u.to_numpy()[:leaves.count]
        vs = leaves.card_v.to_numpy()[:leaves.count]
        norms = np.linalg.norm(us, axis=1) * np.linalg.norm(vs, axis=1)
        cosine = np.abs(np.sum(us * vs, axis=1)) / norms
        self.assertGreater(len(cosine), 0)
        np.testing.assert_array_less(cosine, 1e-5)

    def test_atlas_shapes_and_ranges(self):
        blades = generate_blade_atlas()
        self.assertEqual(blades.shape[0], 3)
        self.assertTrue(np.all((blades >= 0.0) & (blades <= 1.0)))
        leaves = generate_leaf_atlas()
        self.assertEqual(leaves.shape[0], 3)
        self.assertEqual(leaves.shape[1], leaves.shape[2])
        self.assertTrue(np.all((leaves >= 0.0) & (leaves <= 1.0)))


class VegetationFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)
        cls.scene.draw(FixedCamera(), clock=0.0)
        cls.object_id = cls.scene.object_id.to_numpy()
        cls.frame = cls.scene.image.to_numpy()

    def test_both_sides_have_dark_tree_pixels(self):
        width, height = self.object_id.shape
        top = self.object_id[:width // 2, height // 2:]
        left_ids, right_ids = top[:width // 4, :], top[width // 4:, :]
        self.assertGreater((left_ids == 4).sum(), 0)
        self.assertGreater((right_ids == 4).sum(), 0)
        sky = self.frame[:width // 2, :][self.object_id[:width // 2, :] == 0]
        tree = self.frame[:width // 2, :][self.object_id[:width // 2, :] == 4]
        self.assertLess(tree.mean(), sky.mean() * 0.7)

    def test_foreground_has_grass(self):
        width, height = self.object_id.shape
        bottom = self.object_id[:, :height // 4]
        self.assertGreater((bottom == 3).sum(), 0)

    def test_no_grass_in_the_upper_sky(self):
        width, height = self.object_id.shape
        upper = self.object_id[:, 3 * height // 4:]
        self.assertEqual((upper == 3).sum(), 0)

    def test_near_grass_edges_stay_smooth(self):
        # Alpha-tested card edges alias into high-frequency on/off runs
        # along a row; the near field is analytic blades, so the run count
        # of grass toggles per row stays bounded.
        grass = (self.object_id == 3).T
        band = grass[:grass.shape[0] // 2, :]
        toggles = (band[:, 1:] != band[:, :-1]).sum(axis=1).max()
        self.assertLessEqual(toggles, 40)


if __name__ == "__main__":
    unittest.main()
