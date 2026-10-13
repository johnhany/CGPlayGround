"""Screen-tile vegetation acceleration checks (no GUI).

The tile-candidate intersection must return exactly the same hit as the
bounding-group brute-force scan for every sampled primary ray, and the
world-space uniform grid must index every instance at its own cell.
"""

import math
import unittest

import numpy as np
import taichi as ti

from night_lake.accel import TILE_PIXELS
from night_lake.camera import FixedCamera
from night_lake.scene import NightLakeScene


@ti.kernel
def probe_vegetation(s: ti.template(), eye: ti.types.vector(3, ti.f32),
                     forward: ti.types.vector(3, ti.f32),
                     right: ti.types.vector(3, ti.f32),
                     up: ti.types.vector(3, ti.f32), scale: ti.f32,
                     clock: ti.f32, out: ti.types.ndarray()):
    for x, y in s.object_id:
        if x % 3 == 0 and y % 3 == 0:
            sx = (2.0 * (x + 0.5) / s.width - 1.0) * s.width / s.height * scale
            sy = (2.0 * (y + 0.5) / s.height - 1.0) * scale
            direction = (forward + right * sx + up * sy).normalized()
            tile = ti.Vector([x // TILE_PIXELS, y // TILE_PIXELS])
            tb, kb, ib, _, _ = s.vegetation_hit(eye, direction, clock, 1e6)
            tt, kt, it, _, _ = s.vegetation_hit_tiled(eye, direction, clock,
                                                      1e6, tile)
            out[x, y, 0] = tb
            out[x, y, 1] = tt
            out[x, y, 2] = float(kb)
            out[x, y, 3] = float(kt)
            out[x, y, 4] = float(ib)
            out[x, y, 5] = float(it)


class TileConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ti.init(arch=ti.cpu, offline_cache=False)
        cls.scene = NightLakeScene(160, 90)

    def _compare(self):
        scene = self.scene
        camera = FixedCamera()
        eye, forward, right, up = camera.basis()
        scale = math.tan(math.radians(camera.fov / 2.0))
        out = np.zeros((scene.width, scene.height, 6), dtype=np.float32)
        probe_vegetation(scene, eye, forward, right, up, scale, 2.0, out)
        probed = out[::3, ::3]
        flat = probed.reshape(-1, 6)
        tb, tt = flat[:, 0], flat[:, 1]
        hit_both = (tb < 1e5) & (tt < 1e5)
        miss_both = (tb >= 1e5) & (tt >= 1e5)
        same_hit = ((np.abs(flat[:, 0] - flat[:, 1]) < 2e-3)
                    & (flat[:, 2] == flat[:, 3]) & (flat[:, 4] == flat[:, 5]))
        bad = ~(hit_both & same_hit | miss_both)
        return int(bad.sum()), int(len(flat))

    def test_tiled_matches_bruteforce_at_full_density(self):
        # Worst-case sway first: maximum wind pushes every offset to the
        # projection margin the tile lists were built with.
        self.scene.update_vegetation_sway(1.5, 2.0, self.scene.grass.cards.count,
                                          self.scene.grass.blade_count,
                                          self.scene.trees.leaves.count)
        bad, total = self._compare()
        self.assertEqual(bad, 0, f"{bad} of {total} rays disagree")

    def test_tiled_matches_bruteforce_at_half_density(self):
        controls = list(self.scene.vegetation_controls[None])
        controls[0], controls[1] = 0.5, 0.5
        self.scene.vegetation_controls[None] = controls
        try:
            bad, total = self._compare()
            self.assertEqual(bad, 0, f"{bad} of {total} rays disagree")
        finally:
            controls[0], controls[1] = 1.0, 1.0
            self.scene.vegetation_controls[None] = controls


class WorldGridTests(unittest.TestCase):
    def test_every_instance_is_indexed_at_its_cell(self):
        scene = NightLakeScene(160, 90)
        accel = scene.accel
        centres = []
        cards = scene.grass.cards
        centres.extend(cards.card_center.to_numpy()[:cards.count])
        leaves = scene.trees.leaves
        centres.extend(leaves.card_center.to_numpy()[:leaves.count])
        starts = accel.world_cell_start.to_numpy()
        items = accel.world_cell_item.to_numpy()
        missing = 0
        for i, centre in enumerate(centres):
            cell = accel.world_cell_of(centre)
            if i not in items[starts[cell]:starts[cell + 1]]:
                missing += 1
        self.assertEqual(missing, 0)


if __name__ == "__main__":
    unittest.main()
