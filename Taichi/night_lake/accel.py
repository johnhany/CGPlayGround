"""Vegetation acceleration structures for the fixed production view.

Two structures are built once at scene start-up, when the grass field and
the tree field are complete:

Screen-tile candidate lists project every card and blade bounding sphere
onto the production camera and record, per screen tile, which instances
can overlap it. The primary render loop then walks only the candidates
of the pixel's own tile instead of every bounding group. Sway amplitude
is folded into the projection radius with the wind-slider ceiling, so
the lists stay valid at any wind setting without per-frame rebuilds.

A world-space uniform grid over the same instances is built alongside;
this step only maintains the structure, which later stages (contact
shadows among the vegetation) will query.

Instances arrive as parallel arrays: items holds fully encoded
item = source * SRC_STRIDE + instance_index values for the screen
lists, while the world grid addresses each instance by its row index.
"""

import numpy as np
import taichi as ti

__all__ = ["SRC_BLADE", "SRC_GRASS", "SRC_LEAF", "SRC_STRIDE",
           "TILE_PIXELS", "WIND_MARGIN_MAX", "VegetationAccel"]

# Screen tile edge in pixels; 16 keeps tile lists short without rebuilds.
TILE_PIXELS = 16
# Wind slider ceiling from the vegetation panel; sway margins use the
# worst case so the projection stays conservative at any slider value.
WIND_MARGIN_MAX = 1.5
SRC_GRASS, SRC_BLADE, SRC_LEAF = 0, 1, 2
# Instance counts stay below this per source (grass capacity 8192).
SRC_STRIDE = 16384


class VegetationAccel:
    """Screen-tile lists plus a world uniform grid over one instance set.

    items is the encoded screen-list value per instance, centres the
    numpy (n, 3) instance centres, and radii the conservative bounding
    sphere radii including the sway margin. The camera basis is the
    FixedCamera one; anything straddling the near plane is added to
    every tile so the lists stay conservative.
    """

    def __init__(self, width, height, eye, forward, right, up, fov_degrees,
                 items, centres, radii, world_cell=2.0):
        self.width, self.height = width, height
        self.tiles_x = max(1, (width + TILE_PIXELS - 1) // TILE_PIXELS)
        self.tiles_y = max(1, (height + TILE_PIXELS - 1) // TILE_PIXELS)
        scale = float(np.tan(np.radians(fov_degrees / 2.0)))
        self._pixel_angle = 2.0 * scale / height
        eye = np.asarray(eye, dtype=np.float64)
        self._basis = tuple(np.asarray(v, dtype=np.float64)
                            for v in (eye, forward, right, up))
        self._scale = scale
        items = np.asarray(items, dtype=np.int32).reshape(-1)
        centres = np.asarray(centres, dtype=np.float64).reshape(-1, 3)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        tile_lists = [[] for _ in range(self.tiles_x * self.tiles_y)]
        margin = 0.5 * TILE_PIXELS * np.sqrt(2.0) + 2.0
        for row in range(centres.shape[0]):
            for tx, ty in self._overlap_tiles(centres[row], radii[row], margin):
                tile_lists[ty * self.tiles_x + tx].append(items[row])
        starts = np.zeros(self.tiles_x * self.tiles_y + 1, dtype=np.int32)
        np.cumsum([len(t) for t in tile_lists], out=starts[1:])
        flat = np.concatenate([np.asarray(t, dtype=np.int32) for t in tile_lists]) \
            if starts[-1] else np.zeros(0, dtype=np.int32)
        self.tile_start = ti.field(ti.i32, shape=starts.shape[0])
        self.tile_start.from_numpy(starts)
        self.tile_item = ti.field(ti.i32, shape=max(1, flat.shape[0]))
        if flat.shape[0]:
            self.tile_item.from_numpy(flat)
        # World-space uniform grid over the same instances.
        self.world_cell = float(world_cell)
        if centres.shape[0]:
            reach = float(radii.max())
            lo = centres.min(axis=0) - reach
            hi = centres.max(axis=0) + reach
        else:
            lo = hi = np.zeros(3, dtype=np.float64)
        self.world_origin = lo
        span = np.maximum(hi - lo, self.world_cell)
        self.world_dims = np.ceil(span / self.world_cell).astype(np.int32)
        cell_lists = [[] for _ in range(int(np.prod(self.world_dims)))]
        for row in range(centres.shape[0]):
            cell_lists[self.world_cell_of(centres[row])].append(row)
        w_starts = np.zeros(len(cell_lists) + 1, dtype=np.int32)
        np.cumsum([len(c) for c in cell_lists], out=w_starts[1:])
        w_items = np.concatenate([np.asarray(c, dtype=np.int32) for c in cell_lists]) \
            if w_starts[-1] else np.zeros(0, dtype=np.int32)
        self.world_cell_start = ti.field(ti.i32, shape=w_starts.shape[0])
        self.world_cell_start.from_numpy(w_starts)
        self.world_cell_item = ti.field(ti.i32, shape=max(1, w_items.shape[0]))
        if w_items.shape[0]:
            self.world_cell_item.from_numpy(w_items)

    def world_cell_of(self, point):
        """Grid cell index of a world point (clamped to the grid)."""
        index = np.floor((np.asarray(point, dtype=np.float64)
                          - self.world_origin) / self.world_cell).astype(np.int32)
        return int(np.ravel_multi_index(tuple(np.clip(index, 0, self.world_dims - 1)),
                                        tuple(self.world_dims)))

    def _overlap_tiles(self, centre, radius, margin):
        eye, forward, right, up = self._basis
        d = centre - eye
        z = float(d @ forward)
        tiles = []
        if z < radius + 0.2:
            # Straddling the eye plane: only a fully conservative mapping
            # is safe, so this instance votes for every tile.
            return [(tx, ty) for ty in range(self.tiles_y)
                    for tx in range(self.tiles_x)]
        x, y = float(d @ right), float(d @ up)
        half_w = self._scale * self.width / self.height
        # Angular size of the bounding sphere; a 20 percent pad absorbs the
        # tangent stretching toward the frame corners.
        angular = 1.2 * np.arcsin(min(0.9, radius / max(z, 0.5)))
        r_px = angular / self._pixel_angle + margin
        px = (0.5 + 0.5 * (x / z) / half_w) * self.width
        py = (0.5 + 0.5 * (y / z) / self._scale) * self.height
        tx0 = max(0, int((px - r_px) // TILE_PIXELS))
        tx1 = min(self.tiles_x - 1, int((px + r_px) // TILE_PIXELS))
        ty0 = max(0, int((py - r_px) // TILE_PIXELS))
        ty1 = min(self.tiles_y - 1, int((py + r_px) // TILE_PIXELS))
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tiles.append((tx, ty))
        return tiles
