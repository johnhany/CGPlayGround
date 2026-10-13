"""Shore grass: instanced blade cards plus analytic near-field blades.

Level one scatters clusters of three to five crossed blade cards along the
beach band above the waterline; the blade silhouette comes from a
procedurally generated alpha atlas. Level two replaces clusters within a
couple of metres of the (fixed) camera with curved blade strips intersected
analytically, so foreground leaves carry no alpha-test staircases. The two
levels overlap in a narrow distance band and cross-fade by card alpha.

Cards live in a generic CardField with chunk bounding volumes: rays test
the chunk bound first and only then the individual quads. Tree leaf cards
reuse the same field class from trees.py.
"""

import numpy as np
import taichi as ti

from night_lake.terrain import (terrain_gradient_np, terrain_height_np,
                                vegetation_patch_noise)

__all__ = ["GRASS_SEED", "MAX_GRASS_CARDS", "MAX_BLADES", "NEAR_INNER",
           "NEAR_OUTER", "CardField", "GrassField", "generate_blade_atlas",
           "blade_hit_test", "card_hit_test", "sphere_hit", "triangle_hit"]


@ti.func
def card_hit_test(cards: ti.template(), i, origin, direction, density, best_t):
    """Single-card intersection against the current best distance.

    Returns (t, a, b); t is 1e6 on a miss. Shared by the bounding-group
    scan and the screen-tile candidate walk so both stay identical.
    """
    t, a, b = 1e6, 0.0, 0.0
    # Per-card hash spreads density thinning evenly; a full density skips
    # the hash entirely. A multiply-based hash keeps the per-card cost
    # negligible on the GPU.
    gate = 0.0
    if density < 0.999:
        hashed = float(i) * 0.61803398875
        gate = hashed - ti.floor(hashed)
    if gate <= density:
        normal = cards.card_normal[i]
        denom = normal.dot(direction)
        if ti.abs(denom) > 1e-8:
            t = normal.dot(cards.card_center[i] + cards.card_offset[i]
                           - origin) / denom
            if 0.001 < t < best_t:
                p = origin + direction * t - cards.card_center[i] \
                    - cards.card_offset[i]
                u, v = cards.card_u[i], cards.card_v[i]
                a = p.dot(u) * cards.card_inv[i].x
                b = p.dot(v) * cards.card_inv[i].y
                if ti.abs(a) <= 1.0 and ti.abs(b) <= 1.0:
                    meta = cards.card_meta[i]
                    alpha = cards.sample_alpha(ti.cast(meta.y, ti.i32), a, b)
                    if t >= 8.0:
                        alpha = cards.sample_alpha_fast(ti.cast(meta.y, ti.i32), a, b)
                    if alpha * meta.w < 0.5:
                        t = 1e6
                else:
                    t = 1e6
            else:
                t = 1e6
    return t, a, b


@ti.func
def blade_hit_test(grass: ti.template(), i, origin, direction, best_t):
    """Single analytic blade; returns (t, v along blade)."""
    offset = grass.blade_offset[i]
    v0 = grass.blade_vertex[i, 0] + offset
    v1 = grass.blade_vertex[i, 1] + offset
    v2 = grass.blade_vertex[i, 2] + offset
    v3 = grass.blade_vertex[i, 3] + offset
    t, w0, w1, w2 = triangle_hit(origin, direction, v0, v1, v2)
    v_param = w0 * grass.blade_vparam[i, 0] + w1 * grass.blade_vparam[i, 1] \
        + w2 * grass.blade_vparam[i, 2]
    if t > 1e5:
        t, w0, w1, w2 = triangle_hit(origin, direction, v0, v2, v3)
        v_param = w0 * grass.blade_vparam[i, 0] + w1 * grass.blade_vparam[i, 2] \
            + w2 * grass.blade_vparam[i, 3]
    if t >= best_t:
        t = 1e6
    return t, v_param

GRASS_SEED = 17
# Card capacity: clusters times blades per cluster, plus tree leaf headroom.
MAX_GRASS_CARDS = 8192
MAX_BLADES = 1536
# Camera distance band where cards cross-fade into analytic blades.
NEAR_INNER = 2.0
NEAR_OUTER = 3.4
# Clusters per bounding chunk: about one 4 m grid cell's worth, which keeps
# the group bounds tight enough for the sphere test to prune well.
GROUP_CLUSTERS = 12
BLADE_TEXTURES = 3
ATLAS_WIDTH = 64
ATLAS_HEIGHT = 128


@ti.func
def sphere_hit(origin, direction, center, radius):
    """Entry distance of a ray against a sphere, or 1e6 on a miss."""
    oc = origin - center
    b = oc.dot(direction)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - c
    t = 1e6
    if discriminant > 0.0:
        root = ti.sqrt(discriminant)
        t = -b - root
        if t < 0.001:
            t = -b + root
        if t < 0.001:
            t = 1e6
    return t


@ti.func
def triangle_hit(origin, direction, v0, v1, v2):
    """Moller-Trumbore intersection; returns (t, barycentric weights)."""
    e1, e2 = v1 - v0, v2 - v0
    p = direction.cross(e2)
    det = e1.dot(p)
    t, w0, w1, w2 = 1e6, 0.0, 0.0, 0.0
    if ti.abs(det) > 1e-9:
        inverse = 1.0 / det
        tv = origin - v0
        u = tv.dot(p) * inverse
        if -0.002 <= u <= 1.002:
            q = tv.cross(e1)
            v = direction.dot(q) * inverse
            if v >= -0.002 and u + v <= 1.002:
                candidate = e2.dot(q) * inverse
                if candidate > 0.001:
                    t, w0, w1, w2 = candidate, 1.0 - u - v, u, v
    return t, w0, w1, w2


def generate_blade_atlas():
    """Procedural blade silhouettes: tapered width, curved midrib line.

    One alpha channel per texture variant, shape (BLADE_TEXTURES, H, W).
    The taper tightens toward the tip, the centreline drifts (the visible
    bend), and a fine serration breaks the outline.
    """
    rng = np.random.default_rng(GRASS_SEED + 101)
    atlas = np.zeros((BLADE_TEXTURES, ATLAS_HEIGHT, ATLAS_WIDTH), dtype=np.float32)
    for variant in range(BLADE_TEXTURES):
        bend = rng.uniform(-0.16, 0.16)
        width = rng.uniform(0.10, 0.16)
        serration = rng.uniform(18.0, 30.0)
        column = np.arange(ATLAS_WIDTH, dtype=np.float32) + 0.5
        for row in range(ATLAS_HEIGHT):
            t = (row + 0.5) / ATLAS_HEIGHT
            centre = ATLAS_WIDTH * (0.5 + bend * t * t)
            half = (ATLAS_WIDTH * 0.5 * width * (1.0 - t) ** 0.72
                    * (1.0 + 0.10 * np.sin(serration * t + variant)))
            distance = np.abs(column - centre)
            silhouette = np.clip((half - distance) / 1.5, 0.0, 1.0)
            atlas[variant, row] = silhouette
        # Slight tip thinning so the very top fades instead of cutting off.
        tip = np.clip((ATLAS_HEIGHT - 2.0 - np.arange(ATLAS_HEIGHT)) / 3.0, 0.0, 1.0)
        atlas[variant] *= tip[:, None]
    return atlas


@ti.data_oriented
class CardField:
    """Quads with chunk bounds; the shared card container for vegetation."""

    def __init__(self, capacity, atlas, max_groups, group_size=GROUP_CLUSTERS):
        self.capacity = capacity
        self.group_size = group_size
        self.card_center = ti.Vector.field(3, ti.f32, shape=capacity)
        self.card_u = ti.Vector.field(3, ti.f32, shape=capacity)
        self.card_v = ti.Vector.field(3, ti.f32, shape=capacity)
        # kind (0 grass, 1 leaf), texture variant, phase, alpha scale.
        self.card_meta = ti.Vector.field(4, ti.f32, shape=capacity)
        # Precomputed quad normal and inverse squared axis lengths, plus the
        # per-frame wind offset, keep the inner intersection loop cheap.
        self.card_normal = ti.Vector.field(3, ti.f32, shape=capacity)
        self.card_inv = ti.Vector.field(2, ti.f32, shape=capacity)
        self.card_offset = ti.Vector.field(3, ti.f32, shape=capacity)
        # Per-card contact occlusion baked at generation time: concavity
        # of the ground right under the card, darkening grass roots where
        # the terrain dips.
        self.card_ao = ti.field(ti.f32, shape=capacity)
        # Per-card withered fraction (0 lush, 1 straw), baked at
        # generation time from the stand-level wither noise.
        self.card_dry = ti.field(ti.f32, shape=capacity)
        self.group_center = ti.Vector.field(3, ti.f32, shape=max_groups)
        self.group_extent = ti.Vector.field(3, ti.f32, shape=max_groups)
        self.group_range = ti.Vector.field(2, ti.i32, shape=max_groups)
        self.atlas = ti.field(ti.f32, shape=atlas.shape)
        self.atlas.from_numpy(atlas)
        self.atlas_height, self.atlas_width = atlas.shape[1], atlas.shape[2]
        self.count = 0
        self.group_count = 0
        self._pending = []
        self._group_start = 0

    def add_card(self, center, u, v, kind, texture, phase, alpha_scale,
                 ao=1.0, dry=0.0):
        if self.count >= self.capacity:
            raise ValueError("card capacity exceeded")
        i = self.count
        self.card_center[i], self.card_u[i] = center, u
        self.card_v[i], self.card_meta[i] = v, [kind, texture, phase, alpha_scale]
        self.card_ao[i] = ao
        self.card_dry[i] = dry
        normal = np.cross(u, v)
        self.card_normal[i] = normal
        self.card_inv[i] = [1.0 / max(float(u @ u), 1e-8),
                            1.0 / max(float(v @ v), 1e-8)]
        self.card_offset[i] = [0.0, 0.0, 0.0]
        self.count += 1
        lo = np.asarray(center) - np.abs(np.asarray(u)) - np.abs(np.asarray(v))
        hi = np.asarray(center) + np.abs(np.asarray(u)) + np.abs(np.asarray(v))
        self._pending.append((lo, hi))

    def finish_group(self):
        if not self._pending:
            return
        if self.group_count >= self.group_center.shape[0]:
            raise ValueError("group capacity exceeded")
        lo = np.min([b[0] for b in self._pending], axis=0)
        hi = np.max([b[1] for b in self._pending], axis=0)
        i = self.group_count
        self.group_center[i] = (lo + hi) / 2
        self.group_extent[i] = (hi - lo) / 2 + 0.02
        self.group_range[i] = (self._group_start, self.count)
        self.group_count += 1
        self._pending.clear()
        self._group_start = self.count

    @ti.func
    def sample_alpha(self, texture, a, b):
        # Card coordinates run [-1, 1]; v = 0 is the blade base.
        u = (a + 1.0) * 0.5
        v = 1.0 - (b + 1.0) * 0.5
        u = ti.max(0.0, ti.min(1.0, u)) * (self.atlas_width - 1.0)
        v = ti.max(0.0, ti.min(1.0, v)) * (self.atlas_height - 1.0)
        ix, iy = ti.cast(ti.floor(u), ti.i32), ti.cast(ti.floor(v), ti.i32)
        fx, fy = u - ix, v - iy
        result = 0.0
        for i, j in ti.static(ti.ndrange(2, 2)):
            weight = (fx if i else 1.0 - fx) * (fy if j else 1.0 - fy)
            result += weight * self.atlas[texture, ti.min(iy + j, self.atlas_height - 1),
                                          ti.min(ix + i, self.atlas_width - 1)]
        return result

    @ti.func
    def sample_alpha_fast(self, texture, a, b):
        # Single tap for distant cards, where the quad is only a few pixels.
        u = (a + 1.0) * 0.5
        v = 1.0 - (b + 1.0) * 0.5
        ix = ti.cast(ti.max(0.0, ti.min(1.0, u)) * (self.atlas_width - 1.0), ti.i32)
        iy = ti.cast(ti.max(0.0, ti.min(1.0, v)) * (self.atlas_height - 1.0), ti.i32)
        return self.atlas[texture, iy, ix]

    @ti.func
    def cards_hit(self, origin, direction, density, t_max):
        """Closest alpha-passing card; returns (t, index, a, b)."""
        best_t, best_i, best_a, best_b = 1e6, -1, 0.0, 0.0
        for g in range(self.group_count):
            bound = sphere_hit(origin, direction, self.group_center[g],
                               self.group_extent[g].norm())
            if bound < ti.min(t_max, best_t):
                for i in range(self.group_range[g][0], self.group_range[g][1]):
                    t, a, b = card_hit_test(self, i, origin, direction,
                                            density, best_t)
                    if t < best_t:
                        best_t, best_i, best_a, best_b = t, i, a, b
        return best_t, best_i, best_a, best_b


@ti.data_oriented
class GrassField:
    """Scattered shore grass: cards at range, analytic blades up close."""

    def __init__(self, eye, density=1.0):
        self.cards = CardField(MAX_GRASS_CARDS, generate_blade_atlas(), max_groups=128)
        self.blade_vertex = ti.Vector.field(3, ti.f32, shape=(MAX_BLADES, 4))
        self.blade_vparam = ti.field(ti.f32, shape=(MAX_BLADES, 4))
        # [phase, brightness tint, withered fraction]
        self.blade_meta = ti.Vector.field(3, ti.f32, shape=MAX_BLADES)
        self.blade_offset = ti.Vector.field(3, ti.f32, shape=MAX_BLADES)
        self.blade_bound = ti.Vector.field(4, ti.f32, shape=1)
        self.blade_count = 0
        self.cluster_centers = []
        self._group_since = 0
        self._scatter(np.asarray(eye, dtype=np.float64), density)
        self._bake_root_ao()
        if self.blade_count > 0:
            verts = self.blade_vertex.to_numpy()[: self.blade_count].reshape(-1, 3)
            centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
            radius = float(np.linalg.norm(verts.max(axis=0) - centre)) + 0.05
        else:
            centre, radius = np.zeros(3), 0.01
        self.blade_bound[0] = [*centre, radius]

    def _bake_root_ao(self):
        """Contact occlusion per card from the concavity of the ground
        right under the card base, evaluated on two fixed rings so the
        result is deterministic and seed-independent.
        """
        cards = self.cards
        if cards.count == 0:
            return
        centre = cards.card_center.to_numpy()[:cards.count]
        v = cards.card_v.to_numpy()[:cards.count]
        base = centre - v
        h0 = terrain_height_np(base[:, 0].astype(np.float32),
                               base[:, 2].astype(np.float32))
        rise = np.zeros(cards.count, dtype=np.float64)
        for radius, weight in ((0.35, 1.0), (0.8, 0.5)):
            for k in range(6):
                angle = 2.0 * np.pi * k / 6.0
                dx, dz = np.cos(angle) * radius, np.sin(angle) * radius
                h = terrain_height_np((base[:, 0] + dx).astype(np.float32),
                                      (base[:, 2] + dz).astype(np.float32))
                rise += weight * np.clip(h.astype(np.float64) - h0, 0.0, 2.0)
        ao = 1.0 / (1.0 + 1.8 * rise / 9.0)
        for i in range(cards.count):
            cards.card_ao[i] = float(np.clip(ao[i], 0.45, 1.0))

    def _scatter(self, eye, density):
        rng = np.random.default_rng(GRASS_SEED)
        target = int(620 * density)
        pending = []
        attempts = 0
        while len(pending) < target and attempts < target * 40:
            attempts += 1
            x = rng.uniform(-20.0, 20.0)
            # The production eye sits 5.5 m up, so the beach reads as a
            # narrow strip along the waterline at the frame bottom. Keep the
            # grass band hugging the shore; the bias toward the waterline
            # matches where the eye actually lands on the terrain.
            z = 8.0 + 7.0 * rng.random() ** 1.3
            # Frustum-band cull: the fixed camera looks down +z, so keep only
            # clusters inside the visible wedge (generous margin for sway and
            # card half widths). Anything else is never on screen.
            if abs(x) > z * 0.9 + 2.5:
                continue
            height = float(terrain_height_np(np.float32(x), np.float32(z)))
            if height < 0.08 or height > 0.5:
                continue
            gradient = terrain_gradient_np(np.asarray(x), np.asarray(z))
            if float(np.linalg.norm(gradient)) > 0.45:
                continue
            # Dense right at the waterline, thinning up the berm.
            falloff = 1.0 - 0.75 * min(1.0, max(0.0, (height - 0.1) / 0.4))
            if rng.random() > falloff:
                continue
            # Meadow patchiness: a second low-frequency noise field breaks
            # the band into dense tufts and bare gaps, so the shore reads
            # as grown cover rather than uniform stubble.
            patch = float(vegetation_patch_noise(np.float32(x), np.float32(z)))
            if rng.random() > 0.45 + 0.70 * (patch + 0.5):
                continue
            pending.append((x, z, height))
        # Bucket by grid cell so each bounding group stays spatially tight;
        # scatter order alone would produce whole-band bounds and no pruning.
        cell = 4.0
        pending.sort(key=lambda c: (int(c[0] // cell), int(c[2] // cell)))
        for i, (x, z, height) in enumerate(pending):
            self._cluster(rng, eye, x, z, height)
            self._group_since += 1
            if self._group_since >= GROUP_CLUSTERS:
                self.cards.finish_group()
                self._group_since = 0
        if self._group_since > 0:
            self.cards.finish_group()

    def _cluster(self, rng, eye, x, z, ground):
        centre = np.array([x, ground, z])
        distance = float(np.linalg.norm(centre - eye))
        # Far clusters are only a few pixels tall on screen; give them fewer
        # blades so the far field, which most rays cross, stays cheap.
        if distance > 20.0:
            blade_count = int(rng.integers(2, 4))
        else:
            blade_count = int(rng.integers(4, 7))
        phase = rng.random()
        texture = int(rng.integers(0, BLADE_TEXTURES))
        blades_only = distance < NEAR_INNER
        cards_only = distance >= NEAR_OUTER
        alpha_scale = 1.0 if cards_only else (0.0 if blades_only
                                              else (distance - NEAR_INNER) / (NEAR_OUTER - NEAR_INNER))
        # Stand-level variation: two more noise fields decide whether this
        # whole cluster lies flat (wind lodging) and whether it grew in
        # straw, so both read as patches rather than per-blade dice rolls.
        lodge_noise = float(vegetation_patch_noise(np.float32(x + 37.0),
                                                   np.float32(z - 11.0)))
        wither_noise = float(vegetation_patch_noise(np.float32(x - 53.0),
                                                    np.float32(z + 29.0)))
        # Noise over the beach band spans roughly [-0.35, 0.15]; the
        # thresholds below sit at its median and upper tail, so about half
        # the clusters show some lodging and the top decile stands in
        # straw.
        lodging = min(1.0, max(0.0, -0.10 - lodge_noise) * 2.5)
        dry = min(0.85, max(0.0, wither_noise + 0.13) * 2.2)
        # Wind-flattened swathes bend toward one scene-wide direction.
        lodge_dir = np.array([0.82, 0.0, 0.57])
        lodge_dir /= np.linalg.norm(lodge_dir)
        for _ in range(blade_count):
            azimuth = rng.uniform(0.0, 2.0 * np.pi)
            tilt = rng.uniform(0.06, 0.45)
            direction = np.array([np.sin(azimuth) * np.sin(tilt),
                                  np.cos(tilt), np.cos(azimuth) * np.sin(tilt)])
            length = rng.uniform(0.24, 0.66) * (1.0 - 0.35 * lodging)
            width = rng.uniform(0.026, 0.060)
            if lodging > 0.0:
                # Slerp-ish flattening: rotate the blade direction toward
                # the lodge direction by up to ~63 degrees.
                angle = lodging * 1.1
                direction = (direction * np.cos(angle)
                             + lodge_dir * np.sin(angle))
                direction /= np.linalg.norm(direction)
            base = centre + np.array([rng.uniform(-0.09, 0.09), 0.0,
                                      rng.uniform(-0.09, 0.09)])
            side = np.array([-direction[2], 0.0, direction[0]])
            if np.linalg.norm(side) < 1e-4:
                side = np.array([1.0, 0.0, 0.0])
            side /= np.linalg.norm(side)
            if not blades_only:
                u = side * (width * 0.5)
                v = direction * (length * 0.5)
                self.cards.add_card(base + v, u, v, 0, texture, phase,
                                    alpha_scale, dry=dry)
            if not cards_only:
                self._add_blade(rng, base, direction, side, length, width,
                                phase, lodging, dry)
        self.cluster_centers.append(centre)

    def _add_blade(self, rng, base, direction, side, length, width, phase,
                   lodging, dry):
        if self.blade_count >= MAX_BLADES:
            return
        bend = rng.uniform(-0.25, 0.25) * (1.0 - 0.6 * lodging)
        bend_dir = rng.uniform(-0.4, 0.4)
        mid = base + direction * (length * 0.55) + side * (width * 0.18)
        tip = base + direction * length + side * (bend * length * 0.35)
        # Lodged blades lie nearly flat: relax the upright tip floor so the
        # flattening actually shows instead of snapping back vertical.
        tip[1] = max(tip[1], base[1] + length * 0.45 * (1.0 - 0.8 * lodging))
        i = self.blade_count
        self.blade_vertex[i, 0] = base - side * (width * 0.5)
        self.blade_vertex[i, 1] = base + side * (width * 0.5)
        self.blade_vertex[i, 2] = mid + side * (width * 0.28 + bend_dir * 0.01)
        self.blade_vertex[i, 3] = tip
        self.blade_vparam[i, 0], self.blade_vparam[i, 1] = 0.0, 0.0
        self.blade_vparam[i, 2], self.blade_vparam[i, 3] = 0.55, 1.0
        self.blade_meta[i] = [phase, rng.uniform(0.8, 1.2), dry]
        self.blade_count += 1

    @ti.func
    def blades_hit(self, origin, direction, t_max):
        """Closest analytic blade; returns (t, index, v along blade)."""
        best_t, best_i, best_v = 1e6, -1, 0.0
        bound = self.blade_bound[0]
        broad = sphere_hit(origin, direction, ti.Vector([bound.x, bound.y, bound.z]), bound.w)
        if broad < ti.min(t_max, 1e5):
            for i in range(self.blade_count):
                t, v_param = blade_hit_test(self, i, origin, direction, best_t)
                if t < best_t:
                    best_t, best_i, best_v = t, i, v_param
        return best_t, best_i, best_v
