"""Shore trees: recursive branch skeletons, cone segments, leaf card crowns.

The skeleton grows on the CPU from a fixed seed: each branch spawns two or
three children that are shorter, thinner, slightly upward-biased, and a
little droopier with depth, so radius decreases monotonically along every
root-to-tip path. Segments are stored as truncated cones (two endpoints,
two radii) with a per-segment bounding sphere for ray pruning.

Two main trees frame the left and right of the fixed camera; the right one
sits well off the moon azimuth so the disk stays clear. Simplified trees
on the far shore fill the horizon layer. Crowns are card clusters using a
procedurally generated foliage alpha atlas, sharing the CardField
container with the grass.
"""

import numpy as np
import taichi as ti

from night_lake.grass import (CardField, sphere_hit)
from night_lake.terrain import terrain_height_np

__all__ = ["TREE_SEED", "MAX_SEGMENTS", "MAX_TREE_CARDS", "TreeField",
           "generate_leaf_atlas"]

TREE_SEED = 17
MAX_SEGMENTS = 2048
MAX_TREE_CARDS = 2048
# Second-level bounds: segments of one tree bucketed into small spatial
# chunks so rays only test the chunks they actually enter.
MAX_SEG_GROUPS = 256
SEG_GROUP_SIZE = 6
MAX_LEVEL = 6
LEAF_TEXTURES = 3
LEAT_ATLAS_SIZE = 96

# Species parameter sets: every morphological knob of the skeleton
# generator lives here, so a tree is fully described by which entry it
# grows from. The round-crown broadleaf keeps an upward-biased, moderately
# spread branching habit; the umbrella species holds a bare trunk for over
# half its height, then throws nearly horizontal ribs that droop toward
# their tips, giving a wide flat crown.
SPECIES = {
    "round": {
        "trunk_fraction": 0.42,
        "boughs": 2,
        "children": (3, 2),
        "spread": (0.35, 0.75),
        "up_bias": (0.22, -0.06),
        "child_length": (0.60, 0.78),
        "child_radius": (0.55, 0.70),
        "card_scale": 1.0,
    },
    "umbrella": {
        "trunk_fraction": 0.55,
        "boughs": 3,
        "children": (6, 2),
        "spread": (0.85, 1.20),
        "up_bias": (-0.02, -0.13),
        "child_length": (0.62, 0.82),
        "child_radius": (0.42, 0.58),
        "card_scale": 1.15,
    },
}


def generate_leaf_atlas():
    """Foliage cluster silhouettes: overlapping leaf stamps in a disc.

    Returns an alpha atlas of shape (LEAF_TEXTURES, S, S). Stamps are
    elongated and angled like leaves; the summed coverage is thresholded
    with a soft knee and modulated so the interior stays broken up.
    """
    rng = np.random.default_rng(TREE_SEED + 401)
    size = LEAT_ATLAS_SIZE
    axis_u = (np.arange(size, dtype=np.float32) + 0.5) / size
    grid_u, grid_v = np.meshgrid(axis_u, axis_u, indexing="ij")
    atlas = np.zeros((LEAF_TEXTURES, size, size), dtype=np.float32)
    for variant in range(LEAF_TEXTURES):
        coverage = np.zeros((size, size), dtype=np.float32)
        stamps = int(rng.integers(40, 56))
        for _ in range(stamps):
            angle = rng.uniform(0.0, np.pi)
            cu = rng.normal(0.5, 0.16)
            cv = rng.normal(0.5, 0.16)
            length = rng.uniform(0.10, 0.22)
            width = rng.uniform(0.035, 0.065)
            du = grid_u - cu
            dv = grid_v - cv
            along = du * np.cos(angle) + dv * np.sin(angle)
            across = -du * np.sin(angle) + dv * np.cos(angle)
            # Slight pointed-ellipse profile with a central notch.
            stamp = np.clip(1.0 - (along / length) ** 2 - (across / width) ** 2, 0.0, 1.0)
            coverage += stamp
        alpha = np.clip((coverage - 0.35) / 0.8, 0.0, 1.0)
        interior = 0.55 + 0.45 * np.clip(coverage / 2.2, 0.0, 1.0)
        atlas[variant] = alpha * interior
    return atlas


@ti.data_oriented
class TreeField:
    """Branch cones plus leaf cards; bounding volumes prune the ray loops."""

    def __init__(self):
        self.seg_a = ti.Vector.field(3, ti.f32, shape=MAX_SEGMENTS)
        self.seg_b = ti.Vector.field(3, ti.f32, shape=MAX_SEGMENTS)
        self.seg_radius = ti.Vector.field(2, ti.f32, shape=MAX_SEGMENTS)
        self.seg_bound = ti.Vector.field(4, ti.f32, shape=MAX_SEGMENTS)
        self.seg_parent = ti.field(ti.i32, shape=MAX_SEGMENTS)
        # Segments reordered into small spatial chunks (seg_order holds the
        # original segment index) so tree_hit tests chunk bounds first.
        self.seg_order = ti.field(ti.i32, shape=MAX_SEGMENTS)
        self.seg_group_center = ti.Vector.field(3, ti.f32, shape=MAX_SEG_GROUPS)
        self.seg_group_extent = ti.Vector.field(3, ti.f32, shape=MAX_SEG_GROUPS)
        self.seg_group_range = ti.Vector.field(2, ti.i32, shape=MAX_SEG_GROUPS)
        self.seg_group_tree = ti.field(ti.i32, shape=MAX_SEG_GROUPS)
        self.tree_bound = ti.Vector.field(4, ti.f32, shape=16)
        self.tree_range = ti.Vector.field(2, ti.i32, shape=16)
        self.leaves = CardField(MAX_TREE_CARDS, generate_leaf_atlas(), max_groups=64)
        self.count = 0
        self.tree_count = 0
        self.seg_group_count = 0
        self._chunk_write = 0
        self._leaf_group = 0
        self._build()
        if self._leaf_group > 0:
            self.leaves.finish_group()

    def _build(self):
        # Main framing trees, one per species so both silhouettes read in
        # the same frame. The yawed camera leaves a sliver of beach on the
        # left side of the frame, so the tall umbrella stands near the
        # waterline there; the lower round-crown one on the right stays off
        # the moon azimuth and leaves the glitter path open.
        self._main_tree(TREE_SEED + 1, 7.5, 12.5, height=6.5, trunk=0.22,
                        lean=(-0.10, -0.02), species="umbrella")
        self._main_tree(TREE_SEED + 2, -16.5, 10.5, height=4.8, trunk=0.16,
                        lean=(-0.02, 0.03), species="round")
        # Far-shore silhouettes stand on the wooded bank across the lake,
        # each one placed just inland of where the bank crests the
        # waterline, so the row reads as a shoreline treeline. The two
        # species alternate along the row.
        for i, (x, h) in enumerate([(-18.0, 6.2), (14.0, 5.4), (-28.0, 6.6),
                                    (24.0, 5.0), (4.0, 6.0), (-9.0, 5.2)]):
            z = self._far_shoreline(x)
            species = "umbrella" if i % 2 == 0 else "round"
            self._silhouette(TREE_SEED + 10 + i, x, z + 1.2, h, species)

    def _begin_tree(self):
        start = self.count
        return start

    @staticmethod
    def _far_shoreline(x):
        """First far-bank z whose terrain crests above the waterline.

        The bank rise is wiggled by noise, so the across-lake waterline
        meanders; scanning keeps every silhouette on dry ground whatever
        its x position.
        """
        z = 56.0
        while z < 92.0:
            if terrain_height_np(np.float32(x), np.float32(z)) > 0.45:
                return z
            z += 0.25
        return 78.0

    def _end_tree(self, start):
        lo = np.min(self.seg_a.to_numpy()[start:self.count], axis=0)
        hi = np.max(self.seg_b.to_numpy()[start:self.count], axis=0)
        i = self.tree_count
        self.tree_bound[i] = [*((lo + hi) / 2), float(np.linalg.norm((hi - lo) / 2)) + 0.3]
        self.tree_range[i] = (start, self.count)
        self.tree_count += 1
        self._chunk_segments(start, i)

    def _chunk_segments(self, start, tree):
        """Bucket one tree's segments into small spatial chunks.

        Segment indices are appended to seg_order in cell order so each
        chunk bound stays tight; the ray loop skips chunks it never enters.
        seg_parent keeps pointing at original indices, which stay valid.
        """
        a = self.seg_a.to_numpy()[start:self.count]
        b = self.seg_b.to_numpy()[start:self.count]
        r = self.seg_radius.to_numpy()[start:self.count]
        cell = 2.0
        key = (np.minimum(a, b) // cell).astype(np.int64)
        order = np.lexsort((key[:, 2], key[:, 0], key[:, 1]))
        lo_all = np.minimum(a - r[:, 0:1], b - r[:, 1:2])
        hi_all = np.maximum(a + r[:, 0:1], b + r[:, 1:2])
        pending_lo, pending_hi = [], []
        for local in order:
            seg = start + int(local)
            self.seg_order[self._chunk_write] = seg
            self._chunk_write += 1
            pending_lo.append(lo_all[local])
            pending_hi.append(hi_all[local])
            if len(pending_lo) >= SEG_GROUP_SIZE:
                self._finish_seg_chunk(tree, pending_lo, pending_hi)
                pending_lo, pending_hi = [], []
        if pending_lo:
            self._finish_seg_chunk(tree, pending_lo, pending_hi)

    def _finish_seg_chunk(self, tree, pending_lo, pending_hi):
        if self.seg_group_count >= MAX_SEG_GROUPS:
            return
        lo = np.min(pending_lo, axis=0)
        hi = np.max(pending_hi, axis=0)
        i = self.seg_group_count
        self.seg_group_center[i] = (lo + hi) / 2
        self.seg_group_extent[i] = (hi - lo) / 2 + 0.02
        self.seg_group_range[i] = (self._chunk_write - len(pending_lo),
                                   self._chunk_write)
        self.seg_group_tree[i] = tree
        self.seg_group_count += 1

    def _segment(self, start, a, b, radius_a, radius_b, parent):
        if self.count >= MAX_SEGMENTS:
            return -1
        i = self.count
        self.seg_a[i], self.seg_b[i] = a, b
        self.seg_radius[i] = [max(radius_a, 0.004), max(radius_b, 0.003)]
        self.seg_bound[i] = [*((a + b) / 2), float(np.linalg.norm(b - a) / 2
                                                  + max(radius_a, radius_b)) + 0.02]
        self.seg_parent[i] = parent
        self.count += 1
        return i

    def _grow(self, rng, start, base, initial, height, trunk, card_scale,
              species="round"):
        """Recursive skeleton; returns nothing, appends segments and cards.

        The species parameter set drives the branching habit: child count,
        spread, upward bias (negative biases droop), and the length and
        radius decay curves all come from SPECIES.
        """
        params = SPECIES[species]
        # Recursive child lengths compound along upward chains and would
        # roughly double the tree above its nominal stature; cap each
        # branch so no tip rises far past the height parameter. Lateral
        # growth is untouched.
        top_limit = base[1] + height * 1.30

        def branch(origin, direction, length, radius, level, parent):
            direction = direction / np.linalg.norm(direction)
            if direction[1] > 0.05:
                room = (top_limit - origin[1]) / direction[1]
                length = min(length, max(0.25, room))
            end = origin + direction * length
            tip_radius = radius * (0.62 if level + 1 < MAX_LEVEL else 0.35)
            index = self._segment(start, origin, end, radius, tip_radius, parent)
            if index < 0:
                return
            if level + 1 >= MAX_LEVEL or length < 0.22:
                self._leaf_card(rng, end, direction, card_scale
                                * params["card_scale"], 2)
                return
            children = params["children"][0] if level < 2 else params["children"][1]
            for _ in range(children):
                azimuth = rng.uniform(0.0, 2.0 * np.pi)
                spread = rng.uniform(*params["spread"])
                reference = np.array([1.0, 0.0, 0.0])
                if abs(direction[0]) > 0.8:
                    reference = np.array([0.0, 0.0, 1.0])
                perp = np.cross(direction, reference)
                perp /= np.linalg.norm(perp)
                perp2 = np.cross(direction, perp)
                child = (direction * np.cos(spread)
                         + (perp * np.cos(azimuth) + perp2 * np.sin(azimuth)) * np.sin(spread))
                child[1] += params["up_bias"][0] + params["up_bias"][1] * level
                child /= np.linalg.norm(child)
                child_length = length * rng.uniform(*params["child_length"])
                child_radius = radius * rng.uniform(*params["child_radius"])
                junction = end - direction * (length * 0.05)
                branch(junction, child, child_length, child_radius, level + 1, index)
                if level + 2 >= MAX_LEVEL:
                    self._leaf_card(rng, junction + child * (child_length * 0.6),
                                    child, card_scale * params["card_scale"], 1)

        trunk_top = base + initial * (height * params["trunk_fraction"])
        # Base flare: a short buttressed cone at ground level so the trunk
        # grows out of the soil instead of landing on it like a pole.
        flare_top = base + initial * (height * 0.12)
        self._segment(start, base - initial * 0.15, flare_top,
                      trunk * 1.9, trunk * 1.15, -1)
        index = self._segment(start, base, trunk_top, trunk, trunk * 0.62, -1)
        branch(trunk_top, initial, height * (1.0 - params["trunk_fraction"]),
               trunk * 0.62, 1, index)
        # A few low boughs directly off the trunk.
        for _ in range(params["boughs"]):
            azimuth = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(azimuth), 0.55, np.sin(azimuth)])
            direction /= np.linalg.norm(direction)
            origin = base + (trunk_top - base) * rng.uniform(0.55, 0.8)
            child_length = height * rng.uniform(0.16, 0.22)
            child_radius = trunk * rng.uniform(0.30, 0.40)
            bough = self._segment(start, origin, origin + direction * child_length,
                                  child_radius, child_radius * 0.5, index)
            if bough >= 0:
                branch(origin + direction * child_length, direction,
                       child_length * 0.7, child_radius * 0.55, MAX_LEVEL - 2, bough)

    def _leaf_card(self, rng, centre, direction, scale, count):
        if self.leaves.count + count > self.leaves.capacity:
            return
        for _ in range(count):
            azimuth = rng.uniform(0.0, 2.0 * np.pi)
            u = np.array([np.cos(azimuth), rng.uniform(-0.2, 0.2), np.sin(azimuth)])
            u /= np.linalg.norm(u)
            v = np.array([rng.uniform(-0.3, 0.3), 1.0, rng.uniform(-0.3, 0.3)])
            v -= u * np.dot(u, v)
            v /= np.linalg.norm(v)
            half = rng.uniform(0.50, 0.85) * scale
            offset = np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.0, 0.3),
                               rng.uniform(-0.2, 0.2)]) * scale
            self.leaves.add_card(centre + offset, u * half, v * half, 1,
                                 int(rng.integers(0, LEAF_TEXTURES)),
                                 rng.random(), 1.0)
        self._leaf_group += count
        if self._leaf_group >= 48:
            self.leaves.finish_group()
            self._leaf_group = 0

    def _main_tree(self, seed, x, z, height, trunk, lean, species="round"):
        rng = np.random.default_rng(seed)
        start = self._begin_tree()
        ground = float(terrain_height_np(np.float32(x), np.float32(z)))
        base = np.array([x, ground - 0.12, z])
        initial = np.array([lean[0], 1.0, lean[1]])
        initial /= np.linalg.norm(initial)
        self._grow(rng, start, base, initial, height, trunk, card_scale=1.0,
                   species=species)
        self._end_tree(start)

    def _silhouette(self, seed, x, z, height, species="round"):
        rng = np.random.default_rng(seed)
        start = self._begin_tree()
        ground = float(terrain_height_np(np.float32(x), np.float32(z)))
        base = np.array([x, ground - 0.1, z])
        top = base + np.array([rng.uniform(-0.2, 0.2), height * 0.55,
                               rng.uniform(-0.2, 0.2)])
        index = self._segment(start, base, top, 0.14, 0.08, -1)
        if species == "umbrella":
            # Umbrella silhouettes: evenly spaced ribs that droop at the
            # tips, reading as flat caps against the sky.
            for k in range(6):
                azimuth = 2.0 * np.pi * k / 6.0 + rng.uniform(-0.2, 0.2)
                direction = np.array([np.cos(azimuth),
                                      rng.uniform(0.10, 0.30), np.sin(azimuth)])
                direction /= np.linalg.norm(direction)
                origin = base + (top - base) * rng.uniform(0.75, 0.95)
                end = origin + direction * rng.uniform(1.1, 1.7)
                end[1] -= rng.uniform(0.1, 0.35)
                self._segment(start, origin, end, 0.045, 0.018, index)
                self._leaf_card(rng, end, direction,
                                scale=rng.uniform(1.4, 1.9), count=1)
        else:
            for _ in range(5):
                azimuth = rng.uniform(0.0, 2.0 * np.pi)
                direction = np.array([np.cos(azimuth), rng.uniform(0.5, 0.9),
                                      np.sin(azimuth)])
                direction /= np.linalg.norm(direction)
                origin = base + (top - base) * rng.uniform(0.5, 0.95)
                end = origin + direction * rng.uniform(0.9, 1.6)
                self._segment(start, origin, end, 0.05, 0.02, index)
                self._leaf_card(rng, end, direction, scale=rng.uniform(1.3, 1.9),
                                count=1)
        self._leaf_card(rng, top + np.array([0.0, 0.5, 0.0]),
                        np.array([0.0, 1.0, 0.0]), scale=1.6, count=2)
        self._end_tree(start)

    @ti.func
    def cone_hit(self, origin, direction, index):
        """Ray against the truncated cone of one segment."""
        t = 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        bound = self.seg_bound[index]
        if sphere_hit(origin, direction, ti.Vector([bound.x, bound.y, bound.z]), bound.w) < 1e5:
            a = self.seg_a[index]
            axis = self.seg_b[index] - a
            length = axis.norm()
            u = axis / length
            radii = self.seg_radius[index]
            slope = (radii.y - radii.x) / length
            oc = origin - a
            s0 = oc.dot(u)
            d0 = oc - u * s0
            du = direction.dot(u)
            dd = direction - u * du
            r0 = radii.x + slope * s0
            qa = dd.dot(dd) - slope * slope * du * du
            qb = 2.0 * (d0.dot(dd) - r0 * slope * du)
            qc = d0.dot(d0) - r0 * r0
            discriminant = qb * qb - 4.0 * qa * qc
            if ti.abs(qa) < 1e-9:
                qa = 1e-9
            if discriminant > 0.0:
                root = ti.sqrt(discriminant)
                for sign in ti.static((-1.0, 1.0)):
                    candidate = (-qb + sign * root) / (2.0 * qa)
                    s = s0 + candidate * du
                    if candidate > 0.001 and s >= 0.0 and s <= length and candidate < t:
                        t = candidate
                        radial = d0 + dd * candidate
                        radius = ti.max(r0 + slope * (s - s0), 1e-4)
                        normal = (radial - u * (slope * radius)).normalized()
        return t, normal

    @ti.func
    def trees_hit(self, origin, direction, density, t_max):
        """Closest segment across all trees; returns (t, index, normal).

        Chunk bounds prune first, the owning tree's bounding sphere and the
        density gate prune second, and only then the segment cone is tested.
        """
        best_t, best_i = 1e6, -1
        normal = ti.Vector([0.0, 1.0, 0.0])
        for g in range(self.seg_group_count):
            tree = self.seg_group_tree[g]
            gate = ti.sin(float(tree) * 78.233) * 43758.5453
            if gate - ti.floor(gate) <= density:
                bound = self.tree_bound[tree]
                broad = sphere_hit(origin, direction,
                                   ti.Vector([bound.x, bound.y, bound.z]), bound.w)
                if broad < ti.min(t_max, best_t):
                    extent = self.seg_group_extent[g]
                    near = sphere_hit(origin, direction, self.seg_group_center[g],
                                      extent.norm())
                    if near < ti.min(t_max, best_t):
                        for i in range(self.seg_group_range[g][0],
                                       self.seg_group_range[g][1]):
                            t, n = self.cone_hit(origin, direction, self.seg_order[i])
                            if t < best_t:
                                best_t, best_i, normal = t, self.seg_order[i], n
        return best_t, best_i, normal
