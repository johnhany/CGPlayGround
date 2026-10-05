"""Shared courtyard scene, orbit camera, and PBR surface shading for water demos.

The static scene (beveled boxes, grouped oriented ellipsoids, procedural pool
floor), the procedural HDR environment lighting, and the Cook-Torrance shading
are common to the water demo series. Each demo subclasses CourtyardScene and
adds its own wave field, water intersection, and render kernels.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
from PIL import Image
import taichi as ti

if __package__:
    from .pbr import EnvironmentLighting, fresnel_schlick, ggx_brdf
    from .finish import PostProcess
else:
    from pbr import EnvironmentLighting, fresnel_schlick, ggx_brdf
    from finish import PostProcess

# Pool interior in world units; the water surface rests at y = 0.
POOL_HALF_X = 3.2
POOL_HALF_Z = 2.15
BED_HEIGHT = -0.68


@dataclass
class OrbitCamera:
    """World-space orbit camera; restricted to views above the water."""

    yaw: float = 0.43
    pitch: float = 0.43
    distance: float = 11.6
    target: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.08, 0.0]))
    fov: float = 48.0

    def preset(self, name):
        self.target = np.array([0.0, -0.08, 0.0])
        self.yaw, self.pitch, self.distance = {
            "overview": (0.43, 0.43, 11.6),
            "waterline": (0.12, 0.23, 8.5),
            "top": (0.0, 1.40, 9.7),
        }[name]

    def basis(self):
        offset = np.array([
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.cos(self.yaw) * math.cos(self.pitch),
        ]) * self.distance
        eye = self.target + offset
        forward = -offset / np.linalg.norm(offset)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return tuple(v.astype(np.float32) for v in (eye, forward, right, up))

    def orbit(self, dx, dy):
        self.yaw -= dx * 3.5
        self.pitch = float(np.clip(self.pitch - dy * 3.0, 0.10, 1.48))

    def zoom(self, amount):
        self.distance = float(np.clip(self.distance * math.exp(amount), 4.5, 20.0))

    def pan(self, dx, dy):
        _, _, right, _ = self.basis()
        horizontal_forward = np.array([-right[2], 0.0, right[0]])
        self.target -= (right * dx + horizontal_forward * dy) * self.distance
        self.target[0] = float(np.clip(self.target[0], -3.5, 3.5))
        self.target[1] = -0.08
        self.target[2] = float(np.clip(self.target[2], -2.5, 2.5))


@ti.data_oriented
class CourtyardScene:
    """Static geometry, environment lighting, caustic map, and PBR shading.

    Geometry uses beveled boxes and grouped oriented ellipsoids; the pool
    floor uses a procedural pebble material. Subclasses provide the water
    surface (analytical or simulated) and the render kernels.
    """

    def __init__(self, width, height):
        self.environment = EnvironmentLighting()
        self.light_controls = ti.Vector.field(4, ti.f32, shape=())
        self.light_controls[None] = [math.radians(1.2), 4, 0.65, 1.0]
        self.materials = ti.Vector.field(3, ti.f32, shape=10)
        self.materials.from_numpy(np.array([
            [.58, 0, .04], [.48, 0, .04], [.72, 0, .04], [.4, 0, .04],
            [.88, 0, .04], [.68, 0, .04], [.50, 0, .04], [.32, 1, .04],
            [.10, 0, .04], [.95, 0, .04]], dtype=np.float32))
        self.width, self.height = width, height
        self.post = PostProcess(width, height)
        self.image = self.post.image
        self.water_controls = ti.Vector.field(4, ti.f32, shape=())
        self.water_controls[None] = [0.16, 1.0, 1.0, 4.0]
        self.post_controls = ti.Vector.field(3, ti.f32, shape=())
        self.post_controls[None] = [0.08, 1.0, 0.12]
        self.caustic_raw = ti.field(ti.f32, shape=(128, 96))
        self.caustic_map = ti.field(ti.f32, shape=(128, 96))
        self.box_center = ti.Vector.field(3, ti.f32, shape=128)
        self.box_extent = ti.Vector.field(3, ti.f32, shape=128)
        self.box_color = ti.Vector.field(3, ti.f32, shape=128)
        self.box_material = ti.field(ti.i32, shape=128)
        self.box_bevel = ti.field(ti.f32, shape=128)
        self.leaf_center = ti.Vector.field(3, ti.f32, shape=384)
        self.leaf_radius = ti.Vector.field(3, ti.f32, shape=384)
        self.leaf_rotation = ti.Matrix.field(3, 3, ti.f32, shape=384)
        self.leaf_color = ti.Vector.field(3, ti.f32, shape=384)
        self.leaf_material = ti.field(ti.i32, shape=384)
        self.group_center = ti.Vector.field(3, ti.f32, shape=12)
        self.group_extent = ti.Vector.field(3, ti.f32, shape=12)
        self.group_range = ti.Vector.field(2, ti.i32, shape=12)
        self.boxes, self.leaves, self.groups = 0, 0, 0
        self.build_static()

    def build_static(self):
        """Populate the static scene; ocean demos override with their own."""
        self._build_courtyard()

    def _build_courtyard(self):
        """Deterministic modern courtyard; no downloaded or generated assets."""
        def box(center, extent, color=(0.66, 0.62, 0.53), material=0, bevel=0.025):
            if self.boxes >= self.box_center.shape[0]:
                raise ValueError("Courtyard box capacity exceeded")
            i = self.boxes
            self.box_center[i], self.box_extent[i] = center, extent
            self.box_color[i], self.box_material[i] = color, material
            self.box_bevel[i] = min(bevel, min(extent) * 0.8)
            self.boxes += 1

        group_bounds = []

        def ellipsoid(center, radii, color, axis=(0, 1, 0), material=2):
            if self.leaves >= self.leaf_center.shape[0]:
                raise ValueError("Courtyard foliage capacity exceeded")
            up = np.array(axis, dtype=float)
            up /= np.linalg.norm(up)
            reference = np.array([1., 0., 0.]) if abs(up[0]) < 0.8 else np.array([0., 0., 1.])
            right = np.cross(up, reference)
            right /= np.linalg.norm(right)
            forward = np.cross(right, up)
            rotation = np.stack([right, up, forward], axis=1)
            i = self.leaves
            self.leaf_center[i], self.leaf_radius[i] = center, radii
            self.leaf_rotation[i], self.leaf_color[i] = rotation, color
            self.leaf_material[i] = material
            self.leaves += 1
            bounds = np.abs(rotation) @ np.array(radii)
            group_bounds.append((np.array(center) - bounds, np.array(center) + bounds))

        def branch(a, b, radius):
            a, b = np.array(a), np.array(b)
            ellipsoid((a + b) / 2, (radius, np.linalg.norm(b - a) / 2 + radius, radius),
                      (0.19, 0.12, 0.065), axis=b - a, material=1)

        def finish_group(start):
            if self.groups >= self.group_center.shape[0]:
                raise ValueError("Courtyard foliage group capacity exceeded")
            lo = np.min([v[0] for v in group_bounds], axis=0)
            hi = np.max([v[1] for v in group_bounds], axis=0)
            i = self.groups
            self.group_center[i], self.group_extent[i] = (lo + hi) / 2, (hi - lo) / 2 + 0.002
            self.group_range[i] = (start, self.leaves)
            self.groups += 1
            group_bounds.clear()

        stone, plaster = (0.66, 0.62, 0.53), (0.73, 0.69, 0.59)
        wood, metal = (0.23, 0.115, 0.050), (0.042, 0.052, 0.050)
        # Separate coping from submerged walls; the bevel is visible geometry.
        for x in (-3.40, 3.40):
            box((x, -0.025, 0), (0.20, 0.15, 2.55), stone, bevel=0.045)
            box((x, -0.43, 0), (0.20, 0.28, 2.55), (0.29, 0.34, 0.30), 4)
        for z in (-2.35, 2.35):
            box((0, -0.025, z), (3.20, 0.15, 0.20), stone, bevel=0.045)
            box((0, -0.43, z), (3.20, 0.28, 0.20), (0.29, 0.34, 0.30), 4)
        # Enclosing walls establish a designed courtyard instead of an infinite lawn.
        box((0, 0.98, -5.5), (6.6, 1.18, 0.16), plaster, 4, 0.04)
        box((-6.35, 0.61, 0), (0.16, 0.81, 5.5), plaster, 4, 0.04)
        box((6.35, 0.20, 0), (0.16, 0.40, 5.5), stone, 4, 0.04)
        # Recessed facade: dark glazing, stone frame and restrained black trim.
        box((3.35, 0.99, -5.30), (1.90, 1.04, 0.045), metal, 7, 0.01)
        box((3.35, 0.99, -5.24), (1.80, 0.94, 0.02), (0.075, 0.15, 0.16), 8, 0.008)
        for x in (1.55, 3.35, 5.15):
            box((x, 0.99, -5.20), (0.025, 0.99, 0.02), metal, 7, 0.008)
        # Timber pergola: believable thickness, inset metal feet and cross beams.
        for x in (-1.9, 1.05):
            for z in (-4.85, -3.05):
                box((x, 1.01, z), (0.065, 1.21, 0.065), wood, 1, 0.012)
                box((x, -0.12, z), (0.085, 0.08, 0.085), metal, 7, 0.01)
        for z in (-4.90, -3.00):
            box((-0.425, 2.23, z), (1.80, 0.105, 0.075), wood, 1, 0.018)
        for x in np.linspace(-2.05, 1.2, 10):
            box((float(x), 2.38, -3.95), (0.055, 0.055, 1.15), wood, 1, 0.013)
        # Floating bench with spaced timber boards.
        for z in np.linspace(-4.2, -3.6, 5):
            box((-0.4, 0.22, float(z)), (1.18, 0.045, 0.052), wood, 1, 0.014)
        for x in (-1.3, 0.5):
            box((x, -0.03, -3.9), (0.06, 0.20, 0.27), metal, 7, 0.012)
        # Two lounge chairs on the right-hand deck, with slim dark frames.
        for z in (-2.65, -0.85):
            box((4.75, 0.16, z), (0.50, 0.11, 0.61), (0.61, 0.56, 0.44), 9, 0.07)
            box((4.75, 0.45, z - 0.53), (0.50, 0.23, 0.10), (0.61, 0.56, 0.44), 9, 0.055)
            for x in (4.26, 5.24):
                box((x, -0.06, z), (0.025, 0.14, 0.52), metal, 7, 0.014)
        box((4.65, 0.12, 0.45), (0.37, 0.045, 0.37), wood, 1, 0.025)
        box((4.65, -0.06, 0.45), (0.08, 0.15, 0.08), metal, 7, 0.012)
        # Landscape planters and low bollard lights.
        for x, z, ex, ez in [(-4.55, -2.35, 0.80, 1.20), (4.6, 2.45, 0.80, 1.05), (-4.55, 2.3, 0.8, 0.85)]:
            box((x, -0.04, z), (ex, 0.18, ez), (0.31, 0.32, 0.28), 4, 0.035)
            box((x, 0.12, z), (ex - 0.06, 0.025, ez - 0.06), (0.095, 0.105, 0.068), 4, 0.01)
        for x, z in [(-3.95, 3.9), (3.9, 3.9)]:
            box((x, 0.04, z), (0.085, 0.24, 0.085), metal, 7, 0.015)
            box((x, 0.25, z), (0.075, 0.04, 0.075), (0.8, 0.48, 0.20), 3, 0.01)

        rng = np.random.default_rng(17)
        # Two sculptural olive trees: branched stems and flattened leaf sprays.
        for x, z, size in [(-4.55, -2.6, 1.0), (5.65, -4.4, 0.82)]:
            start = self.leaves
            root = np.array([x, 0.14, z])
            fork = root + np.array([0.05, 1.28 * size, 0.03])
            branch(root, fork, 0.065 * size)
            for angle in np.linspace(0, 2 * math.pi, 7, endpoint=False):
                tip = fork + np.array([math.cos(angle) * 0.83, 0.67 + 0.27 * rng.random(), math.sin(angle) * 0.76]) * size
                branch(fork, tip, 0.027 * size)
                for _ in range(14):
                    offset = rng.normal(0, [0.30, 0.19, 0.30]) * size
                    center = tip + offset
                    radii = np.array([0.20, 0.055, 0.115]) * size * rng.uniform(0.7, 1.3)
                    green = np.array([0.19, 0.27, 0.16]) * rng.uniform(0.75, 1.20)
                    axis = (rng.uniform(-0.7, 0.7), 1.0, rng.uniform(-0.7, 0.7))
                    ellipsoid(center, radii, green, axis)
            finish_group(start)
        # Upright, irregular grasses replace the old spherical shrubs.
        for x, z in [(-4.5, 2.3), (4.55, 2.45), (-4.5, -1.8)]:
            start = self.leaves
            for _ in range(23):
                radius, angle = rng.uniform(0.05, 0.6), rng.uniform(0, 2 * math.pi)
                height = rng.uniform(0.26, 0.60)
                axis = (math.cos(angle) * 0.65, 1.0, math.sin(angle) * 0.65)
                center = (x + radius * math.cos(angle), 0.18 + height * 0.65, z + radius * math.sin(angle))
                color = np.array([0.25, 0.29, 0.12]) * rng.uniform(0.65, 1.3)
                ellipsoid(center, (0.034, height, 0.045), color, axis)
            finish_group(start)
        # Asymmetric river rocks form a foreground group, with visible silhouettes.
        start = self.leaves
        for center, radii in [((-4.7, 0.26, 2.65), (0.35, 0.20, 0.27)),
                              ((-4.35, 0.23, 2.4), (0.22, 0.16, 0.31)),
                              ((-4.9, 0.20, 1.95), (0.23, 0.11, 0.18))]:
            ellipsoid(center, radii, (0.35, 0.37, 0.32), (0.3, 1, 0.2), 4)
        finish_group(start)

    @ti.func
    def smooth(self, a, b, x):
        u = ti.max(0.0, ti.min(1.0, (x - a) / (b - a)))
        return u * u * (3.0 - 2.0 * u)

    @ti.func
    def hash2(self, p):
        v = ti.sin(p.dot(ti.Vector([127.1, 311.7]))) * 43758.5453
        return v - ti.floor(v)

    @ti.func
    def box_hit(self, origin, direction, center, extent):
        near, far = -1e6, 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        exit_normal = normal
        valid = 1
        for axis in ti.static(range(3)):
            delta = origin[axis] - center[axis]
            if ti.abs(direction[axis]) < 1e-6:
                if ti.abs(delta) > extent[axis]:
                    valid = 0
            else:
                a = (-extent[axis] - delta) / direction[axis]
                b = (extent[axis] - delta) / direction[axis]
                lo, hi = ti.min(a, b), ti.max(a, b)
                if lo > near:
                    near = lo
                    normal = ti.Vector([0.0, 0.0, 0.0])
                    normal[axis] = -ti.math.sign(direction[axis])
                if hi < far:
                    far = hi
                    exit_normal = ti.Vector([0.0, 0.0, 0.0])
                    exit_normal[axis] = ti.math.sign(direction[axis])
        t = 1e6
        if valid and far >= ti.max(near, 0.001):
            t = near
            if near < 0.001:
                t, normal = far, exit_normal
        return t, normal

    @ti.func
    def rounded_box_hit(self, origin, direction, center, extent, radius):
        broad_t, normal = self.box_hit(origin, direction, center, extent)
        result = 1e6
        if broad_t < 1e5:
            t = broad_t
            if (ti.abs(origin - center) < extent).all():
                t = 0.0011
            found = 0
            # Sphere tracing is bounded to the already intersected box.
            for _ in range(24):
                if not found and t < broad_t + 2.0 * extent.norm() + 0.01:
                    p = origin + direction * t - center
                    q = ti.abs(p) - (extent - radius)
                    positive = ti.max(q, 0.0)
                    distance = positive.norm() + ti.min(ti.max(q.x, ti.max(q.y, q.z)), 0.0) - radius
                    if ti.abs(distance) < 0.0002:
                        found, result = 1, t
                        if positive.norm() > 1e-6:
                            normal = positive.normalized() * ti.math.sign(p)
                        else:
                            axis = 0
                            if q.y > q.x:
                                axis = 1
                            if q.z > q[axis]:
                                axis = 2
                            normal = ti.Vector([0.0, 0.0, 0.0])
                            normal[axis] = ti.math.sign(p[axis])
                    else:
                        t += ti.max(ti.abs(distance), 0.0001)
        return result, normal

    @ti.func
    def ellipsoid_hit(self, origin, direction, index):
        rotation = self.leaf_rotation[index]
        radii = self.leaf_radius[index]
        local_origin = rotation.transpose() @ (origin - self.leaf_center[index])
        local_direction = rotation.transpose() @ direction
        o, d = local_origin / radii, local_direction / radii
        a, b, c = d.dot(d), o.dot(d), o.dot(o) - 1.0
        discriminant = b * b - a * c
        t = 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        if discriminant > 0.0:
            root = ti.sqrt(discriminant)
            candidate = (-b - root) / a
            if candidate < 0.001:
                candidate = (-b + root) / a
            if candidate > 0.001:
                t = candidate
                normal = (rotation @ ((local_origin + local_direction * t) / (radii * radii))).normalized()
        return t, normal

    @ti.func
    def geometry(self, origin, direction):
        closest, material = 1e6, -1
        normal, color = ti.Vector([0.0, 1.0, 0.0]), ti.Vector([0.5, 0.5, 0.5])
        if ti.abs(direction.y) > 1e-6:
            # The ground has a cutout for the pool; the bed is a separate plane.
            for level in ti.static(range(2)):
                elevation = -0.20
                if ti.static(level == 1):
                    elevation = BED_HEIGHT
                t = (elevation - origin.y) / direction.y
                p = origin + direction * t
                inside = ti.abs(p.x) < POOL_HALF_X and ti.abs(p.z) < POOL_HALF_Z
                valid = not inside
                if ti.static(level == 1):
                    valid = inside
                if t > 0.001 and t < closest and valid:
                    closest, material = t, 5 + level
        for i in range(self.boxes):
            t, n = self.rounded_box_hit(origin, direction, self.box_center[i], self.box_extent[i], self.box_bevel[i])
            if t < closest:
                closest, normal = t, n
                color, material = self.box_color[i], self.box_material[i]
        for group in range(self.groups):
            bound, _ = self.box_hit(origin, direction, self.group_center[group], self.group_extent[group])
            inside = (ti.abs(origin - self.group_center[group]) < self.group_extent[group]).all()
            if bound < closest or inside:
                for i in range(self.group_range[group][0], self.group_range[group][1]):
                    t, n = self.ellipsoid_hit(origin, direction, i)
                    if t < closest:
                        closest, normal = t, n
                        material, color = self.leaf_material[i], self.leaf_color[i]
        return closest, normal, color, material

    @ti.func
    def sun_direction(self, lighting):
        return ti.Vector([-0.50, 0.22 + 0.73 * lighting, -0.57]).normalized()

    @ti.func
    def sky(self, direction, lighting):
        color = self.environment.sample_map(self.environment.environment, direction, lighting)
        alignment = ti.max(0.0, direction.dot(self.sun_direction(lighting)))
        color += ti.Vector([1.0, 0.65, 0.29]) * (0.08 * alignment ** 24 + 4.0 * alignment ** 1600)
        return color

    @ti.func
    def occlusion_distance(self, origin, direction, limit):
        closest = limit
        for i in range(self.boxes):
            bound, _ = self.box_hit(origin, direction, self.box_center[i], self.box_extent[i])
            inside = (ti.abs(origin - self.box_center[i]) < self.box_extent[i]).all()
            if bound < closest or inside:
                t, _ = self.rounded_box_hit(origin, direction, self.box_center[i], self.box_extent[i], self.box_bevel[i])
                closest = ti.min(closest, t)
        for group in range(self.groups):
            bound, _ = self.box_hit(origin, direction, self.group_center[group], self.group_extent[group])
            inside = (ti.abs(origin - self.group_center[group]) < self.group_extent[group]).all()
            if bound < closest or inside:
                for i in range(self.group_range[group][0], self.group_range[group][1]):
                    t, _ = self.ellipsoid_hit(origin, direction, i)
                    closest = ti.min(closest, t)
        # Ground and pool bed also occlude short contact rays.
        if direction.y < -0.0001:
            for level in ti.static(range(2)):
                height = -0.20 - 0.48 * level
                t = (height - origin.y) / direction.y
                p = origin + direction * t
                inside = ti.abs(p.x) < POOL_HALF_X and ti.abs(p.z) < POOL_HALF_Z
                valid = not inside
                if ti.static(level == 1):
                    valid = inside
                if t > 0.001 and valid:
                    closest = ti.min(closest, t)
        return closest

    @ti.func
    def tangent_basis(self, normal):
        reference = ti.Vector([0.0, 1.0, 0.0])
        if ti.abs(normal.y) > 0.95:
            reference = ti.Vector([1.0, 0.0, 0.0])
        tangent = reference.cross(normal).normalized()
        return tangent, normal.cross(tangent)

    @ti.func
    def visibility(self, p, normal, sun, detail):
        tangent, bitangent = self.tangent_basis(sun)
        count = 1
        if detail == 1:
            count = ti.cast(self.light_controls[None].y, ti.i32)
        visible = 0.0
        for sample in range(count):
            offset = ti.Vector([0.0, 0.0])
            if count == 4:
                angle = sample * 2.399963 + 0.4
                radius = ti.sqrt((sample + 0.5) / 4.0) * self.light_controls[None].x
                offset = ti.Vector([ti.cos(angle), ti.sin(angle)]) * radius
            direction = (sun + tangent * offset.x + bitangent * offset.y).normalized()
            visible += ti.cast(self.occlusion_distance(p + normal * 0.006, direction, 18.0) >= 18.0, ti.f32)
        return visible / count

    @ti.func
    def contact_ao(self, p, normal):
        tangent, bitangent = self.tangent_basis(normal)
        occlusion = 0.0
        for sample in range(3):
            angle = sample * 2.399963 + 0.7
            direction = (normal * 0.7 + tangent * ti.cos(angle) * 0.714 + bitangent * ti.sin(angle) * 0.714).normalized()
            distance = self.occlusion_distance(p + normal * 0.008, direction, 0.40)
            occlusion += 1.0 - self.smooth(0.01, 0.40, distance)
        return 1.0 - self.light_controls[None].z * occlusion / 3.0

    @ti.func
    def floor_material(self, p, clock, amplitude):
        # Elliptic cellular pebbles with approximate relief normals.
        uv = ti.Vector([p.x, p.z]) * 6.3
        cell = ti.floor(uv)
        nearest = 10.0
        chosen, seed = ti.Vector([0.0, 0.0]), 0.0
        for a, b in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            index = cell + ti.Vector([a, b])
            random = self.hash2(index)
            center = index + ti.Vector([0.24 + 0.52 * random, 0.24 + 0.52 * self.hash2(index + 19.1)])
            delta = (uv - center) * ti.Vector([1.0, 1.35])
            distance = delta.norm()
            if distance < nearest:
                nearest, chosen, seed = distance, delta, random
        stone = ti.Vector([0.38, 0.42, 0.34]) * (0.70 + 0.45 * seed)
        stone = stone * (1.0 - 0.32 * self.smooth(0.36, 0.57, nearest))
        grout = self.smooth(0.47, 0.55, nearest)
        color = stone * (1.0 - grout) + ti.Vector([0.15, 0.18, 0.13]) * grout
        n = ti.Vector([chosen.x * 0.8, 1.0, chosen.y * 1.1]).normalized()
        return color, n

    @ti.func
    def surface_properties(self, p, material, properties):
        return properties

    @ti.func
    def surface_sun_color(self, p, material, lighting, color):
        return color

    @ti.func
    def surface_sun_visibility(self, p, normal, material, lighting, detail):
        shade = 1.0
        if material == 6 and self.water_controls[None].z > 0.0:
            shade = 1.0 - self.water_controls[None].z + self.water_controls[None].z * self.caustic_density(p)
        elif normal.dot(self.sun_direction(lighting)) > 0.0:
            shade = self.visibility(p, normal, self.sun_direction(lighting), detail)
        return shade

    @ti.func
    def shade_surface(self, origin, direction, t, normal, base, material, clock, lighting, amplitude, detail):
        color = self.sky(direction, lighting)
        if material >= 0:
            p = origin + direction * t
            grain = self.hash2(ti.floor(ti.Vector([p.x + p.y, p.z - p.y]) * 65.0))
            if material == 0:
                # Coping joints run along each long edge.
                coordinate = p.x
                if ti.abs(p.x) > 3.2:
                    coordinate = p.z
                joint = coordinate * 1.8 - ti.floor(coordinate * 1.8)
                base *= (0.96 + 0.04 * grain) * (0.72 + 0.28 * self.smooth(0.01, 0.04, joint))
            elif material == 1:
                grain_line = ti.sin(70.0 * p.z + 1.8 * ti.sin(p.x * 5.0 + p.y * 3.0))
                base *= 0.95 + 0.05 * grain_line
            elif material == 2:
                base *= 0.90 + 0.10 * ti.max(0.0, normal.y)
            elif material == 4:
                base *= 0.99 + 0.02 * grain
            elif material == 5:
                if 3.6 < p.x < 6.2 and ti.abs(p.z) < 4.4:
                    plank = p.z * 6.5 - ti.floor(p.z * 6.5)
                    base = ti.Vector([0.23, 0.115, 0.055]) * (0.91 + 0.045 * ti.sin(p.x * 27 + ti.sin(p.z * 4)))
                    base *= 0.65 + 0.35 * self.smooth(0.006, 0.04, ti.min(plank, 1.0 - plank))
                elif ti.abs(p.x) < 6.2 and ti.abs(p.z) < 5.4:
                    uv = ti.Vector([p.x / 1.2, p.z / 0.8])
                    tile = ti.floor(uv)
                    f = uv - tile
                    edge = ti.min(ti.min(f.x, 1.0 - f.x), ti.min(f.y, 1.0 - f.y))
                    base = ti.Vector([0.57, 0.53, 0.45]) * (0.96 + 0.06 * self.hash2(tile))
                    base *= 0.72 + 0.28 * self.smooth(0.003, 0.01, edge)
                else:
                    base = ti.Vector([0.43, 0.44, 0.37]) * (0.94 + 0.04 * grain)
            elif material == 6:
                base, normal = self.floor_material(p, clock, amplitude)
            properties = self.surface_properties(p, material, self.materials[material])
            roughness, metallic, dielectric_f0 = properties.x, properties.y, properties.z
            if material == 5 and 3.6 < p.x < 6.2 and ti.abs(p.z) < 4.4:
                roughness = 0.48
            # Low amplitude, world-space texture relief preserves silhouettes.
            geometric_normal = normal
            if material == 0 or material == 1 or material == 4 or material == 5 or material == 9:
                tangent, bitangent = self.tangent_basis(normal)
                bump = 0.012
                if material == 1 or material == 9:
                    bump = 0.025
                normal = (normal + tangent * ti.sin(p.x * 51 + p.z * 11) * bump + bitangent * ti.sin(p.z * 67 + p.y * 23) * bump).normalized()
                roughness = ti.min(1.0, roughness + (grain - 0.5) * 0.05)
            view = -direction
            nv = ti.max(0.001, normal.dot(view))
            f0 = ti.Vector([dielectric_f0, dielectric_f0, dielectric_f0]) * (1.0 - metallic) + base * metallic
            fresnel = fresnel_schlick(nv, f0)
            reflected = (direction - 2.0 * direction.dot(normal) * normal).normalized()
            irradiance = self.environment.sample_map(self.environment.irradiance, normal, lighting)
            env_specular = self.environment.specular(reflected, roughness, lighting)
            lut = self.environment.integrated_brdf(nv, roughness)
            ao = 1.0
            if detail == 1 and material != 2 and self.light_controls[None].z > 0.0:
                ao = self.contact_ao(p, geometric_normal)
            indirect = base * (1.0 - metallic) * (1.0 - fresnel) * irradiance / math.pi
            indirect *= ao
            indirect += env_specular * (f0 * lut.x + lut.y) * (1.0 - roughness + roughness * ao)
            color = indirect * self.light_controls[None].w
            sun = self.sun_direction(lighting)
            shade = self.surface_sun_visibility(p, geometric_normal, material, lighting, detail)
            light_color = (ti.Vector([1.0, 0.94, 0.82]) * lighting + ti.Vector([1.0, 0.52, 0.24]) * (1.0 - lighting)) * 3.2
            light_color = self.surface_sun_color(p, material, lighting, light_color)
            color += ggx_brdf(base, roughness, metallic, normal, view, sun, dielectric_f0) * light_color * shade
            # Thin leaves get a modest transmitted sunlight approximation.
            if material == 2:
                color += base * light_color * ti.max(0.0, -normal.dot(sun)) * shade * 0.12
            # Warm bollards illuminate nearby ground at sunset.
            if lighting < 0.65 and material != 3:
                for lamp in range(2):
                    lamp_x = -3.95 + lamp * 7.85
                    delta = ti.Vector([lamp_x, 0.25, 3.9]) - p
                    distance2 = delta.dot(delta)
                    if distance2 < 4.0:
                        lamp_direction = delta.normalized()
                        unblocked = self.occlusion_distance(p + geometric_normal * 0.008, lamp_direction, ti.sqrt(distance2) - 0.12)
                        if unblocked >= ti.sqrt(distance2) - 0.12:
                            color += ggx_brdf(base, roughness, metallic, normal, view, lamp_direction, dielectric_f0) * ti.Vector([1.0, 0.42, 0.12]) * (1.0 - lighting) * 0.8 / (0.12 + distance2)
            if material == 3:
                color = base * (2.0 + 3.0 * (1.0 - lighting))
            mist = 1.0 - ti.exp(-t * 0.003)
            color = color * (1.0 - mist) + ti.Vector([0.67, 0.73, 0.70]) * mist
        return color, t

    @ti.kernel
    def filter_caustics(self):
        for x, y in self.caustic_map:
            value = 0.0
            for a, b in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                weight = (2.0 if a == 0 else 1.0) * (2.0 if b == 0 else 1.0) / 16.0
                value += self.caustic_raw[ti.min(127, ti.max(0, x + a)), ti.min(95, ti.max(0, y + b))] * weight
            self.caustic_map[x, y] = ti.min(8.0, value)

    @ti.func
    def caustic_density(self, p):
        u = ti.max(0.0, ti.min(127.0, (p.x + POOL_HALF_X) / (2 * POOL_HALF_X) * 128 - 0.5))
        v = ti.max(0.0, ti.min(95.0, (p.z + POOL_HALF_Z) / (2 * POOL_HALF_Z) * 96 - 0.5))
        ix, iy = ti.cast(u, ti.i32), ti.cast(v, ti.i32)
        fx, fy = u - ix, v - iy
        value = 0.0
        for a, b in ti.static(ti.ndrange(2, 2)):
            weight = (fx if a else 1.0 - fx) * (fy if b else 1.0 - fy)
            value += self.caustic_map[ti.min(127, ix + a), ti.min(95, iy + b)] * weight
        return value

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Taichi images are x-first, with the origin at the bottom left.
        pixels = self.image.to_numpy().transpose(1, 0, 2)[::-1]
        if not np.isfinite(pixels).all():
            raise RuntimeError("Renderer produced a non-finite image")
        Image.fromarray(np.round(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
        return path.resolve()
