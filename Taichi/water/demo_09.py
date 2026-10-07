"""Tank splash: three-dimensional dam-break flow striking a pillar.

A column of water inside a small tank collapses under gravity, rushes across
the floor, and slams into a square pillar. The fluid is integrated with
Position Based Fluids (Macklin and Müller 2013): poly6 density estimation,
spiky-gradient constraint solve with tensile correction, position projection
against the tank walls and the pillar, then vorticity confinement and XSPH
viscosity on the velocities. The surface is rendered in screen space:
particles splat front depth, thickness, and foam into per-pixel buffers, the
depth is smoothed, normals are reconstructed from depth gradients, and the
surface refracts the already-shaded tank interior.

Run: uv run python water/demo_09.py
Export: uv run python water/demo_09.py --headless --output output/tank.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .demo_06 import simulation_budget
    from .scene import CourtyardScene, OrbitCamera
else:
    from demo_06 import simulation_budget
    from scene import CourtyardScene, OrbitCamera

__all__ = ["TankFluid", "TankRenderer", "TankCamera", "TANK_PRESETS",
           "apply_tank_preset", "poly6_value_py", "spiky_gradient_py",
           "main", "parse_args", "TANK_X", "TANK_Z", "TANK_Y", "OBSTACLE_CENTER",
           "OBSTACLE_EXTENT", "SIM_DT"]

GRAVITY = 9.81
SIM_DT = 1.0 / 60.0
PBF_ITERS = 5
# Interior dimensions of the tank in metres.
TANK_X, TANK_Z, TANK_Y = 2.0, 1.0, 1.2
CEILING_Y = 1.4
OBSTACLE_CENTER = (1.15, 0.30, 0.5)
OBSTACLE_EXTENT = (0.12, 0.30, 0.12)

# PBF constants from the Taichi pbf2d example by Ye Kuang. The spiky gradient
# scales as 1/h^4, so these only work when the kernel radius is near 1. The
# simulation therefore runs in kernel units where h = 1.1 exactly, the same
# regime the constants were tuned in, and converts to metres at the renderer.
H_ = 1.1
LAMBDA_EPSILON = 100.0
CORR_DELTA_Q = 0.3
CORR_K = 0.001
NEIGHBOR_RADIUS = H_ * 1.05
POLY6_FACTOR = 315.0 / 64.0 / math.pi
SPIKY_GRAD_FACTOR = -45.0 / math.pi
MAX_PER_CELL = 32
MAX_NEIGHBORS = 64
CELL_SIZE = 1.2
# Per-substep gravity displacement in kernel units, matched to pbf2d
# (9.8 * (1/20)^2 ~= 0.0245) so the constraint solver stays in its stable
# regime regardless of the chosen particle spacing.
TARGET_G_DT2 = 0.0245


def poly6_value_py(distance, h):
    """Python mirror of the poly6 kernel for tests."""
    if 0.0 < distance < h:
        x = (h * h - distance * distance) / (h * h * h)
        return POLY6_FACTOR * x ** 3
    return 0.0


def spiky_gradient_py(offset, h):
    """Python mirror of the spiky gradient for tests."""
    offset = np.asarray(offset, dtype=float)
    length = float(np.linalg.norm(offset))
    if 1e-12 < length < h:
        x = (h - length) / (h * h * h)
        return SPIKY_GRAD_FACTOR * x * x * offset / length
    return np.zeros(3)


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = POLY6_FACTOR * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 1e-12 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = SPIKY_GRAD_FACTOR * x * x
        result = r * g_factor / r_len
    return result


@ti.data_oriented
class TankFluid:
    """PBF particle set in a rectangular tank with one square pillar.

    Positions are stored in kernel units (h = 1.1); length_unit gives the
    size of one kernel unit in metres.
    """

    def __init__(self, spacing=0.025,
                 column=(0.08, 0.02, 0.2, 0.68, 0.80, 0.8)):
        # column = (x0, y0, z0, x1, y1, z1) initial water block in metres.
        self.length_unit = spacing / 0.8 / H_
        su = lambda w: w / self.length_unit
        self.spacing = spacing / self.length_unit
        self.h = H_
        self.neighbor_radius = NEIGHBOR_RADIUS
        self.particle_radius = self.spacing * 0.5
        self.gravity = GRAVITY / self.length_unit
        self.velocity_clamp = 6.0 / self.length_unit
        self.foam_speed = 2.5 / self.length_unit
        self.foam_gain = self.length_unit / 2.5
        # Tank interior in kernel units: x in [0, tx], y in [0, ty],
        # z in [0, tz].
        self.tx = su(TANK_X)
        self.ty = su(TANK_Y)
        self.tz = su(TANK_Z)
        self.ceiling = su(CEILING_Y)
        nx = int((column[3] - column[0]) / spacing + 0.5)
        ny = int((column[4] - column[1]) / spacing + 0.5)
        nz = int((column[5] - column[2]) / spacing + 0.5)
        self.column = tuple(su(c) for c in column)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.count = nx * ny * nz
        # Uniform grid covers the confined domain plus a margin. The cell
        # must be at least neighbor_radius so a 3x3x3 scan finds every pair.
        self.grid_min = (-2.0, -2.0, -2.0)
        self.cell_recpr = 1.0 / CELL_SIZE
        self.grid_size = (int((self.tx + 2.0 - self.grid_min[0]) * self.cell_recpr) + 1,
                          int((self.ceiling + 2.0 - self.grid_min[1]) * self.cell_recpr) + 1,
                          int((self.tz + 2.0 - self.grid_min[2]) * self.cell_recpr) + 1)
        gx, gy, gz = self.grid_size
        self.positions = ti.Vector.field(3, ti.f32, shape=self.count)
        self.old_positions = ti.Vector.field(3, ti.f32, shape=self.count)
        self.velocities = ti.Vector.field(3, ti.f32, shape=self.count)
        self.lambdas = ti.field(ti.f32, shape=self.count)
        self.deltas = ti.Vector.field(3, ti.f32, shape=self.count)
        self.rho = ti.field(ti.f32, shape=self.count)
        self.rho_prev = ti.field(ti.f32, shape=self.count)
        self.foam = ti.field(ti.f32, shape=self.count)
        self.vorticity = ti.Vector.field(3, ti.f32, shape=self.count)
        self.grid_count = ti.field(ti.i32, shape=self.grid_size)
        self.grid_ids = ti.field(ti.i32, shape=(gx, gy, gz, MAX_PER_CELL))
        self.nb_count = ti.field(ti.i32, shape=self.count)
        self.nb_list = ti.field(ti.i32, shape=(self.count, MAX_NEIGHBORS))
        self.rho_rest = ti.field(ti.f32, shape=())
        self.step_counter = ti.field(ti.i32, shape=())
        self.obstacle_center = ti.Vector.field(3, ti.f32, shape=())
        self.obstacle_extent = ti.Vector.field(3, ti.f32, shape=())
        self.obstacle_center[None] = [su(c) for c in OBSTACLE_CENTER]
        self.obstacle_extent[None] = [su(c) for c in OBSTACLE_EXTENT]
        self.reset()

    def reset(self):
        self._init_particles()
        self._grid_clear()
        self._grid_fill()
        self._find_neighbors()
        self._solve_lambdas()
        self._measure_rho_rest()
        self.step_counter[None] = 0

    @ti.kernel
    def _init_particles(self):
        x0, y0, z0 = self.column[0], self.column[1], self.column[2]
        for i in range(self.count):
            kx = i % self.nx
            ky = (i // self.nx) % self.ny
            kz = i // (self.nx * self.ny)
            self.positions[i] = ti.Vector([x0 + kx * self.spacing,
                                           y0 + ky * self.spacing,
                                           z0 + kz * self.spacing])
            self.old_positions[i] = self.positions[i]
            self.velocities[i] = ti.Vector([0.0, 0.0, 0.0])
            self.foam[i] = 0.0

    @ti.func
    def _confine(self, p):
        r = self.particle_radius
        margin = 0.02
        # Anti-stick jitter is hashed from the position and step, so paused
        # renders and repeated runs stay reproducible.
        jitter = (self.hash01(ti.cast(p.x * 13.0 + p.y * 7.0, ti.i32)
                              + self.step_counter[None]) - 0.5) * 4e-3
        for axis in ti.static(range(3)):
            lo = r + margin
            hi_axis = 0.0
            if ti.static(axis == 0):
                hi_axis = self.tx - r - margin
            elif ti.static(axis == 1):
                hi_axis = self.ceiling - r - margin
            else:
                hi_axis = self.tz - r - margin
            if p[axis] < lo:
                p[axis] = lo + jitter
            elif p[axis] > hi_axis:
                p[axis] = hi_axis + jitter
        # The pillar is an axis-aligned box; push out along the face with the
        # least penetration.
        c = self.obstacle_center[None]
        e = self.obstacle_extent[None] + ti.Vector([r, r, r])
        q = ti.abs(p - c)
        inside = True
        for axis in ti.static(range(3)):
            if q[axis] >= e[axis]:
                inside = False
        if inside:
            pen = e - q
            axis = 0
            if pen.y < pen.x:
                axis = 1
            if pen.z < pen[axis]:
                axis = 2
            sign = 1.0
            if p[axis] < c[axis]:
                sign = -1.0
            p[axis] = c[axis] + sign * (e[axis] + 1e-3)
        return p

    @ti.func
    def hash01(self, i):
        v = ti.sin(ti.cast(i, ti.f32) * 12.9898) * 43758.5453
        return v - ti.floor(v)

    @ti.func
    def _cell_of(self, p):
        return ti.cast((p - ti.Vector(self.grid_min)) * self.cell_recpr, ti.i32)

    @ti.func
    def _cell_valid(self, cell):
        ok = True
        for axis in ti.static(range(3)):
            if cell[axis] < 0 or cell[axis] >= self.grid_size[axis]:
                ok = False
        return ok

    @ti.kernel
    def _predict(self, dt: ti.f32):
        for i in range(self.count):
            self.old_positions[i] = self.positions[i]
            velocity = self.velocities[i]
            velocity.y -= self.gravity * dt
            p = self.positions[i] + velocity * dt
            self.positions[i] = self._confine(p)

    @ti.kernel
    def _grid_clear(self):
        for cell in ti.grouped(self.grid_count):
            self.grid_count[cell] = 0

    @ti.kernel
    def _grid_fill(self):
        for i in range(self.count):
            cell = self._cell_of(self.positions[i])
            if self._cell_valid(cell):
                slot = ti.atomic_add(self.grid_count[cell], 1)
                if slot < MAX_PER_CELL:
                    self.grid_ids[cell, slot] = i

    @ti.kernel
    def _find_neighbors(self):
        for i in range(self.count):
            pos_i = self.positions[i]
            cell = self._cell_of(pos_i)
            nb = 0
            for a, b, c in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                check = cell + ti.Vector([a, b, c])
                if self._cell_valid(check):
                    for slot in range(self.grid_count[check]):
                        p_j = self.grid_ids[check, slot]
                        if p_j != i and (pos_i - self.positions[p_j]).norm() < self.neighbor_radius:
                            if nb < MAX_NEIGHBORS:
                                self.nb_list[i, nb] = p_j
                                nb += 1
            self.nb_count[i] = nb

    @ti.kernel
    def _solve_lambdas(self):
        for i in range(self.count):
            pos_i = self.positions[i]
            grad_i = ti.Vector([0.0, 0.0, 0.0])
            grad_sqr = 0.0
            density = 0.0
            for n in range(self.nb_count[i]):
                pos_ji = pos_i - self.positions[self.nb_list[i, n]]
                grad_j = spiky_gradient(pos_ji, self.h)
                grad_i += grad_j
                grad_sqr += grad_j.dot(grad_j)
                density += poly6_value(pos_ji.norm(), self.h)
            self.rho[i] = density
            # rho0 is the measured lattice rest density, so the constraint
            # vanishes at the undisplaced packing and the fluid keeps its
            # volume (the pbf2d example hardcodes 1 and accepts ~30 percent
            # squash; a held water column makes that loss visible).
            constraint = density / ti.max(self.rho_rest[None], 1e-6) - 1.0
            grad_sqr += grad_i.dot(grad_i)
            self.lambdas[i] = (-constraint) / (grad_sqr + LAMBDA_EPSILON)

    @ti.kernel
    def _solve_deltas(self):
        for i in range(self.count):
            pos_i = self.positions[i]
            lambda_i = self.lambdas[i]
            delta = ti.Vector([0.0, 0.0, 0.0])
            for n in range(self.nb_count[i]):
                p_j = self.nb_list[i, n]
                pos_ji = pos_i - self.positions[p_j]
                x = poly6_value(pos_ji.norm(), self.h) / poly6_value(CORR_DELTA_Q * self.h, self.h)
                x = x * x
                x = x * x
                scorr = (-CORR_K) * x
                delta += (lambda_i + self.lambdas[p_j] + scorr) * spiky_gradient(pos_ji, self.h)
            self.deltas[i] = delta

    @ti.kernel
    def _apply_deltas(self):
        for i in range(self.count):
            self.positions[i] += self.deltas[i]

    @ti.kernel
    def _confine_and_update_velocities(self, dt: ti.f32):
        for i in range(self.count):
            p = self._confine(self.positions[i])
            self.positions[i] = p
            velocity = (p - self.old_positions[i]) / dt
            speed = velocity.norm()
            if speed > self.velocity_clamp:
                velocity *= self.velocity_clamp / speed
            # The clamp above turns an infinite speed into NaN (inf * 0).
            if not (velocity.x == velocity.x and velocity.y == velocity.y
                    and velocity.z == velocity.z):
                velocity = ti.Vector([0.0, 0.0, 0.0])
                speed = 0.0
            # Contact with the floor bleeds tangential speed.
            if p.y < self.particle_radius * 1.6 and velocity.y < 0.0:
                velocity.x *= 0.98
                velocity.z *= 0.98
            self.velocities[i] = velocity
            # Foam marks recent cavitation (density dropping below the
            # particle's own previous value) plus fast spray. A plain density
            # threshold would paint the whole free surface permanently, since
            # surface particles always read sparse against the interior mean.
            rho_now = self.rho[i]
            rho_rest = ti.max(self.rho_rest[None], 1e-6)
            cavit = ti.max(0.0, self.rho_prev[i] - rho_now) / rho_rest
            self.rho_prev[i] = rho_now
            speed_foam = ti.max(0.0, speed - self.foam_speed) * self.foam_gain
            self.foam[i] = ti.min(1.0, cavit * 1.2 + speed_foam)

    @ti.kernel
    def _vorticity_pass(self):
        for i in range(self.count):
            omega = ti.Vector([0.0, 0.0, 0.0])
            velocity_i = self.velocities[i]
            pos_i = self.positions[i]
            for n in range(self.nb_count[i]):
                p_j = self.nb_list[i, n]
                omega += (self.velocities[p_j] - velocity_i).cross(
                    spiky_gradient(pos_i - self.positions[p_j], self.h))
            self.vorticity[i] = omega

    @ti.kernel
    def _vorticity_xsph(self, dt: ti.f32, vorticity_eps: ti.f32, xsph_c: ti.f32):
        for i in range(self.count):
            pos_i = self.positions[i]
            omega_i = self.vorticity[i]
            omega_len = omega_i.norm()
            gradient = ti.Vector([0.0, 0.0, 0.0])
            for n in range(self.nb_count[i]):
                p_j = self.nb_list[i, n]
                pos_ji = pos_i - self.positions[p_j]
                dist2 = ti.max(1e-8, pos_ji.dot(pos_ji))
                gradient += (self.vorticity[p_j].norm() - omega_len) * pos_ji / dist2
            force = ti.Vector([0.0, 0.0, 0.0])
            if gradient.norm() > 1e-8 and omega_len > 1e-8:
                normal = gradient.normalized()
                force = vorticity_eps * normal.cross(omega_i)
            velocity = self.velocities[i] + force * dt
            smoothing = ti.Vector([0.0, 0.0, 0.0])
            for n in range(self.nb_count[i]):
                p_j = self.nb_list[i, n]
                pos_ji = pos_i - self.positions[p_j]
                smoothing += (self.velocities[p_j] - velocity) * poly6_value(pos_ji.norm(), self.h)
            velocity += xsph_c * smoothing
            self.velocities[i] = velocity

    def substep(self, dt):
        self.step_counter[None] += 1
        self._predict(dt)
        self._grid_clear()
        self._grid_fill()
        self._find_neighbors()
        for _ in range(PBF_ITERS):
            self._solve_lambdas()
            self._solve_deltas()
            self._apply_deltas()
        self._confine_and_update_velocities(dt)
        self._vorticity_pass()
        self._vorticity_xsph(dt, self.vorticity_eps, self.xsph_c)

    # Default viscosity and vorticity strengths in kernel units; sliders in
    # the UI write to these attributes between frames.
    vorticity_eps = 0.08
    xsph_c = 0.08

    def substep_dt(self):
        """Substep size matched to the pbf2d stable regime."""
        return math.sqrt(TARGET_G_DT2 / self.gravity)

    def advance(self, seconds):
        steps = max(1, int(round(seconds / self.substep_dt())))
        dt = seconds / steps
        for _ in range(steps):
            self.substep(dt)

    @ti.kernel
    def _measure_rho_rest(self):
        total = 0.0
        weight = 0.0
        for i in range(self.count):
            self.rho_prev[i] = self.rho[i]
            if self.nb_count[i] > 4:
                total += self.rho[i]
                weight += 1.0
        if weight > 0.0:
            self.rho_rest[None] = total / weight


TANK_PRESETS = {
    # name: (target, (yaw, pitch, distance), fov)
    "overview": ((1.00, 0.22, 0.50), (0.80, 0.46, 3.2), 50.0),
    "impact": ((0.95, 0.24, 0.50), (-1.30, 0.20, 2.3), 48.0),
    "top": ((1.00, 0.15, 0.50), (0.02, 1.44, 2.7), 50.0),
    "side": ((1.00, 0.20, 0.50), (0.03, 0.16, 2.5), 45.0),
}

IOR = 1.333
# Base extinction per metre of water at clarity 1; the clarity slider
# scales it. Chosen so a 0.4 m pool reads turquoise without going black.
EXTINCTION = 5.0
# Depth-buffer sentinel; anything closer than this is fluid.
DEPTH_FAR = 1e9


def apply_tank_preset(camera, name):
    target, (yaw, pitch, distance), fov = TANK_PRESETS[name]
    camera.target = np.array(target, dtype=float)
    camera.yaw, camera.pitch, camera.distance = yaw, pitch, distance
    camera.fov = fov


@ti.data_oriented
class TankRenderer(CourtyardScene):
    """Small tank scene with screen-space reconstruction of the PBF surface.

    The static tank is shaded once into a background buffer; particles then
    splat front depth, thickness, and foam into screen buffers, the depth is
    smoothed, normals come from depth gradients, and the front surface mixes
    sky reflection with refraction toward the background at the refracted
    floor point. Fast droplets composite additively as spray.
    """

    def __init__(self, width, height, samples=4, spacing=0.025):
        super().__init__(width, height)
        self.samples = samples
        self.fluid = TankFluid(spacing=spacing)
        self.lu = self.fluid.length_unit
        # x: foam display gain, y: particle draw radius, z: spray gain,
        # w: thickness edge fade.
        self.fluid_controls = ti.Vector.field(4, ti.f32, shape=())
        self.fluid_controls[None] = [0.35, 2.4, 1.0, 1.0]
        self.bg_rgb = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.bg_depth = ti.field(ti.f32, shape=(width, height))
        self.water_depth = ti.field(ti.f32, shape=(width, height))
        self.water_depth_sm = ti.field(ti.f32, shape=(width, height))
        self.thickness = ti.field(ti.f32, shape=(width, height))
        self.foam_n = ti.field(ti.f32, shape=(width, height))
        self.foam_d = ti.field(ti.f32, shape=(width, height))
        self.spray_accum = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.spray_weight = ti.field(ti.f32, shape=(width, height))

    def build_static(self):
        """A shallow rectangular tank with one square pillar."""
        wall = (0.60, 0.58, 0.52)
        coping = (0.68, 0.65, 0.58)
        thickness, height = 0.06, 0.40

        def box(center, extent, color, material=0, bevel=0.012):
            i = self.boxes
            self.box_center[i], self.box_extent[i] = center, extent
            self.box_color[i], self.box_material[i] = color, material
            self.box_bevel[i] = min(bevel, min(extent) * 0.8)
            self.boxes += 1

        # Four tank walls, reaching above the resting water level.
        box((0.0, height * 0.5, 0.5), (thickness * 0.5, height * 0.5, 0.5 + thickness), wall)
        box((TANK_X, height * 0.5, 0.5), (thickness * 0.5, height * 0.5, 0.5 + thickness), wall)
        box((TANK_X * 0.5, height * 0.5, 0.0), (TANK_X * 0.5, height * 0.5, thickness * 0.5), wall)
        box((TANK_X * 0.5, height * 0.5, TANK_Z), (TANK_X * 0.5, height * 0.5, thickness * 0.5), wall)
        # Coping rims make the wall tops read as built edges.
        box((0.0, height + 0.015, 0.5), (0.055, 0.015, 0.5 + 0.055), coping, 0, 0.008)
        box((TANK_X, height + 0.015, 0.5), (0.055, 0.015, 0.5 + 0.055), coping, 0, 0.008)
        box((TANK_X * 0.5, height + 0.015, 0.0), (TANK_X * 0.5 + 0.055, 0.015, 0.055), coping, 0, 0.008)
        box((TANK_X * 0.5, height + 0.015, TANK_Z), (TANK_X * 0.5 + 0.055, 0.015, 0.055), coping, 0, 0.008)
        # The pillar the water slams into.
        box(OBSTACLE_CENTER, OBSTACLE_EXTENT, (0.30, 0.32, 0.35), 4, 0.02)

    @ti.func
    def geometry(self, origin, direction):
        closest, material = 1e6, -1
        normal = ti.Vector([0.0, 1.0, 0.0])
        color = ti.Vector([0.5, 0.5, 0.5])
        if ti.abs(direction.y) > 1e-6:
            t = -origin.y / direction.y
            if t > 0.001 and t < closest:
                closest, material = t, 6
                normal = ti.Vector([0.0, -ti.math.sign(direction.y), 0.0])
        for i in range(self.boxes):
            t, n = self.rounded_box_hit(origin, direction, self.box_center[i],
                                        self.box_extent[i], self.box_bevel[i])
            if t < closest:
                closest, normal = t, n
                color, material = self.box_color[i], self.box_material[i]
        return closest, normal, color, material

    @ti.func
    def floor_material(self, p, clock, amplitude):
        # Light ceramic tiles inside the tank, dark concrete outside; both
        # carry a faint procedural relief so grazing views do not go flat.
        inside = (0.0 <= p.x <= TANK_X) and (0.0 <= p.z <= TANK_Z)
        grain = self.hash2(ti.floor(ti.Vector([p.x * 43.0, p.z * 43.0])))
        color = ti.Vector([0.60, 0.65, 0.68]) * (0.90 + 0.10 * grain)
        normal = ti.Vector([0.0, 1.0, 0.0])
        if inside:
            uv = ti.Vector([p.x, p.z]) * 10.0
            cell = ti.floor(uv)
            f = uv - cell
            edge = ti.min(ti.min(f.x, 1.0 - f.x), ti.min(f.y, 1.0 - f.y))
            grout = self.smooth(0.02, 0.06, edge)
            tile = 0.88 + 0.12 * self.hash2(cell + 3.7)
            color = ti.Vector([0.55, 0.63, 0.67]) * tile
            color = color * (1.0 - grout) + ti.Vector([0.22, 0.25, 0.27]) * grout
        else:
            color = ti.Vector([0.30, 0.29, 0.27]) * (0.88 + 0.12 * grain)
        return color, normal

    @ti.func
    def pixel_ray(self, x, y, forward, right, up, scale):
        sx = (2.0 * (x + 0.5) / self.width - 1.0) * self.width / self.height * scale
        sy = (2.0 * (y + 0.5) / self.height - 1.0) * scale
        return (forward + right * sx + up * sy).normalized()

    @ti.kernel
    def _render_bg(self, eye: ti.types.vector(3, ti.f32),
                   forward: ti.types.vector(3, ti.f32),
                   right: ti.types.vector(3, ti.f32),
                   up: ti.types.vector(3, ti.f32),
                   scale: ti.f32, clock: ti.f32, lighting: ti.f32):
        for x, y in self.image:
            color = ti.Vector([0.0, 0.0, 0.0])
            depth = 1e6
            for sample in range(self.samples):
                offset = ti.Vector([0.5, 0.5])
                if self.samples == 4:
                    offset = ti.Vector([0.25 + 0.5 * (sample % 2),
                                        0.25 + 0.5 * (sample // 2)])
                direction = self.pixel_ray(x + offset.x - 0.5, y + offset.y - 0.5,
                                           forward, right, up, scale)
                t, normal, base, material = self.geometry(eye, direction)
                sample_color, _ = self.shade_surface(eye, direction, t, normal,
                                                     base, material, clock,
                                                     lighting, 0.0, 1.0)
                color += sample_color
                depth = ti.min(depth, t)
            self.bg_rgb[x, y] = color / self.samples
            self.bg_depth[x, y] = depth

    @ti.kernel
    def _splat_particles(self, eye: ti.types.vector(3, ti.f32),
                         forward: ti.types.vector(3, ti.f32),
                         right: ti.types.vector(3, ti.f32),
                         up: ti.types.vector(3, ti.f32),
                         scale: ti.f32, lighting: ti.f32):
        draw_radius = self.fluid_controls[None].y
        spray_gain = self.fluid_controls[None].z
        sphere_radius = self.fluid.particle_radius * self.lu * draw_radius
        sun = self.sun_direction(lighting)
        for i in range(self.fluid.count):
            position = self.fluid.positions[i] * self.lu
            rel = position - eye
            cz = rel.dot(forward)
            if cz > 0.03:
                px = 0.5 * self.width + rel.dot(right) / cz * 0.5 * self.height / scale
                py = 0.5 * self.height + rel.dot(up) / cz * 0.5 * self.height / scale
                radius = sphere_radius * 0.5 * self.height / (scale * cz)
                reach = ti.cast(ti.ceil(ti.max(1.0, radius * 1.6)), ti.i32)
                center_x = ti.cast(ti.floor(px), ti.i32)
                center_y = ti.cast(ti.floor(py), ti.i32)
                # World size of one pixel at the particle depth; the chord a
                # sphere cuts through each pixel is the thickness splat.
                pixel_world = cz * 2.0 * scale / self.height
                speed = self.fluid.velocities[i].norm() * self.lu
                foam = self.fluid.foam[i]
                spray = ti.max(0.0, speed - 2.6) * 0.8 + ti.max(0.0, foam - 0.70) * 1.4
                spray = ti.min(1.0, spray * spray_gain)
                view = -rel.normalized()
                for gx, gy in ti.ndrange((-reach, reach + 1), (-reach, reach + 1)):
                    x, y = center_x + gx, center_y + gy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        # The tank wall or pillar may sit between the camera
                        # and this particle; occluded pixels keep the solid.
                        if cz >= self.bg_depth[x, y]:
                            continue
                        dxw = (x + 0.5 - px) * pixel_world
                        dyw = (y + 0.5 - py) * pixel_world
                        d2 = dxw * dxw + dyw * dyw
                        if d2 < sphere_radius * sphere_radius:
                            chord = 2.0 * ti.sqrt(sphere_radius * sphere_radius - d2)
                            coverage = chord / (2.0 * sphere_radius)
                            ti.atomic_min(self.water_depth[x, y], cz)
                            ti.atomic_add(self.thickness[x, y], chord)
                            ti.atomic_add(self.foam_n[x, y], foam * coverage)
                            ti.atomic_add(self.foam_d[x, y], coverage)
                            if spray > 0.0:
                                drop_normal = (view * 0.75 + right * (dxw / sphere_radius) * 0.25
                                               + up * (dyw / sphere_radius) * 0.25).normalized()
                                reflected = (view - 2.0 * view.dot(drop_normal) * drop_normal).normalized()
                                glint = ti.pow(ti.max(0.0, reflected.dot(sun)), 240.0) * 4.0
                                tint = self.environment.sample_map(self.environment.environment,
                                                                   reflected, lighting)
                                drop_color = (tint * 0.5 + ti.Vector([0.65, 0.74, 0.80]) * 0.5
                                              + ti.Vector([1.0, 0.95, 0.85]) * glint)
                                alpha = coverage * spray * 0.45
                                ti.atomic_add(self.spray_accum[x, y], drop_color * alpha)
                                ti.atomic_add(self.spray_weight[x, y], alpha)

    @ti.kernel
    def _smooth_depth(self):
        for x, y in self.image:
            center = self.water_depth[x, y]
            if center < DEPTH_FAR:
                total = 0.0
                weight = 0.0
                for a, b in ti.ndrange((-1, 2), (-1, 2)):
                    u = ti.min(self.width - 1, ti.max(0, x + a))
                    v = ti.min(self.height - 1, ti.max(0, y + b))
                    depth = self.water_depth[u, v]
                    if depth < DEPTH_FAR:
                        total += depth
                        weight += 1.0
                self.water_depth_sm[x, y] = total / ti.max(1.0, weight)
            else:
                self.water_depth_sm[x, y] = DEPTH_FAR

    @ti.kernel
    def _smooth_depth_pass2(self):
        for x, y in self.image:
            center = self.water_depth_sm[x, y]
            if center < DEPTH_FAR:
                total = 0.0
                weight = 0.0
                for a, b in ti.ndrange((-1, 2), (-1, 2)):
                    u = ti.min(self.width - 1, ti.max(0, x + a))
                    v = ti.min(self.height - 1, ti.max(0, y + b))
                    depth = self.water_depth_sm[u, v]
                    if depth < DEPTH_FAR:
                        total += depth
                        weight += 1.0
                self.water_depth_sm[x, y] = total / ti.max(1.0, weight)

    @ti.kernel
    def _shade_water(self, eye: ti.types.vector(3, ti.f32),
                     forward: ti.types.vector(3, ti.f32),
                     right: ti.types.vector(3, ti.f32),
                     up: ti.types.vector(3, ti.f32),
                     scale: ti.f32, lighting: ti.f32, clarity: ti.f32):
        foam_gain = self.fluid_controls[None].x
        edge_fade = self.fluid_controls[None].w
        sun = self.sun_direction(lighting)
        eta = 1.0 / IOR
        for x, y in self.image:
            background = self.bg_rgb[x, y]
            depth = self.water_depth_sm[x, y]
            color = background
            if depth < DEPTH_FAR:
                direction = self.pixel_ray(x, y, forward, right, up, scale)
                p = eye + direction * depth
                # World positions of the four neighbours, each on its own
                # pixel ray; invalid neighbours reuse the centre depth so
                # silhouette pixels keep a flat, camera-facing normal.
                d_r = self.water_depth_sm[ti.min(self.width - 1, x + 1), y]
                d_l = self.water_depth_sm[ti.max(0, x - 1), y]
                d_u = self.water_depth_sm[x, ti.min(self.height - 1, y + 1)]
                d_d = self.water_depth_sm[x, ti.max(0, y - 1)]
                if d_r > DEPTH_FAR:
                    d_r = depth
                if d_l > DEPTH_FAR:
                    d_l = depth
                if d_u > DEPTH_FAR:
                    d_u = depth
                if d_d > DEPTH_FAR:
                    d_d = depth
                p_r = eye + self.pixel_ray(x + 1, y, forward, right, up, scale) * d_r
                p_l = eye + self.pixel_ray(x - 1, y, forward, right, up, scale) * d_l
                p_u = eye + self.pixel_ray(x, y + 1, forward, right, up, scale) * d_u
                p_d = eye + self.pixel_ray(x, y - 1, forward, right, up, scale) * d_d
                normal = (p_r - p_l).cross(p_u - p_d).normalized()
                if normal.dot(direction) > 0.0:
                    normal = -normal
                view = -direction
                fresnel = 0.02 + 0.98 * ti.pow(1.0 - ti.max(0.0, normal.dot(view)), 5.0)
                reflected = (direction - 2.0 * direction.dot(normal) * normal).normalized()
                reflection = self.sky(reflected, lighting)
                reflection += ti.Vector([1.0, 0.94, 0.82]) * ti.pow(
                    ti.max(0.0, reflected.dot(sun)), 320.0) * 3.0 * lighting
                d = normal.dot(direction)
                k = 1.0 - eta * eta * (1.0 - d * d)
                refracted = reflected
                if k > 0.0:
                    refracted = (eta * direction - (eta * d + ti.sqrt(k)) * normal).normalized()
                t_q, _, _, _ = self.geometry(p + normal * 0.003, refracted)
                if t_q > 1e5:
                    t_q = 0.1
                q = p + refracted * t_q
                rel = q - eye
                cq = rel.dot(forward)
                refraction = background
                if cq > 0.01:
                    qx = ti.cast(0.5 * self.width + rel.dot(right) / cq * 0.5 * self.height / scale, ti.i32)
                    qy = ti.cast(0.5 * self.height + rel.dot(up) / cq * 0.5 * self.height / scale, ti.i32)
                    if 0 <= qx < self.width and 0 <= qy < self.height:
                        refraction = self.bg_rgb[qx, qy]
                absorb = ti.exp(-t_q * clarity * EXTINCTION)
                # Thin coverage is airborne spray and sheet edges: it must stay
                # transparent instead of reading as a dark droplet of bulk water.
                bulk = self.smooth(0.0, 0.05, self.thickness[x, y])
                absorb = 1.0 - (1.0 - absorb) * bulk
                deep = ti.Vector([0.03, 0.10, 0.12]) * (0.35 + 0.65 * lighting)
                refraction = refraction * absorb + deep * (1.0 - absorb) * bulk
                foam = ti.min(1.0, self.foam_n[x, y] / ti.max(1e-6, self.foam_d[x, y]) * foam_gain)
                foam_color = ti.Vector([0.82, 0.88, 0.92]) * (0.55 + 0.45 * lighting)
                water = refraction * (1.0 - fresnel) + reflection * fresnel
                water = water * (1.0 - foam) + foam_color * foam
                alpha = self.smooth(0.05, 0.05 + 1.2 * edge_fade, self.foam_d[x, y])
                color = background * (1.0 - alpha) + water * alpha
            self.post.hdr[x, y] = color

    @ti.kernel
    def _composite_spray(self):
        for x, y in self.image:
            weight = self.spray_weight[x, y]
            if weight > 1e-6:
                alpha = 1.0 - ti.exp(-weight)
                spray_color = self.spray_accum[x, y] / weight
                self.post.hdr[x, y] = (self.post.hdr[x, y] * (1.0 - alpha)
                                       + spray_color * alpha)

    def prepare(self, camera, clock, clarity=1.4, lighting=1.0, exposure=1.05):
        self.draw(camera, clock, clarity, lighting, exposure)

    def draw(self, camera, clock, clarity=1.4, lighting=1.0, exposure=1.05):
        eye, forward, right, up = (ti.Vector(v, dt=ti.f32) for v in camera.basis())
        scale = math.tan(math.radians(camera.fov * 0.5))
        self.water_depth.fill(DEPTH_FAR)
        self.thickness.fill(0.0)
        self.foam_n.fill(0.0)
        self.foam_d.fill(0.0)
        self.spray_accum.fill(0.0)
        self.spray_weight.fill(0.0)
        self._render_bg(eye, forward, right, up, scale, clock, lighting)
        self._splat_particles(eye, forward, right, up, scale, lighting)
        self._smooth_depth()
        self._smooth_depth_pass2()
        self._shade_water(eye, forward, right, up, scale, lighting, clarity)
        self._composite_spray()
        bloom, threshold, vignette = self.post_controls[None]
        self.post.apply(exposure, bloom, threshold, vignette)


class TankCamera(OrbitCamera):
    """Orbit camera framed on the small tank."""

    def zoom(self, amount):
        self.distance = float(np.clip(self.distance * math.exp(amount), 0.7, 8.0))

    def pan(self, dx, dy):
        _, _, right, _ = self.basis()
        horizontal_forward = np.array([-right[2], 0.0, right[0]])
        self.target -= (right * dx + horizontal_forward * dy) * self.distance
        self.target[0] = float(np.clip(self.target[0], 0.0, TANK_X))
        self.target[1] = float(np.clip(self.target[1], 0.0, 0.8))
        self.target[2] = float(np.clip(self.target[2], 0.0, TANK_Z))

    def orbit(self, dx, dy):
        self.yaw -= dx * 3.5
        self.pitch = float(np.clip(self.pitch - dy * 3.0, 0.05, 1.5))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 09: dam-break tank splash")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=1,
                        help="Spatial samples per pixel (default: 1)")
    parser.add_argument("--spacing", type=float, default=0.025,
                        help="Initial particle spacing in metres")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="day")
    parser.add_argument("--shadow-samples", type=int, choices=(1, 4), default=1)
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--preset", choices=tuple(TANK_PRESETS), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_09.png"))
    parser.add_argument("--time", type=float, default=0.0,
                        help="Simulation warmup in seconds before the first frame")
    parser.add_argument("--frames", type=int, default=1,
                        help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0,
                        help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0:
        parser.error("width/height must be >= 32, frames >= 1, window-frames >= 0")
    if not math.isfinite(args.time):
        parser.error("time must be finite")
    if not math.isfinite(args.spacing) or not 0.015 <= args.spacing <= 0.08:
        parser.error("spacing must be finite and between 0.015 and 0.08")
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = TankRenderer(args.width, args.height, args.samples, args.spacing)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples,
                                     0.0 if args.no_ao else 0.65, 1.0]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = TankCamera()
    apply_tank_preset(camera, args.preset)
    speed = 1.0
    lighting = 0.0 if args.lighting == "sunset" else 1.0
    clarity, exposure = 1.4, 1.05
    clock = 0.0
    sim_accumulator = 0.0

    def advance(frames):
        nonlocal clock
        for _ in range(int(frames)):
            renderer.fluid.advance(SIM_DT)
            clock += SIM_DT

    advance(int(max(0.0, args.time) / SIM_DT))

    if args.headless:
        start = time.perf_counter()
        timings = []
        for _ in range(args.frames):
            frame_start = time.perf_counter()
            advance(1)
            renderer.draw(camera, clock, clarity, lighting, exposure)
            ti.sync()
            timings.append(time.perf_counter() - frame_start)
        print(f"Saved {renderer.save(args.output)} ({args.width}x{args.height})")
        print(f"{args.frames} frame(s), including compilation: {time.perf_counter() - start:.2f}s")
        if len(timings) > 1:
            print(f"Post-compilation render mean: {np.mean(timings[1:]) * 1000:.1f} ms/frame")
        return

    # Cold JIT compilation must finish before creating a native window: GGUI
    # polls OS events in show(), which cannot run while draw() is compiling.
    print("[Startup] Preparing first frame; cold compilation can take a minute or more. Window opens when ready.", flush=True)
    warmup_start = time.perf_counter()
    renderer.prepare(camera, clock, clarity, lighting, exposure)
    ti.sync()
    print(f"[Startup] First frame ready in {time.perf_counter() - warmup_start:.2f}s. Opening window...", flush=True)
    window = ti.ui.Window("09 / Tank splash", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    sim_accumulator, frame_ms, window_frames = 0.0, 0.0, 1
    preset_keys = {"1": "overview", "2": "impact", "3": "top", "4": "side"}
    print("Drag LMB orbit | RMB pan | W/S zoom | 1-4 views | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
    while window.running and (not args.window_frames or window_frames < args.window_frames):
        now = time.perf_counter()
        dt = min(now - previous_time, 0.25)
        previous_time = now
        step, save_frame = False, False
        for event in window.get_events(ti.ui.PRESS):
            key = event.key
            if key == ti.ui.ESCAPE:
                window.running = False
            elif key == " ":
                paused = not paused
            elif key == "n":
                step = True
            elif key == "a":
                auto_orbit = not auto_orbit
            elif key == "h":
                show_panel = not show_panel
            elif key in preset_keys:
                apply_tank_preset(camera, preset_keys[key])
                auto_orbit = False
            elif key == "r":
                renderer.fluid.reset()
                clock, sim_accumulator = 0.0, 0.0
                paused, auto_orbit = False, False
            elif key == "p":
                save_frame = True
        if not window.running:
            break
        mouse = np.array(window.get_cursor_pos())
        in_panel = show_panel and ((mouse[0] < 0.33 and mouse[1] > 0.15) or (mouse[0] > 0.70 and mouse[1] > 0.04))
        if previous_mouse is not None and not in_panel:
            if window.is_pressed(ti.ui.LMB):
                camera.orbit(*(mouse - previous_mouse))
                auto_orbit = False
            elif window.is_pressed(ti.ui.RMB):
                camera.pan(*(mouse - previous_mouse))
                auto_orbit = False
        previous_mouse = mouse
        if window.is_pressed("w", ti.ui.UP):
            camera.zoom(-dt)
        if window.is_pressed("s", ti.ui.DOWN):
            camera.zoom(dt)
        if auto_orbit:
            camera.yaw += dt * 0.12
        if show_panel:
            with gui.sub_window("LIGHT / MATERIALS", 0.72, 0.025, 0.26, 0.24):
                controls = renderer.light_controls[None]
                controls[1] = 4 if gui.checkbox("Soft sun shadows", controls[1] == 4) else 1
                controls[0] = math.radians(gui.slider_float("Sun radius (deg)", math.degrees(controls[0]), 0.0, 3.0))
                controls[2] = gui.slider_float("Contact AO", controls[2], 0.0, 1.0)
                controls[3] = gui.slider_float("Environment", controls[3], 0.0, 2.0)
                renderer.light_controls[None] = controls
            with gui.sub_window("WATER / FINISH", 0.72, 0.30, 0.26, 0.42):
                fluid_ui = renderer.fluid_controls[None]
                fluid_ui[0] = gui.slider_float("Foam gain", fluid_ui[0], 0.0, 2.0)
                fluid_ui[1] = gui.slider_float("Particle radius", fluid_ui[1], 0.8, 3.0)
                fluid_ui[2] = gui.slider_float("Spray gain", fluid_ui[2], 0.0, 2.0)
                fluid_ui[3] = gui.slider_float("Edge fade", fluid_ui[3], 0.4, 2.0)
                renderer.fluid_controls[None] = fluid_ui
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
            with gui.sub_window("TANK / 09", 0.02, 0.025, 0.30, 0.70):
                gui.text("A water column breaks and strikes a pillar")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                if gui.button("Single step [N]"):
                    step = True
                if gui.button("Reset fluid [R]"):
                    renderer.fluid.reset()
                    clock, sim_accumulator = 0.0, 0.0
                speed = gui.slider_float("Time speed", speed, 0.0, 2.0)
                clarity = gui.slider_float("Water clarity", clarity, 0.35, 3.0)
                lighting = gui.slider_float("Day / sunset", lighting, 0.0, 1.0)
                exposure = gui.slider_float("Exposure", exposure, 0.5, 1.8)
                renderer.fluid.xsph_c = gui.slider_float("Viscosity", renderer.fluid.xsph_c, 0.0, 0.3)
                renderer.fluid.vorticity_eps = gui.slider_float("Vorticity", renderer.fluid.vorticity_eps, 0.0, 0.3)
                renderer.samples = 4 if gui.checkbox("4x spatial AA", renderer.samples == 4) else 1
                if gui.button("Impact view [2]"):
                    apply_tank_preset(camera, "impact")
                    auto_orbit = False
                if gui.button("Overview [1]"):
                    apply_tank_preset(camera, "overview")
                    auto_orbit = False
                gui.text("W/S or arrows: zoom")
                gui.text("H: hide panel / P: save / Esc: exit")
        if paused:
            if step:
                advance(1)
        else:
            frames, sim_accumulator = simulation_budget(sim_accumulator, dt, speed)
            advance(frames)
        renderer.draw(camera, clock, clarity, lighting, exposure)
        if save_frame:
            print(f"Saved {renderer.save(args.output)}")
        canvas.set_image(renderer.image)
        window.show()
        window_frames += 1
        if args.window_frames and window_frames >= args.window_frames:
            window.running = False
        frame_ms = 0.9 * frame_ms + 0.1 * (time.perf_counter() - now) * 1000


if __name__ == "__main__":
    main()
