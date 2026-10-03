"""Harbor floats: buoys and a small boat riding Gerstner swell.

Rigid bodies float on the demo 03 ocean through five-sample buoyancy: each
body samples the wave height and vertical water velocity at its center and
four waterline corners, integrates heave from the average excess buoyancy,
and integrates pitch/roll from the corner torque. The boat can be dragged
with the mouse or set to cruise a slow circle; hull motion injects velocity
into a 2D wave-equation wake field that feeds back into the rendered surface
through the extra height/slope hooks. Contact foam rings the hulls where
they pierce the surface.

Run: uv run python water/demo_04.py
Export: uv run python water/demo_04.py --headless --output output/harbor.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .demo_02 import simulation_budget
    from .demo_03 import (OCEAN_PRESETS, SEABED, SunsetOceanRenderer, apply_preset)
    from .scene import OrbitCamera
else:
    from demo_02 import simulation_budget
    from demo_03 import (OCEAN_PRESETS, SEABED, SunsetOceanRenderer, apply_preset)
    from scene import OrbitCamera

__all__ = ["WakeField", "HarborRenderer", "main", "parse_args",
           "MAX_BODIES", "PARTS_PER_BODY", "WAKE_N", "WAKE_CELL", "WAKE_DT",
           "GRAVITY", "BOAT_INDEX"]

GRAVITY = 9.81
BOAT_INDEX = 0
MAX_BODIES = 4
PARTS_PER_BODY = 4
# Wake field: 32 m square domain centered on the origin, open (absorbing)
# boundaries so the swell leaving the domain does not reflect back.
WAKE_N = 256
WAKE_CELL = 0.125
WAKE_HALF = WAKE_N * WAKE_CELL * 0.5
WAKE_DT = 1.0 / 120.0
WAKE_ABSORB = 12
WAKE_SPEED = 3.0
WAKE_DAMPING = 0.995
BOAT_CRUISE_RADIUS = 6.0
BOAT_CRUISE_RATE = 0.15
BOAT_MAX_SPEED = 2.2
DRAG_LIMIT = 14.0
# Buoyancy: per-sample excess acceleration 160*depth balances gravity at a
# ~6 cm draft; the vertical relative-velocity term damps heave.
BUOYANCY_K = 160.0
BUOYANCY_D = 6.0
TORQUE_GAIN = 1.5
ANGULAR_DAMP = 3.0
MAX_TILT = 0.45
# Collisions: sphere proxies per part, horizontal-only response so the
# buoyancy system keeps owning the vertical motion. The dragged boat is
# treated as infinite mass and shoves floats aside without slowing down.
RESTITUTION = 0.15
COLLISION_ITERATIONS = 4
YAW_IMPULSE_GAIN = 2.0
MAX_YAW_RATE = 1.5


@ti.func
def rot_x(a):
    c, s = ti.cos(a), ti.sin(a)
    return ti.Matrix([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@ti.func
def rot_y(a):
    c, s = ti.cos(a), ti.sin(a)
    return ti.Matrix([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@ti.func
def rot_z(a):
    c, s = ti.cos(a), ti.sin(a)
    return ti.Matrix([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@ti.data_oriented
class WakeField:
    """2D wave equation on a regular grid with absorbing boundaries.

    Same leapfrog scheme as the demo 02 pool, but the clamped-edge
    reflection is swallowed by a 12-cell absorption band that scales the
    new height toward zero at the domain edge, so wakes leave quietly.
    """

    def __init__(self):
        self.h = ti.field(ti.f32, shape=(WAKE_N, WAKE_N))
        self.h_prev = ti.field(ti.f32, shape=(WAKE_N, WAKE_N))
        self.h_next = ti.field(ti.f32, shape=(WAKE_N, WAKE_N))
        self.steps = 0

    @ti.kernel
    def clear(self):
        for i, j in self.h:
            self.h[i, j] = 0.0
            self.h_prev[i, j] = 0.0
            self.h_next[i, j] = 0.0

    @ti.kernel
    def step(self, courant2: ti.f32, damping: ti.f32):
        for i, j in self.h:
            center = self.h[i, j]
            # Clamped neighbors; the absorption band below eats what the
            # clamp reflects, approximating an open boundary.
            left = self.h[ti.max(0, i - 1), j]
            right = self.h[ti.min(WAKE_N - 1, i + 1), j]
            down = self.h[i, ti.max(0, j - 1)]
            up = self.h[i, ti.min(WAKE_N - 1, j + 1)]
            laplacian = left + right + down + up - 4.0 * center
            value = center + damping * (center - self.h_prev[i, j]) + courant2 * laplacian
            edge = ti.min(ti.min(i, WAKE_N - 1 - i), ti.min(j, WAKE_N - 1 - j))
            absorb = 1.0
            if edge < WAKE_ABSORB:
                absorb = 0.95 + 0.05 * edge / WAKE_ABSORB
            self.h_next[i, j] = value * absorb

    @ti.kernel
    def commit_step(self):
        for i, j in self.h:
            self.h_prev[i, j] = self.h[i, j]
            self.h[i, j] = self.h_next[i, j]

    @ti.func
    def disturbance(self, cx, cz, radius, strength, velocity):
        # Deposit a discrete negative Laplacian of a smooth Gaussian. Every
        # contribution sums to zero, so injection moves no net water.
        if ti.abs(cx) < WAKE_HALF - 0.5 and ti.abs(cz) < WAKE_HALF - 0.5:
            gx = (cx + WAKE_HALF) / WAKE_CELL - 0.5
            gz = (cz + WAKE_HALF) / WAKE_CELL - 0.5
            reach = ti.cast(ti.ceil(3 * radius / WAKE_CELL), ti.i32)
            ci, cj = ti.cast(ti.floor(gx), ti.i32), ti.cast(ti.floor(gz), ti.i32)
            for a, b in ti.ndrange((-reach, reach + 1), (-reach, reach + 1)):
                i, j = ci + a, cj + b
                if 0 <= i < WAKE_N and 0 <= j < WAKE_N:
                    dx, dz = (i - gx) * WAKE_CELL, (j - gz) * WAKE_CELL
                    r2 = (dx * dx + dz * dz) / (radius * radius)
                    value = 0.0
                    if r2 < 9.0:
                        # Value and radial derivative vanish at the support edge.
                        value = strength * (ti.exp(-r2) - ti.exp(-9.0) * (10.0 - r2)) * radius * radius / (4 * WAKE_CELL * WAKE_CELL)
                    for tap in ti.static(range(5)):
                        ii, jj, weight = i, j, 4.0
                        if ti.static(tap == 1):
                            ii, weight = ti.max(0, i - 1), -1.0
                        elif ti.static(tap == 2):
                            ii, weight = ti.min(WAKE_N - 1, i + 1), -1.0
                        elif ti.static(tap == 3):
                            jj, weight = ti.max(0, j - 1), -1.0
                        elif ti.static(tap == 4):
                            jj, weight = ti.min(WAKE_N - 1, j + 1), -1.0
                        delta = value * weight
                        if velocity:
                            # strength is the injected fluid speed in m/s.
                            ti.atomic_add(self.h_prev[ii, jj], -delta * WAKE_DT)
                        else:
                            ti.atomic_add(self.h[ii, jj], delta)
                            ti.atomic_add(self.h_prev[ii, jj], delta)

    @ti.kernel
    def inject(self, cx: ti.f32, cz: ti.f32, radius: ti.f32, strength: ti.f32):
        """Zero-volume displacement (metres), with zero initial velocity."""
        self.disturbance(cx, cz, ti.max(WAKE_CELL, radius), strength, False)

    @ti.func
    def sample(self, x, z):
        """Bilinear height with the cell-exact bilinear gradient."""
        u = ti.max(0.0, ti.min(WAKE_N - 1.0, (x + WAKE_HALF) / WAKE_CELL - 0.5))
        v = ti.max(0.0, ti.min(WAKE_N - 1.0, (z + WAKE_HALF) / WAKE_CELL - 0.5))
        i0, j0 = ti.min(WAKE_N - 2, ti.cast(u, ti.i32)), ti.min(WAKE_N - 2, ti.cast(v, ti.i32))
        fu, fv = u - i0, v - j0
        h00 = self.h[i0, j0]
        h10 = self.h[i0 + 1, j0]
        h01 = self.h[i0, j0 + 1]
        h11 = self.h[i0 + 1, j0 + 1]
        height = (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv
        dhdx = ((h10 - h00) * (1.0 - fv) + (h11 - h01) * fv) / WAKE_CELL
        dhdz = ((h01 - h00) * (1.0 - fu) + (h11 - h10) * fu) / WAKE_CELL
        return ti.Vector([height, dhdx, dhdz])

    @ti.func
    def vertical_velocity(self, x, z):
        """Bilinear vertical velocity of the surface (leapfrog derivative)."""
        u = ti.max(0.0, ti.min(WAKE_N - 1.0, (x + WAKE_HALF) / WAKE_CELL - 0.5))
        v = ti.max(0.0, ti.min(WAKE_N - 1.0, (z + WAKE_HALF) / WAKE_CELL - 0.5))
        i0, j0 = ti.min(WAKE_N - 2, ti.cast(u, ti.i32)), ti.min(WAKE_N - 2, ti.cast(v, ti.i32))
        fu, fv = u - i0, v - j0
        v00 = (self.h[i0, j0] - self.h_prev[i0, j0]) / WAKE_DT
        v10 = (self.h[i0 + 1, j0] - self.h_prev[i0 + 1, j0]) / WAKE_DT
        v01 = (self.h[i0, j0 + 1] - self.h_prev[i0, j0 + 1]) / WAKE_DT
        v11 = (self.h[i0 + 1, j0 + 1] - self.h_prev[i0 + 1, j0 + 1]) / WAKE_DT
        return (v00 * (1.0 - fu) + v10 * fu) * (1.0 - fv) + (v01 * (1.0 - fu) + v11 * fu) * fv

    def advance(self, wave_speed, damping, substeps):
        courant2 = (wave_speed * WAKE_DT / WAKE_CELL) ** 2
        for _ in range(substeps):
            self.step(courant2, damping)
            self.commit_step()
        self.steps += substeps


@ti.data_oriented
class HarborRenderer(SunsetOceanRenderer):
    """Demo 03 ocean plus floating bodies, contact foam, and a wake field."""

    def __init__(self, width, height, samples=4):
        super().__init__(width, height, samples)
        self.wake = WakeField()
        self.body_kind = ti.field(ti.i32, shape=MAX_BODIES)
        self.body_pos = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)
        self.body_vel = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)
        self.body_yaw = ti.field(ti.f32, shape=MAX_BODIES)
        self.body_pitch = ti.field(ti.f32, shape=MAX_BODIES)
        self.body_roll = ti.field(ti.f32, shape=MAX_BODIES)
        self.body_ang = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)
        self.body_rot = ti.Matrix.field(3, 3, ti.f32, shape=MAX_BODIES)
        self.body_half = ti.Vector.field(3, ti.f32, shape=MAX_BODIES)
        self.part_local = ti.Vector.field(3, ti.f32, shape=(MAX_BODIES, PARTS_PER_BODY))
        self.part_extent = ti.Vector.field(3, ti.f32, shape=(MAX_BODIES, PARTS_PER_BODY))
        self.part_color = ti.Vector.field(3, ti.f32, shape=(MAX_BODIES, PARTS_PER_BODY))
        self.part_material = ti.field(ti.i32, shape=(MAX_BODIES, PARTS_PER_BODY))
        self.part_bevel = ti.field(ti.f32, shape=(MAX_BODIES, PARTS_PER_BODY))
        self.part_radius = ti.field(ti.f32, shape=(MAX_BODIES, PARTS_PER_BODY))
        self.part_count = ti.field(ti.i32, shape=MAX_BODIES)
        self.boat_target = ti.Vector.field(2, ti.f32, shape=())
        self.boat_forced = ti.field(ti.i32, shape=())
        self._init_parts()
        self.wake.clear()
        self.reset_bodies()

    def _init_parts(self):
        # Rotated rounded boxes in body-local coordinates (origin at the
        # waterline); index 0 is the boat, 1..3 are buoys.
        boat_parts = [
            ((0.0, -0.05, 0.0), (0.85, 0.22, 0.34), 0.12, (0.45, 0.30, 0.18), 4),
            ((0.0, 0.13, 0.0), (0.95, 0.05, 0.40), 0.03, (0.50, 0.34, 0.20), 4),
            ((-0.1, 0.08, 0.0), (0.08, 0.03, 0.30), 0.015, (0.55, 0.38, 0.22), 4),
            ((-0.95, 0.10, 0.0), (0.10, 0.16, 0.09), 0.03, (0.06, 0.065, 0.07), 7),
        ]
        buoy_parts = [
            ((0.0, -0.12, 0.0), (0.26, 0.16, 0.26), 0.08, (0.72, 0.10, 0.06), 4),
            ((0.0, 0.25, 0.0), (0.035, 0.30, 0.035), 0.012, (0.12, 0.12, 0.13), 4),
            ((0.0, 0.56, 0.0), (0.09, 0.06, 0.09), 0.02, (0.85, 0.50, 0.18), 3),
        ]
        local = np.zeros((MAX_BODIES, PARTS_PER_BODY, 3), dtype=np.float32)
        extent = np.zeros((MAX_BODIES, PARTS_PER_BODY, 3), dtype=np.float32)
        color = np.zeros((MAX_BODIES, PARTS_PER_BODY, 3), dtype=np.float32)
        bevel = np.zeros((MAX_BODIES, PARTS_PER_BODY), dtype=np.float32)
        radius = np.zeros((MAX_BODIES, PARTS_PER_BODY), dtype=np.float32)
        material = np.zeros((MAX_BODIES, PARTS_PER_BODY), dtype=np.int32)
        counts = np.zeros(MAX_BODIES, dtype=np.int32)
        for parts, body in ((boat_parts, BOAT_INDEX), (buoy_parts, 1),
                            (buoy_parts, 2), (buoy_parts, 3)):
            for p, (center, half, radius_b, tint, mat) in enumerate(parts):
                local[body, p] = center
                extent[body, p] = half
                bevel[body, p] = min(radius_b, min(half) * 0.8)
                # Collision proxy: bounding sphere with the horizontal
                # footprint of the part, slightly shrunk by the bevel so
                # contact happens near the visible hull, not the AABB.
                radius[body, p] = max(0.05, math.hypot(half[0], half[2]) - bevel[body, p])
                color[body, p] = tint
                material[body, p] = mat
            counts[body] = len(parts)
        self.part_local.from_numpy(local)
        self.part_extent.from_numpy(extent)
        self.part_color.from_numpy(color)
        self.part_bevel.from_numpy(bevel)
        self.part_radius.from_numpy(radius)
        self.part_material.from_numpy(material)
        self._part_counts = counts

    @ti.kernel
    def reset_bodies(self):
        self.boat_target[None] = ti.Vector([-1.2, 1.0])
        self.boat_forced[None] = 0
        for b in ti.static(range(MAX_BODIES)):
            self.body_kind[b] = 0
            self.body_vel[b] = ti.Vector([0.0, 0.0, 0.0])
            self.body_ang[b] = ti.Vector([0.0, 0.0, 0.0])
            self.body_pitch[b] = 0.0
            self.body_roll[b] = 0.0
            self.body_yaw[b] = 0.0
            self.body_half[b] = ti.Vector([0.26, 0.16, 0.26])
            self.part_count[b] = 3
            self.body_pos[b] = ti.Vector([0.0, 0.0, 0.0])
        self.body_kind[BOAT_INDEX] = 1
        # The spawn keeps at least a hull length clear of the foreground
        # rock at (1.9, 3.4) so the collision proxy starts contact-free.
        self.body_pos[BOAT_INDEX] = ti.Vector([-1.2, 0.0, 1.0])
        self.body_half[BOAT_INDEX] = ti.Vector([0.85, 0.22, 0.34])
        self.part_count[BOAT_INDEX] = 4
        self.body_yaw[BOAT_INDEX] = 1.2
        self.body_pos[1] = ti.Vector([-3.5, 0.0, 2.5])
        self.body_pos[2] = ti.Vector([-1.0, 0.0, -3.5])
        self.body_pos[3] = ti.Vector([3.0, 0.0, 3.5])
        # One buoy starts heeled over to show the righting moment.
        self.body_roll[1] = 0.3
        for b in ti.static(range(MAX_BODIES)):
            self.body_rot[b] = (rot_y(self.body_yaw[b]) @ rot_x(self.body_pitch[b])
                                @ rot_z(self.body_roll[b]))

    @ti.kernel
    def step_bodies(self, dt: ti.f32, clock: ti.f32, advance: ti.i32):
        for b in ti.static(range(MAX_BODIES)):
            rot = (rot_y(self.body_yaw[b]) @ rot_x(self.body_pitch[b])
                   @ rot_z(self.body_roll[b]))
            total_a, torque_p, torque_r = 0.0, 0.0, 0.0
            # Five waterline samples: the center plus four corners.
            for s in ti.static(range(5)):
                ox, oz = 0.0, 0.0
                if ti.static(s == 1):
                    ox, oz = 0.62 * self.body_half[b].x, 0.62 * self.body_half[b].z
                elif ti.static(s == 2):
                    ox, oz = 0.62 * self.body_half[b].x, -0.62 * self.body_half[b].z
                elif ti.static(s == 3):
                    ox, oz = -0.62 * self.body_half[b].x, 0.62 * self.body_half[b].z
                elif ti.static(s == 4):
                    ox, oz = -0.62 * self.body_half[b].x, -0.62 * self.body_half[b].z
                off = rot @ ti.Vector([ox, 0.0, oz])
                px = self.body_pos[b].x + off.x
                py = self.body_pos[b].y + off.y
                pz = self.body_pos[b].z + off.z
                water_y = self.spectrum.height(px, pz, clock) + self.extra_height(px, pz, clock)
                water_vy = (self.spectrum.vertical_velocity(px, pz, clock)
                            + self.wake.vertical_velocity(px, pz))
                depth = water_y - py
                gate = self.smooth(-0.02, 0.05, depth)
                a = (BUOYANCY_K * depth + BUOYANCY_D * (water_vy - self.body_vel[b].y)) * gate
                total_a += a
                # Pitch is rotation about +x (bow at +z sinks for positive
                # pitch), roll about +z (+x side rises for positive roll);
                # these signs make a deeper corner lift, not capsize.
                torque_p += -off.z * a
                torque_r += off.x * a
            vel = self.body_vel[b]
            vel.y += (total_a * 0.2 - GRAVITY) * dt
            pos = self.body_pos[b] + vel * dt
            speed = ti.sqrt(vel.x * vel.x + vel.z * vel.z)
            if self.body_kind[b] == 1:
                if advance == 1:
                    # Steer toward the target at a bounded speed; the hull
                    # heading follows the velocity so the wake trails behind.
                    dx = self.boat_target[None].x - pos.x
                    dz = self.boat_target[None].y - pos.z
                    dist = ti.sqrt(dx * dx + dz * dz)
                    if dist > 1e-4:
                        step = ti.min(dist, BOAT_MAX_SPEED * dt)
                        vel.x = dx / dist * (step / dt)
                        vel.z = dz / dist * (step / dt)
                        pos.x += dx / dist * step
                        pos.z += dz / dist * step
                else:
                    vel.x *= ti.max(0.0, 1.0 - 2.0 * dt)
                    vel.z *= ti.max(0.0, 1.0 - 2.0 * dt)
                speed = ti.sqrt(vel.x * vel.x + vel.z * vel.z)
                if speed > 0.05:
                    heading = ti.atan2(-vel.z, vel.x)
                    turn = heading - self.body_yaw[b]
                    turn -= 2.0 * math.pi * ti.round(turn / (2.0 * math.pi))
                    self.body_yaw[b] += turn * ti.min(1.0, 2.0 * dt)
                # The stern digs in and pushes water outward as it moves.
                stern = pos + rot @ ti.Vector([-0.9, 0.0, 0.0])
                self.wake.disturbance(stern.x, stern.z, 0.45,
                                      -0.40 * ti.min(speed, 2.0), True)
            else:
                # A bobbing buoy squeezes a small ring of water each way.
                vy = ti.max(-1.0, ti.min(1.0, vel.y))
                self.wake.disturbance(pos.x, pos.z, 0.25, -0.12 * vy, True)
            ang = self.body_ang[b]
            ang.x += torque_p * 0.2 * TORQUE_GAIN * dt
            ang.z += torque_r * 0.2 * TORQUE_GAIN * dt
            damp = ti.max(0.0, 1.0 - ANGULAR_DAMP * dt)
            ang.x *= damp
            ang.y *= damp
            ang.z *= damp
            self.body_ang[b] = ang
            # Collision impulses write ang.y; the heading integrates it.
            self.body_yaw[b] += ang.y * dt
            self.body_pitch[b] = ti.max(-MAX_TILT, ti.min(MAX_TILT, self.body_pitch[b] + ang.x * dt))
            self.body_roll[b] = ti.max(-MAX_TILT, ti.min(MAX_TILT, self.body_roll[b] + ang.z * dt))
            self.body_vel[b] = vel
            self.body_pos[b] = pos
            rot = (rot_y(self.body_yaw[b]) @ rot_x(self.body_pitch[b])
                   @ rot_z(self.body_roll[b]))
            self.body_rot[b] = rot

    @ti.func
    def solve_body_pair(self, a, b):
        """Sphere-proxy contact between two bodies, horizontal response only."""
        rot_a = self.body_rot[a]
        rot_b = self.body_rot[b]
        pos_a = self.body_pos[a]
        pos_b = self.body_pos[b]
        for p in range(self.part_count[a]):
            ca = rot_a @ self.part_local[a, p] + pos_a
            ra = self.part_radius[a, p]
            for q in range(self.part_count[b]):
                cb = rot_b @ self.part_local[b, q] + pos_b
                reach = ra + self.part_radius[b, q]
                delta = ca - cb
                dist = delta.norm()
                if dist < reach and dist > 1e-5:
                    n = ti.Vector([delta.x, 0.0, delta.z])
                    nl = n.norm()
                    if nl > 1e-5:
                        n = n / nl
                        # The dragged boat is kinematic: the float takes the
                        # whole correction instead of splitting it.
                        share_a, share_b = 0.5, 0.5
                        if self.boat_forced[None] == 1:
                            if a == BOAT_INDEX:
                                share_a, share_b = 0.0, 1.0
                            elif b == BOAT_INDEX:
                                share_a, share_b = 1.0, 0.0
                        pos_a += n * ((reach - dist) * share_a)
                        pos_b -= n * ((reach - dist) * share_b)
                        vel_a = self.body_vel[a]
                        vel_b = self.body_vel[b]
                        vrel = (vel_a - vel_b).dot(n)
                        if vrel < 0.0:
                            j = -(1.0 + RESTITUTION) * vrel
                            imp_a = n * (j * share_a)
                            imp_b = n * (j * share_b)
                            self.body_vel[a] = vel_a + imp_a
                            self.body_vel[b] = vel_b - imp_b
                            # A glancing hit swings the struck body's bow
                            # (and the boat's) away from the contact.
                            r = ca - self.body_pos[a]
                            torque_y = r.z * imp_a.x - r.x * imp_a.z
                            ang = self.body_ang[a]
                            ang.y = ti.max(-MAX_YAW_RATE, ti.min(MAX_YAW_RATE, ang.y + torque_y * YAW_IMPULSE_GAIN))
                            self.body_ang[a] = ang
        self.body_pos[a] = pos_a
        self.body_pos[b] = pos_b

    @ti.func
    def solve_body_static(self, a, i):
        """Sphere-proxy contact against an axis-aligned static box."""
        rot = self.body_rot[a]
        pos = self.body_pos[a]
        center = self.box_center[i]
        half = self.box_extent[i]
        for p in range(self.part_count[a]):
            ca = rot @ self.part_local[a, p] + pos
            ra = self.part_radius[a, p]
            closest = ti.Vector([
                ti.max(center.x - half.x, ti.min(ca.x, center.x + half.x)),
                ti.max(center.y - half.y, ti.min(ca.y, center.y + half.y)),
                ti.max(center.z - half.z, ti.min(ca.z, center.z + half.z))])
            delta = ca - closest
            dist = delta.norm()
            if dist < ra:
                n = ti.Vector([delta.x, 0.0, delta.z])
                nl = n.norm()
                if dist > 1e-5 and nl > 1e-5:
                    n = n / nl
                    pos += n * (ra - dist)
                    vel = self.body_vel[a]
                    vrel = vel.dot(n)
                    if vrel < 0.0:
                        self.body_vel[a] = vel - n * ((1.0 + RESTITUTION) * vrel)
                else:
                    # Sphere center inside the box: escape along the axis
                    # that needs the least travel.
                    px = half.x - ti.abs(ca.x - center.x) + ra
                    pz = half.z - ti.abs(ca.z - center.z) + ra
                    if px < pz:
                        pos.x += ti.math.sign(ca.x - center.x) * px
                    else:
                        pos.z += ti.math.sign(ca.z - center.z) * pz
        self.body_pos[a] = pos

    @ti.kernel
    def resolve_collisions(self):
        """Iterated positional projection plus impulses; runs after stepping.

        Sequential rounds keep one pair's correction from leaving the next
        pair overlapping; four rounds settle the small body count.
        """
        for _ in range(COLLISION_ITERATIONS):
            for a in range(MAX_BODIES):
                for b in range(a + 1, MAX_BODIES):
                    self.solve_body_pair(a, b)
            for a in range(MAX_BODIES):
                for i in range(self.boxes):
                    self.solve_body_static(a, i)

    @ti.kernel
    def set_boat_target(self, x: ti.f32, z: ti.f32):
        self.boat_target[None] = ti.Vector([x, z])

    @ti.kernel
    def pick_boat(self, origin: ti.types.vector(3, ti.f32),
                  direction: ti.types.vector(3, ti.f32)) -> ti.f32:
        best = 1e6
        rot_t = self.body_rot[BOAT_INDEX].transpose()
        local_origin = rot_t @ (origin - self.body_pos[BOAT_INDEX])
        local_direction = rot_t @ direction
        for p in range(self.part_count[BOAT_INDEX]):
            t, _ = self.rounded_box_hit(local_origin, local_direction,
                                        self.part_local[BOAT_INDEX, p],
                                        self.part_extent[BOAT_INDEX, p],
                                        self.part_bevel[BOAT_INDEX, p])
            best = ti.min(best, t)
        hit = -1.0
        if best < 1e5:
            hit = best
        return hit

    @ti.func
    def body_hit(self, origin, direction, b):
        rot_t = self.body_rot[b].transpose()
        local_origin = rot_t @ (origin - self.body_pos[b])
        local_direction = rot_t @ direction
        closest, normal = 1e6, ti.Vector([0.0, 1.0, 0.0])
        color, material = ti.Vector([0.5, 0.5, 0.5]), -1
        # Dynamic part loop: unrolling 16 sphere traces per body per ray
        # made the render kernel take minutes to compile (demo 03 lesson).
        for p in range(self.part_count[b]):
            t, n = self.rounded_box_hit(local_origin, local_direction,
                                        self.part_local[b, p],
                                        self.part_extent[b, p],
                                        self.part_bevel[b, p])
            if t < closest:
                closest = t
                normal = self.body_rot[b] @ n
                color = self.part_color[b, p]
                material = self.part_material[b, p]
        return closest, normal, color, material

    @ti.func
    def geometry(self, origin, direction):
        closest, material = 1e6, -1
        normal, color = ti.Vector([0.0, 1.0, 0.0]), ti.Vector([0.5, 0.5, 0.5])
        if direction.y < -1e-6:
            t = (SEABED - origin.y) / direction.y
            if t > 0.001:
                closest, material = t, 6
        for i in range(self.boxes):
            t, n = self.rounded_box_hit(origin, direction, self.box_center[i],
                                        self.box_extent[i], self.box_bevel[i])
            if t < closest:
                closest, normal = t, n
                color, material = self.box_color[i], self.box_material[i]
        for b in range(MAX_BODIES):
            t, n, c, m = self.body_hit(origin, direction, b)
            if t < closest:
                closest, normal = t, n
                color, material = c, m
        return closest, normal, color, material

    @ti.func
    def occlusion_distance(self, origin, direction, limit):
        closest = limit
        for i in range(self.boxes):
            bound, _ = self.box_hit(origin, direction, self.box_center[i], self.box_extent[i])
            inside = (ti.abs(origin - self.box_center[i]) < self.box_extent[i]).all()
            if bound < closest or inside:
                t, _ = self.rounded_box_hit(origin, direction, self.box_center[i],
                                            self.box_extent[i], self.box_bevel[i])
                closest = ti.min(closest, t)
        # Ground and pool bed also occlude short contact rays.
        if direction.y < -0.0001:
            for level in ti.static(range(2)):
                height = -0.20 - 0.48 * level
                t = (height - origin.y) / direction.y
                if t > 0.001:
                    closest = ti.min(closest, t)
        for b in range(MAX_BODIES):
            t, _, _, _ = self.body_hit(origin, direction, b)
            closest = ti.min(closest, t)
        return closest

    @ti.func
    def extra_height(self, x, z, clock):
        return self.wake.sample(x, z).x

    @ti.func
    def extra_slope(self, x, z, clock):
        s = self.wake.sample(x, z)
        return ti.Vector([s.y, s.z])

    @ti.func
    def foam(self, p, clock, lighting):
        result = ti.Vector([0.0, 0.0, 0.0])
        for b in range(MAX_BODIES):
            rl = self.body_rot[b].transpose() @ (p - self.body_pos[b])
            if ti.abs(rl.y) < 0.45:
                speed = ti.sqrt(self.body_vel[b].x ** 2 + self.body_vel[b].z ** 2)
                activity = ti.min(1.0, speed * 1.5 + 0.35)
                noise = 0.6 + 0.4 * ti.sin(19.0 * p.x + 13.0 * p.z + clock * 3.1)
                ring = 0.0
                if self.body_kind[b] == 1:
                    q = ti.abs(ti.Vector([rl.x, rl.z])) - ti.Vector(
                        [self.body_half[b].x - 0.08, self.body_half[b].z - 0.08])
                    d = ti.max(q, 0.0).norm() + ti.min(ti.max(q.x, q.y), 0.0)
                    ring = (1.0 - self.smooth(0.02, 0.40, d)) * self.smooth(-0.30, -0.02, d)
                else:
                    radius = ti.sqrt(rl.x ** 2 + rl.z ** 2)
                    r0 = self.body_half[b].x
                    ring = (self.smooth(r0 - 0.10, r0 - 0.02, radius)
                            * (1.0 - self.smooth(r0 + 0.04, r0 + 0.42, radius)))
                result += ti.Vector([0.92, 0.88, 0.80]) * (0.5 * ring * activity * noise)
        return result


def cursor_ray(camera, u, v, width, height):
    """World-space ray through a normalized cursor position (v from the top)."""
    eye, forward, right, up = camera.basis()
    scale = math.tan(math.radians(camera.fov / 2.0))
    px = (2.0 * u - 1.0) * width / height * scale
    py = (2.0 * (1.0 - v) - 1.0) * scale
    direction = forward + right * px + up * py
    direction /= np.linalg.norm(direction)
    return eye.astype(np.float32), direction.astype(np.float32)


def ground_point(camera, u, v, width, height, limit=DRAG_LIMIT):
    """Intersect the cursor ray with the y = 0 plane, clamped to the harbor."""
    eye, direction = cursor_ray(camera, u, v, width, height)
    point = None
    if direction[1] < -1e-5:
        t = -eye[1] / direction[1]
        hit = eye + direction * t
        if abs(hit[0]) < limit and abs(hit[2]) < limit:
            point = (float(np.clip(hit[0], -limit, limit)),
                     float(np.clip(hit[2], -limit, limit)))
    return point


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 04: buoys and a boat on the sunset sea")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=4, help="Spatial samples per pixel (default: 4)")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="sunset")
    parser.add_argument("--shadow-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--water-roughness", type=float, default=0.05)
    parser.add_argument("--reflection-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-water-detail", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--cruise", action="store_true", help="Start with the boat cruising a slow circle")
    parser.add_argument("--wind", type=float, default=8.0, help="Wind speed in m/s")
    parser.add_argument("--wind-dir", type=float, default=35.0, help="Wind direction in degrees")
    parser.add_argument("--wave-height", type=float, default=1.3, help="Total wave amplitude in metres")
    parser.add_argument("--steepness", type=float, default=0.55, help="Gerstner steepness sum Q k A, 0..1")
    parser.add_argument("--wave-scale", type=float, default=0.8, help="Wavelength scale multiplier")
    parser.add_argument("--preset", choices=("overview", "waterline", "top"), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_04.png"))
    parser.add_argument("--time", type=float, default=3.0, help="Simulation warmup in seconds before the first frame")
    parser.add_argument("--frames", type=int, default=1, help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0, help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache for troubleshooting")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0 or not math.isfinite(args.time):
        parser.error("width/height must be >= 32, frames >= 1, and time finite")
    if not math.isfinite(args.water_roughness) or not 0.0 <= args.water_roughness <= 0.45:
        parser.error("water-roughness must be finite and between 0 and 0.45")
    if not math.isfinite(args.wind) or not 0.0 <= args.wind <= 24.0:
        parser.error("wind must be finite and between 0 and 24")
    if not math.isfinite(args.wind_dir):
        parser.error("wind-dir must be finite")
    if not math.isfinite(args.wave_height) or not 0.05 <= args.wave_height <= 3.0:
        parser.error("wave-height must be finite and between 0.05 and 3")
    if not math.isfinite(args.steepness) or not 0.0 <= args.steepness <= 1.0:
        parser.error("steepness must be finite and between 0 and 1")
    if not math.isfinite(args.wave_scale) or not 0.5 <= args.wave_scale <= 3.0:
        parser.error("wave-scale must be finite and between 0.5 and 3")
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = HarborRenderer(args.width, args.height, args.samples)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 0.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = OrbitCamera()
    apply_preset(camera, args.preset)
    wind = (args.wind, args.wind_dir, args.wave_height, args.steepness, args.wave_scale)
    speed = 1.0
    lighting = 0.0 if args.lighting == "sunset" else 1.0
    clarity, exposure = 1.4, 1.05
    clock = 0.0
    renderer.spectrum.rebuild(*wind)
    cruise = args.cruise
    dragging = False

    def advance(substeps):
        nonlocal clock
        for _ in range(substeps):
            if cruise and not dragging:
                angle = clock * BOAT_CRUISE_RATE
                renderer.set_boat_target(BOAT_CRUISE_RADIUS * math.cos(angle),
                                         BOAT_CRUISE_RADIUS * math.sin(angle))
            renderer.wake.advance(WAKE_SPEED, WAKE_DAMPING, 1)
            renderer.step_bodies(WAKE_DT, clock, 1 if (dragging or cruise) else 0)
            renderer.resolve_collisions()
            clock += WAKE_DT

    # Warmup settles the bodies onto the swell before the first visible frame.
    warmup_steps = int(max(0.0, args.time) * 120.0)
    for _ in range(warmup_steps):
        advance(1)
    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            advance(2)
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
    renderer.draw(camera, clock, clarity, lighting, exposure)
    ti.sync()
    print(f"[Startup] First frame ready in {time.perf_counter() - warmup_start:.2f}s. Opening window...", flush=True)
    window = ti.ui.Window("04 / Harbor Floats", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    sim_accumulator, frame_ms, window_frames = 0.0, 0.0, 1
    print("Drag boat: move it | Drag LMB orbit | RMB pan | W/S zoom | 1/2/3 views | C cruise | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
    while window.running and (not args.window_frames or window_frames < args.window_frames):
        now = time.perf_counter()
        dt = min(now - previous_time, 0.25)
        previous_time = now
        step, save_frame = False, False
        mouse = np.array(window.get_cursor_pos())
        in_panel = show_panel and ((mouse[0] < 0.33 and mouse[1] > 0.15) or (mouse[0] > 0.70 and mouse[1] > 0.04))
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
            elif key == "c":
                cruise = not cruise
            elif key == "h":
                show_panel = not show_panel
            elif key in ("1", "2", "3"):
                apply_preset(camera, {"1": "overview", "2": "waterline", "3": "top"}[key])
                auto_orbit = False
            elif key == "r":
                apply_preset(camera, "overview")
                renderer.samples = args.samples
                renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 0.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                wind = (args.wind, args.wind_dir, args.wave_height, args.steepness, args.wave_scale)
                renderer.spectrum.rebuild(*wind)
                renderer.wake.clear()
                renderer.reset_bodies()
                lighting = 0.0 if args.lighting == "sunset" else 1.0
                clarity, exposure, clock = 1.4, 1.05, 0.0
                sim_accumulator = 0.0
                paused, auto_orbit, cruise, dragging = False, False, args.cruise, False
            elif key == "p":
                save_frame = True
            elif key == ti.ui.LMB and not in_panel:
                eye, direction = cursor_ray(camera, mouse[0], mouse[1], args.width, args.height)
                if renderer.pick_boat(eye, direction) > 0.0:
                    dragging = True
                    auto_orbit = False
                    point = ground_point(camera, mouse[0], mouse[1], args.width, args.height)
                    if point is not None:
                        renderer.set_boat_target(*point)
        if not window.running:
            break
        # While dragging, the cursor steers the boat; a missed pick orbits.
        if dragging:
            if window.is_pressed(ti.ui.LMB):
                point = ground_point(camera, mouse[0], mouse[1], args.width, args.height)
                if point is not None:
                    renderer.set_boat_target(*point)
            else:
                dragging = False
        elif previous_mouse is not None and not in_panel and window.is_pressed(ti.ui.LMB):
            camera.orbit(*(mouse - previous_mouse))
            auto_orbit = False
        elif previous_mouse is not None and not in_panel and window.is_pressed(ti.ui.RMB):
            camera.pan(*(mouse - previous_mouse))
            auto_orbit = False
        renderer.boat_forced[None] = 1 if dragging else 0
        previous_mouse = mouse
        if window.is_pressed("w", ti.ui.UP):
            camera.zoom(-dt)
        if window.is_pressed("s", ti.ui.DOWN):
            camera.zoom(dt)
        if auto_orbit:
            camera.yaw += dt * 0.12
        if show_panel:
            with gui.sub_window("LIGHT / MATERIALS", 0.72, 0.025, 0.26, 0.28):
                controls = renderer.light_controls[None]
                controls[1] = 4 if gui.checkbox("Soft sun shadows", controls[1] == 4) else 1
                controls[0] = math.radians(gui.slider_float("Sun radius (deg)", math.degrees(controls[0]), 0.0, 3.0))
                controls[2] = gui.slider_float("Contact AO", controls[2], 0.0, 1.0)
                controls[3] = gui.slider_float("Environment", controls[3], 0.0, 2.0)
                renderer.light_controls[None] = controls
            with gui.sub_window("OCEAN / FINISH", 0.72, 0.335, 0.26, 0.52):
                controls = renderer.water_controls[None]
                controls[0] = gui.slider_float("Water roughness", controls[0], 0.0, 0.45)
                controls[1] = gui.slider_float("Fine waves", controls[1], 0.0, 2.0)
                controls[3] = 4 if gui.checkbox("4x reflections", controls[3] == 4) else 1
                renderer.water_controls[None] = controls
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
            with gui.sub_window("HARBOR / 04", 0.02, 0.025, 0.30, 0.78):
                gui.text("Buoyancy / wake / contact foam")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                cruise = gui.checkbox("Boat cruise [C]", cruise)
                if gui.button("Single step [N]"):
                    step = True
                wind_speed = gui.slider_float("Wind speed", wind[0], 0.0, 24.0)
                wind_dir = gui.slider_float("Wind direction", wind[1], 0.0, 360.0)
                wave_height = gui.slider_float("Wave height", wind[2], 0.05, 3.0)
                steepness = gui.slider_float("Steepness", wind[3], 0.0, 1.0)
                wave_scale = gui.slider_float("Wave scale", wind[4], 0.5, 3.0)
                wind = (wind_speed, wind_dir, wave_height, steepness, wave_scale)
                speed = gui.slider_float("Time speed", speed, 0.0, 2.0)
                clarity = gui.slider_float("Water clarity", clarity, 0.35, 3.0)
                lighting = gui.slider_float("Day / sunset", lighting, 0.0, 1.0)
                exposure = gui.slider_float("Exposure", exposure, 0.5, 1.8)
                renderer.samples = 4 if gui.checkbox("4x spatial AA", renderer.samples == 4) else 1
                if gui.button("Overview [1]"):
                    apply_preset(camera, "overview")
                    auto_orbit = False
                if gui.button("Waterline [2]"):
                    apply_preset(camera, "waterline")
                    auto_orbit = False
                if gui.button("Top view [3]"):
                    apply_preset(camera, "top")
                    auto_orbit = False
                gui.text("Drag the boat to move it")
                gui.text("Drag LMB orbit / RMB pan")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        if paused:
            if step:
                advance(1)
        else:
            substeps, sim_accumulator = simulation_budget(sim_accumulator, dt, speed)
            advance(substeps)
        # The Gerstner set is rebuilt on the host only when its inputs move.
        current = tuple(float(v) for v in wind)
        if current != getattr(renderer, "_wind_key", None):
            renderer.spectrum.rebuild(*current)
            renderer._wind_key = current
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
