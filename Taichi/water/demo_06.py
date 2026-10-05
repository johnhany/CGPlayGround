"""Shore reef: shallow-water waves rolling onto a sandy beach around rocks.

The sea is driven by the shallow-water equations on a 256x256 grid over a
128 m square domain. A limited MUSCL finite-volume update with Rusanov fluxes
carries the depth and the two discharge components; the analytic seabed
(a sloping beach plus four Gaussian reefs) enters through hydrostatic
reconstruction so still water over any bed stays still, and a light bed
friction keeps runup thin. The west edge is a wave-making
boundary that injects the incident swell; the other three edges transmit.
Wave runup and retreat leave foam and wet-sand traces that the shore
material reads back. Rendering reuses the demo 03 ray-marched water with
wet-aware cubic surface sampling and replaces the seabed plane with an analytic
terrain march so the waterline, wet sand band, and protruding reefs shade
correctly.

Run: uv run python water/demo_06.py
Export: uv run python water/demo_06.py --headless --output output/shore.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .demo_03 import (BISECT_STEPS, GRAVITY, MARCH_GROWTH, MARCH_MAX_STEPS,
                          MARCH_MIN_STEP, SunsetOceanRenderer)
    from .scene import OrbitCamera
else:
    from demo_03 import (BISECT_STEPS, GRAVITY, MARCH_GROWTH, MARCH_MAX_STEPS,
                         MARCH_MIN_STEP, SunsetOceanRenderer)
    from scene import OrbitCamera

__all__ = ["ShallowSea", "ShoreRenderer", "ShoreCamera", "main", "parse_args",
           "sea_parameters", "simulation_budget", "water_pick", "apply_preset",
           "shore_terrain", "shore_terrain_grad", "reef_factor",
           "terrain_march_step", "MAX_BED_SLOPE",
           "SEA_HALF", "SEA_N", "SEA_DX", "SIM_DT",
           "SHORE_PRESETS", "SHORE_PRESETS_SEA"]

# Domain: x, z in [-64, 64] metres; the sea sits at -x, the beach at +x
# and the still water level rests at y = 0.
SEA_HALF = 64.0
SEA_N = 256
SEA_DX = 0.5
SIM_DT = 1.0 / 30.0
DRY_EPS = 0.01
WET_EPS = 0.004
FOAM_TAU = 5.0
WET_TAU = 15.0
# Ray marching clamps the terrain search to this vertical slab.
TERRAIN_Y_LO = -4.5
TERRAIN_Y_HI = 3.5
# Global bound on |grad shore_terrain|: the steepest reef flank is
# top * sqrt(2) * exp(-1/2) / sigma ~= 0.86 * top / sigma, before the small crag modulation. The 2.3 bound includes that modulation. The terrain marcher divides the ray error by
# -dir.y + MAX_BED_SLOPE * |dir_h| to get a step that cannot jump an
# intersection; the bound must stay a true superset of every flank slope.
MAX_BED_SLOPE = 2.3
# Gaussian reefs: (cx, cz, height, sigma). The first three pierce the
# surface and diffract the swell; the last one sits high on the beach.
REEFS = ((-18.0, -10.0, 3.0, 1.6),
         (-6.0, 8.0, 2.2, 1.2),
         (-30.0, 18.0, 2.3, 2.0),
         (20.0, -20.0, 0.9, 1.4))

SHORE_PRESETS = {
    "overview": (0.50, 0.35, 15.0),
    "waterline": (0.50, 0.12, 9.0),
    "top": (0.50, 1.42, 26.0),
}
# Sea presets: (incident amplitude m, period s)
SHORE_PRESETS_SEA = {
    "calm": (0.12, 9.0),
    "surf": (0.30, 10.0),
    "storm": (0.55, 13.0),
}


def apply_preset(camera, name):
    camera.target = np.array([2.0, 0.0, 0.0])
    camera.yaw, camera.pitch, camera.distance = SHORE_PRESETS[name]


@ti.func
def shore_terrain(x, z):
    """Analytic seabed height b(x, z); b = 0 is the still shoreline.

    The alongshore bend is a sum of incommensurate sines, not one sine: a
    single sine makes the waterline a literal sinusoid, which reads as an
    obviously artificial beach. The phases cancel at z = 0 so the still
    waterline stays near x = 4 there.
    """
    bounded_x = x
    if x > SEA_HALF:
        bounded_x = SEA_HALF + 8.0 * ti.tanh((x - SEA_HALF) / 8.0)
    elif x < -SEA_HALF:
        bounded_x = -SEA_HALF + 8.0 * ti.tanh((x + SEA_HALF) / 8.0)
    height = (0.045 * (bounded_x - 4.0)
              + 0.16 * ti.sin(0.031 * z + 0.9)
              + 0.10 * ti.sin(0.083 * z + 3.94)
              + 0.06 * ti.sin(0.017 * z + 4.32))
    for r in ti.static(range(4)):
        cx, cz, top, sigma = -18.0, -10.0, 3.0, 1.6
        if ti.static(r == 1):
            cx, cz, top, sigma = -6.0, 8.0, 2.2, 1.2
        if ti.static(r == 2):
            cx, cz, top, sigma = -30.0, 18.0, 2.3, 2.0
        if ti.static(r == 3):
            cx, cz, top, sigma = 20.0, -20.0, 0.9, 1.4
        dx, dz = x - cx, z - cz
        crag = 1.0 + 0.06 * ti.sin(1.8 * dx) * ti.sin(1.3 * dz)
        height += top * ti.exp(-(dx * dx + dz * dz) / (sigma * sigma)) * crag
    return height


@ti.func
def shore_terrain_grad(x, z):
    """Analytic gradient of the seabed height."""
    gx = 0.045
    if ti.abs(x) > SEA_HALF:
        extension = (ti.abs(x) - SEA_HALF) / 8.0
        gx *= 1.0 - ti.tanh(extension) ** 2
    gz = (0.16 * 0.031 * ti.cos(0.031 * z + 0.9)
          + 0.10 * 0.083 * ti.cos(0.083 * z + 3.94)
          + 0.06 * 0.017 * ti.cos(0.017 * z + 4.32))
    for r in ti.static(range(4)):
        cx, cz, top, sigma = -18.0, -10.0, 3.0, 1.6
        if ti.static(r == 1):
            cx, cz, top, sigma = -6.0, 8.0, 2.2, 1.2
        if ti.static(r == 2):
            cx, cz, top, sigma = -30.0, 18.0, 2.3, 2.0
        if ti.static(r == 3):
            cx, cz, top, sigma = 20.0, -20.0, 0.9, 1.4
        dx, dz = x - cx, z - cz
        gauss = top * ti.exp(-(dx * dx + dz * dz) / (sigma * sigma))
        sx, cz = ti.sin(1.8 * dx), ti.cos(1.3 * dz)
        sz, cx = ti.sin(1.3 * dz), ti.cos(1.8 * dx)
        crag = 1.0 + 0.06 * sx * sz
        gx += gauss * (-2.0 * dx / (sigma * sigma) * crag + 0.108 * cx * sz)
        gz += gauss * (-2.0 * dz / (sigma * sigma) * crag + 0.078 * sx * cz)
    return ti.Vector([gx, gz])


@ti.func
def reef_factor(x, z):
    """0..1 rock coverage from the Gaussian reef bumps."""
    cover = 0.0
    for r in ti.static(range(4)):
        cx, cz, top, sigma = -18.0, -10.0, 3.0, 1.6
        if ti.static(r == 1):
            cx, cz, top, sigma = -6.0, 8.0, 2.2, 1.2
        if ti.static(r == 2):
            cx, cz, top, sigma = -30.0, 18.0, 2.3, 2.0
        if ti.static(r == 3):
            cx, cz, top, sigma = 20.0, -20.0, 0.9, 1.4
        dx, dz = x - cx, z - cz
        cover += ti.exp(-(dx * dx + dz * dz) / (sigma * sigma))
    return ti.min(1.0, cover)


@ti.func
def incident_surface(x, z, t, amplitude, period):
    """Ramped multi-component incident sea surface.

    A single sine train arrives as perfectly regular, shore-parallel crests
    and leaves evenly spaced foam arcs. Two incommensurate frequencies, a
    slight opposing obliqueness, and alongshore amplitude drift break that
    up: crest lines tilt, wave groups form, and the breaking band wanders
    along the beach. The wavenumbers use the shallow-water value at the
    boundary depth (3 m), and the phase is referenced to the west edge
    (x + SEA_HALF = 0) so the ghost column and the far-field extension
    agree at the boundary.
    """
    period = ti.max(period, 1.0)
    ramp = ti.min(1.0, t / period)
    omega1 = 2.0 * math.pi / period
    omega2 = omega1 * 1.71
    k1 = omega1 / ti.sqrt(GRAVITY * 3.0)
    k2 = omega2 / ti.sqrt(GRAVITY * 3.0)
    kz1 = 0.105 * k1
    kz2 = -0.141 * k2
    drift = ti.sin(0.045 * z + 2.0)
    a1 = amplitude * 0.72 * (1.0 + 0.20 * drift)
    a2 = amplitude * 0.28 * (1.0 - 0.50 * drift)
    phase1 = omega1 * t - kz1 * z - k1 * (x + SEA_HALF)
    phase2 = omega2 * t + 1.7 - kz2 * z - k2 * (x + SEA_HALF)
    eta = ramp * (a1 * ti.sin(phase1) + a2 * ti.sin(phase2))
    gx = -ramp * (a1 * k1 * ti.cos(phase1) + a2 * k2 * ti.cos(phase2))
    drift_dz = 0.045 * ti.cos(0.045 * z + 2.0)
    gz = ramp * (amplitude * 0.144 * drift_dz * ti.sin(phase1)
                 - amplitude * 0.14 * drift_dz * ti.sin(phase2)
                 - a1 * kz1 * ti.cos(phase1) - a2 * kz2 * ti.cos(phase2))
    return ti.Vector([eta, gx, gz])


@ti.func
def incident_eta(x, z, t, amplitude, period):
    return incident_surface(x, z, t, amplitude, period).x


@ti.func
def terrain_march_step(error, current, dir_y, dir_h):
    """Conservative terrain-march step for one ray sample.

    For g(t) = p.y - shore_terrain(p.xz), the gradient bound gives
    g'(t) >= dir.y - MAX_BED_SLOPE * |dir_h| = -decay, so g cannot reach
    zero within 0.9 * error / decay: a larger fixed multiple (the open-ocean
    error * 1.4) strides over a reef summit, and bisection then brackets the
    far flank, clipping the peak along a line that wanders with the view.
    """
    decay = -dir_y + MAX_BED_SLOPE * dir_h
    step = 0.9 * error / decay
    return ti.max(ti.min(step, MARCH_MIN_STEP + MARCH_GROWTH * current),
                  MARCH_MIN_STEP)


@ti.func
def _cubic_weights(t):
    """Catmull-Rom interpolation weights and their derivatives (demo 05)."""
    t2, t3 = t * t, t * t * t
    weights = ti.Vector([-0.5 * t + t2 - 0.5 * t3,
                         1 - 2.5 * t2 + 1.5 * t3,
                         0.5 * t + 2 * t2 - 1.5 * t3,
                         -0.5 * t2 + 0.5 * t3])
    derivative = ti.Vector([-0.5 + 2 * t - 1.5 * t2,
                            -5 * t + 4.5 * t2,
                            0.5 + 4 * t - 4.5 * t2,
                            -t + 1.5 * t2])
    return weights, derivative


@ti.func
def _velocity(qx, qz, h):
    """Desingularized velocity; dry cells and fast jets are clamped."""
    u, v = 0.0, 0.0
    if h >= 0.0005:
        u = ti.max(-6.0, ti.min(6.0, qx / h))
        v = ti.max(-6.0, ti.min(6.0, qz / h))
    return u, v


@ti.func
def _flux_x(h, qx, qz):
    u, v = _velocity(qx, qz, h)
    return ti.Vector([qx, qx * u + 0.5 * GRAVITY * h * h, qx * v])


@ti.func
def _flux_z(h, qx, qz):
    u, v = _velocity(qx, qz, h)
    return ti.Vector([qz, qz * u, qz * v + 0.5 * GRAVITY * h * h])


@ti.func
def _rusanov_x(hL, qxL, qzL, hR, qxR, qzR):
    """Local Lax-Friedrichs flux across an x-normal face."""
    uL, _ = _velocity(qxL, qzL, hL)
    uR, _ = _velocity(qxR, qzR, hR)
    speed = ti.max(ti.abs(uL) + ti.sqrt(GRAVITY * ti.max(hL, 0.0)),
                   ti.abs(uR) + ti.sqrt(GRAVITY * ti.max(hR, 0.0)))
    flux = 0.5 * (_flux_x(hL, qxL, qzL) + _flux_x(hR, qxR, qzR))
    return flux - 0.5 * speed * ti.Vector([hR - hL, qxR - qxL, qzR - qzL])


@ti.func
def _rusanov_z(hL, qxL, qzL, hR, qxR, qzR):
    """Local Lax-Friedrichs flux across a z-normal face."""
    _, vL = _velocity(qxL, qzL, hL)
    _, vR = _velocity(qxR, qzR, hR)
    speed = ti.max(ti.abs(vL) + ti.sqrt(GRAVITY * ti.max(hL, 0.0)),
                   ti.abs(vR) + ti.sqrt(GRAVITY * ti.max(hR, 0.0)))
    flux = 0.5 * (_flux_z(hL, qxL, qzL) + _flux_z(hR, qxR, qzR))
    return flux - 0.5 * speed * ti.Vector([hR - hL, qxR - qxL, qzR - qzL])


@ti.func
def _face_x(hL, qxL, qzL, bL, hR, qxR, qzR, bR):
    """Well-balanced x-face flux: hydrostatic reconstruction plus the
    left-cell pressure correction, so still water over any bed stays still."""
    b_face = ti.max(bL, bR)
    hLs = ti.max(hL + bL - b_face, 0.0)
    hRs = ti.max(hR + bR - b_face, 0.0)
    qxLs, qzLs = 0.0, 0.0
    if hL > 1e-6:
        qxLs = qxL * ti.min(1.0, hLs / hL)
        qzLs = qzL * ti.min(1.0, hLs / hL)
    qxRs, qzRs = 0.0, 0.0
    if hR > 1e-6:
        qxRs = qxR * ti.min(1.0, hRs / hR)
        qzRs = qzR * ti.min(1.0, hRs / hR)
    flux = _rusanov_x(hLs, qxLs, qzLs, hRs, qxRs, qzRs)
    flux.y += 0.5 * GRAVITY * (hL * hL - hLs * hLs)
    return flux


@ti.func
def _face_z(hL, qxL, qzL, bL, hR, qxR, qzR, bR):
    """Well-balanced z-face flux; see _face_x."""
    b_face = ti.max(bL, bR)
    hLs = ti.max(hL + bL - b_face, 0.0)
    hRs = ti.max(hR + bR - b_face, 0.0)
    qxLs, qzLs = 0.0, 0.0
    if hL > 1e-6:
        qxLs = qxL * ti.min(1.0, hLs / hL)
        qzLs = qzL * ti.min(1.0, hLs / hL)
    qxRs, qzRs = 0.0, 0.0
    if hR > 1e-6:
        qxRs = qxR * ti.min(1.0, hRs / hR)
        qzRs = qzR * ti.min(1.0, hRs / hR)
    flux = _rusanov_z(hLs, qxLs, qzLs, hRs, qxRs, qzRs)
    flux.z += 0.5 * GRAVITY * (hL * hL - hLs * hLs)
    return flux


@ti.data_oriented
class ShallowSea:
    """Finite-volume shallow-water solver with wet/dry cells and open edges.

    State is the depth h and the unit-width discharges qx = h u, qz = h v.
    One step predicts into the *_new buffers; a commit pass rotates the
    buffers, advances the foam and wetness traces, and reduces the wet-cell
    surface elevation range used by the renderer for march bounds.
    """

    def __init__(self):
        self.h = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.qx = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.qz = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.h_new = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.qx_new = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.qz_new = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.bed = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.old_h = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.old_qx = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.old_qz = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.slope_x = ti.Vector.field(3, ti.f32, shape=(SEA_N, SEA_N))
        self.slope_z = ti.Vector.field(3, ti.f32, shape=(SEA_N, SEA_N))
        self.max_speed = ti.field(ti.f32, shape=())
        self.foam_adv = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.strand = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.strand_view = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.foam = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.wetness = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.h_view = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.foam_view = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.wet_view = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.eta_view = ti.field(ti.f32, shape=(SEA_N, SEA_N))
        self.eta_min = ti.field(ti.f32, shape=())
        self.eta_max = ti.field(ti.f32, shape=())
        self.amplitude = ti.field(ti.f32, shape=())
        self.period = ti.field(ti.f32, shape=())
        self.amplitude[None] = 0.30
        self.period[None] = 10.0
        self.time = 0.0
        self.steps = 0
        self.reset()

    def set_wave(self, amplitude, period):
        self.amplitude[None] = float(amplitude)
        self.period[None] = float(period)

    @ti.kernel
    def _clear(self):
        for i, j in self.h:
            x = -SEA_HALF + i * SEA_DX
            z = -SEA_HALF + j * SEA_DX
            self.bed[i, j] = shore_terrain(x, z)
            h = ti.max(-self.bed[i, j], 0.0)
            self.strand[i, j] = 0.0
            self.strand_view[i, j] = 0.0
            self.h[i, j] = h
            self.qx[i, j] = 0.0
            self.qz[i, j] = 0.0
            self.h_new[i, j] = h
            self.qx_new[i, j] = 0.0
            self.qz_new[i, j] = 0.0
            self.foam[i, j] = 0.0
            self.wetness[i, j] = 0.0
            self.h_view[i, j] = h
            self.eta_view[i, j] = h + shore_terrain(x, z)
            self.foam_view[i, j] = 0.0
            self.wet_view[i, j] = 0.0

    @ti.kernel
    def _sync_view(self):
        """Wet-aware free-surface reconstruction preserves a flat lake."""
        for i, j in self.h:
            eta_sum, weight = 0.0, 0.0
            foam, wet, strand = 0.0, 0.0, 0.0
            for a, b in ti.static(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
                ii, jj = ti.max(0, ti.min(SEA_N - 1, i + a)), ti.max(0, ti.min(SEA_N - 1, j + b))
                w = 0.125
                if ti.static(a == 0 and b == 0):
                    w = 0.5
                if self.h[ii, jj] > 0.0001:
                    eta_sum += w * (self.h[ii, jj] + self.bed[ii, jj])
                    weight += w
                foam += w * self.foam[ii, jj]
                wet += w * self.wetness[ii, jj]
                strand += w * self.strand[ii, jj]
            elevation = 0.0
            if weight > 0.0:
                elevation = eta_sum / weight
            self.eta_view[i, j] = elevation
            self.h_view[i, j] = ti.max(0.0, elevation - self.bed[i, j])
            self.foam_view[i, j] = foam
            self.wet_view[i, j] = wet
            self.strand_view[i, j] = strand

    def reset(self):
        """Still water over the seabed, zero momentum and clean traces."""
        self._clear()
        self.time = 0.0
        self.steps = 0
        self.eta_min[None] = 0.0
        self.eta_max[None] = 0.0
        self.last_cfl = 0.0
        self.internal_steps = 0
        self._sync_view()

    @ti.func
    def _primitive(self, i, j, t):
        ii, jj = ti.max(0, ti.min(SEA_N - 1, i)), ti.max(0, ti.min(SEA_N - 1, j))
        h, bed = self.h[ii, jj], self.bed[ii, jj]
        u, v = _velocity(self.qx[ii, jj], self.qz[ii, jj], h)
        if i < 0:
            z = -SEA_HALF + jj * SEA_DX
            bed = self.bed[0, jj]
            eta = incident_eta(-SEA_HALF, z, t, self.amplitude[None], self.period[None])
            h = ti.max(-bed + eta, 0.0)
            u = eta * ti.sqrt(GRAVITY / ti.max(-bed, 0.05))
            v = 0.0
        return ti.Vector([h + bed, u, v]), bed

    @ti.func
    def _limit(self, left, right):
        result = ti.Vector([0.0, 0.0, 0.0])
        for k in ti.static(range(3)):
            if left[k] * right[k] > 0.0:
                result[k] = ti.select(left[k] > 0.0, 1.0, -1.0) * ti.min(ti.abs(left[k]), ti.abs(right[k]))
        return result

    @ti.kernel
    def _reconstruct(self, t: ti.f32):
        for i, j in self.h:
            center, _ = self._primitive(i, j, t)
            west, _ = self._primitive(i - 1, j, t)
            east, _ = self._primitive(i + 1, j, t)
            south, _ = self._primitive(i, j - 1, t)
            north, _ = self._primitive(i, j + 1, t)
            sx, sz = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
            # Wet/dry faces revert locally to first order; dry bed elevations
            # cannot be mistaken for a free-surface slope.
            if 0 < i < SEA_N - 1 and self.h[i - 1, j] > WET_EPS and self.h[i, j] > WET_EPS and self.h[i + 1, j] > WET_EPS:
                sx = self._limit(center - west, east - center)
            if 0 < j < SEA_N - 1 and self.h[i, j - 1] > WET_EPS and self.h[i, j] > WET_EPS and self.h[i, j + 1] > WET_EPS:
                sz = self._limit(center - south, north - center)
            depth = self.h[i, j]
            sx.x = ti.max(-2 * depth, ti.min(2 * depth, sx.x))
            sz.x = ti.max(-2 * depth, ti.min(2 * depth, sz.x))
            self.slope_x[i, j], self.slope_z[i, j] = sx, sz

    @ti.func
    def _interface(self, i, j, axis, t):
        di, dj = 1, 0
        if axis == 1:
            di, dj = 0, 1
        left, bl = self._primitive(i, j, t)
        right, br = self._primitive(i + di, j + dj, t)
        if 0 <= i < SEA_N and 0 <= j < SEA_N:
            slope = self.slope_x[i, j]
            if axis == 1:
                slope = self.slope_z[i, j]
            left += 0.5 * slope
        if 0 <= i + di < SEA_N and 0 <= j + dj < SEA_N:
            slope = self.slope_x[i + di, j + dj]
            if axis == 1:
                slope = self.slope_z[i + di, j + dj]
            right -= 0.5 * slope
        hl, hr = ti.max(left.x - bl, 0.0), ti.max(right.x - br, 0.0)
        face_bed = ti.max(bl, br)
        hls, hrs = ti.max(left.x - face_bed, 0.0), ti.max(right.x - face_bed, 0.0)
        flux = _rusanov_x(hls, hls * left.y, hls * left.z, hrs, hrs * right.y, hrs * right.z)
        if axis == 1:
            flux = _rusanov_z(hls, hls * left.y, hls * left.z, hrs, hrs * right.y, hrs * right.z)
        # Each cell receives its own hydrostatic pressure correction. This
        # replaces the old one-sided bed source without upsetting a lake.
        cl = 0.5 * GRAVITY * (hl * hl - hls * hls)
        cr = 0.5 * GRAVITY * (hr * hr - hrs * hrs)
        return flux, cl, cr

    @ti.kernel
    def _step(self, t: ti.f32, dt: ti.f32):
        for i, j in self.h:
            west, _, cw = self._interface(i - 1, j, 0, t)
            east, ce, _ = self._interface(i, j, 0, t)
            south, _, cs = self._interface(i, j - 1, 1, t)
            north, cn, _ = self._interface(i, j, 1, t)
            west.y += cw
            east.y += ce
            south.z += cs
            north.z += cn
            state = ti.Vector([self.h[i, j], self.qx[i, j], self.qz[i, j]])
            state -= dt / SEA_DX * (east - west + north - south)
            state.x = ti.max(0.0, state.x)
            state.y /= 1.0 + 0.06 * dt
            state.z /= 1.0 + 0.06 * dt
            u, v = _velocity(state.y, state.z, state.x)
            self.h_new[i, j] = state.x
            self.qx_new[i, j], self.qz_new[i, j] = state.x * u, state.x * v

    @ti.kernel
    def _begin_step(self):
        self.max_speed[None] = 0.0
        for i, j in self.h:
            h = self.h[i, j]
            self.old_h[i, j], self.old_qx[i, j], self.old_qz[i, j] = h, self.qx[i, j], self.qz[i, j]
            u, v = _velocity(self.qx[i, j], self.qz[i, j], h)
            speed = ti.abs(u) + ti.abs(v) + 2 * ti.sqrt(GRAVITY * h)
            ti.atomic_max(self.max_speed[None], speed)

    @ti.kernel
    def _stage(self):
        for i, j in self.h:
            self.h[i, j], self.qx[i, j], self.qz[i, j] = self.h_new[i, j], self.qx_new[i, j], self.qz_new[i, j]

    @ti.kernel
    def _average(self):
        for i, j in self.h:
            self.h_new[i, j] = 0.5 * (self.old_h[i, j] + self.h_new[i, j])
            self.qx_new[i, j] = 0.5 * (self.old_qx[i, j] + self.qx_new[i, j])
            self.qz_new[i, j] = 0.5 * (self.old_qz[i, j] + self.qz_new[i, j])

    @ti.func
    def _bilinear(self, field: ti.template(), x, z):
        u = ti.max(0.0, ti.min(SEA_N - 1.0, (x + SEA_HALF) / SEA_DX))
        v = ti.max(0.0, ti.min(SEA_N - 1.0, (z + SEA_HALF) / SEA_DX))
        i, j = ti.min(SEA_N - 2, ti.cast(u, ti.i32)), ti.min(SEA_N - 2, ti.cast(v, ti.i32))
        fx, fz = u - i, v - j
        return ((1 - fx) * field[i, j] + fx * field[i + 1, j]) * (1 - fz) + ((1 - fx) * field[i, j + 1] + fx * field[i + 1, j + 1]) * fz

    @ti.kernel
    def _advect_foam(self, dt: ti.f32):
        for i, j in self.h:
            u, v = _velocity(self.old_qx[i, j], self.old_qz[i, j], self.old_h[i, j])
            x, z = -SEA_HALF + i * SEA_DX - dt * u, -SEA_HALF + j * SEA_DX - dt * v
            self.foam_adv[i, j] = self._bilinear(self.foam, x, z)

    @ti.kernel
    def _commit(self, dt: ti.f32):
        foam_decay, wet_decay = ti.exp(-dt / FOAM_TAU), ti.exp(-dt / WET_TAU)
        for i, j in self.h:
            depth = self.h_new[i, j]
            dhdt = (depth - self.old_h[i, j]) / dt
            ip, im = ti.min(SEA_N - 1, i + 1), ti.max(0, i - 1)
            jp, jm = ti.min(SEA_N - 1, j + 1), ti.max(0, j - 1)
            eta = depth + self.bed[i, j]
            ee, ew, en, es = eta, eta, eta, eta
            if self.h_new[ip, j] > WET_EPS:
                ee = self.h_new[ip, j] + self.bed[ip, j]
            if self.h_new[im, j] > WET_EPS:
                ew = self.h_new[im, j] + self.bed[im, j]
            if self.h_new[i, jp] > WET_EPS:
                en = self.h_new[i, jp] + self.bed[i, jp]
            if self.h_new[i, jm] > WET_EPS:
                es = self.h_new[i, jm] + self.bed[i, jm]
            slope = ti.sqrt(((ee - ew) / (2 * SEA_DX)) ** 2 + ((en - es) / (2 * SEA_DX)) ** 2)
            u, v = _velocity(self.qx_new[i, j], self.qz_new[i, j], depth)
            ue, _ = _velocity(self.qx_new[ip, j], self.qz_new[ip, j], self.h_new[ip, j])
            uw, _ = _velocity(self.qx_new[im, j], self.qz_new[im, j], self.h_new[im, j])
            _, vn = _velocity(self.qx_new[i, jp], self.qz_new[i, jp], self.h_new[i, jp])
            _, vs = _velocity(self.qx_new[i, jm], self.qz_new[i, jm], self.h_new[i, jm])
            speed = ti.sqrt(u * u + v * v)
            compression = ti.max(0.0, -(ue - uw + vn - vs) / (2 * SEA_DX))
            activity = ti.max(compression * 3, speed / ti.sqrt(GRAVITY * ti.max(depth, 0.01)) - 0.55)
            foam = self.foam_adv[i, j] * foam_decay
            if WET_EPS < depth < 0.8 and speed > 0.12 and slope > 0.025:
                foam += dt * 1.2 * ti.min(1.0, (slope - 0.025) * 14) * ti.min(1.0, activity)
            # A stationary, separate strand field records retreating foam.
            strand = self.strand[i, j] * ti.exp(-dt / 8.0)
            if 0.0004 < depth < 0.12 and dhdt < -0.002 and speed > 0.08:
                strand += dt * foam * ti.min(1.0, -dhdt * 12)
            self.strand[i, j] = ti.min(1.0, strand)
            self.foam[i, j] = ti.max(0.0, ti.min(1.0, foam))
            wet = self.wetness[i, j] * wet_decay
            if depth > 0.0004:
                wet = 1.0
            self.wetness[i, j] = wet
            self.h[i, j], self.qx[i, j], self.qz[i, j] = depth, self.qx_new[i, j], self.qz_new[i, j]
            if depth > WET_EPS:
                ti.atomic_min(self.eta_min[None], eta)
                ti.atomic_max(self.eta_max[None], eta)

    def advance(self, steps=1):
        for _ in range(int(steps)):
            target_time = self.time + SIM_DT
            self.eta_min[None], self.eta_max[None] = 1e6, -1e6
            while target_time - self.time > 1e-9:
                self._begin_step()
                speed = max(float(self.max_speed[None]), 1e-6)
                dt = min(target_time - self.time, 0.42 * SEA_DX / speed)
                self.last_cfl = dt * speed / SEA_DX
                self._advect_foam(dt)
                self._reconstruct(self.time)
                self._step(self.time, dt)
                self._stage()
                self._reconstruct(self.time + dt)
                self._step(self.time + dt, dt)
                self._average()
                self._commit(dt)
                self.time += dt
                self.internal_steps += 1
            self.time = target_time
            self.steps += 1
        self._sync_view()

    @ti.kernel
    def _inject(self, cx: ti.f32, cz: ti.f32, radius: ti.f32, dh: ti.f32):
        """Gaussian splash added directly to the depth field."""
        gx = (cx + SEA_HALF) / SEA_DX
        gz = (cz + SEA_HALF) / SEA_DX
        reach = ti.cast(ti.ceil(3.0 * radius / SEA_DX), ti.i32)
        ci, cj = ti.cast(ti.floor(gx), ti.i32), ti.cast(ti.floor(gz), ti.i32)
        for a, b in ti.ndrange((-reach, reach + 1), (-reach, reach + 1)):
            i, j = ci + a, cj + b
            if 0 <= i < SEA_N and 0 <= j < SEA_N:
                dx, dz = (i - gx) * SEA_DX, (j - gz) * SEA_DX
                r2 = (dx * dx + dz * dz) / (radius * radius)
                if r2 < 9.0:
                    self.h[i, j] += dh * ti.exp(-r2)

    def inject(self, cx, cz, radius, dh):
        self._inject(cx, cz, radius, dh)
        self._sync_view()

    @ti.func
    def sample(self, x, z, t):
        """Reconstruct eta, then intersect it with the analytic bed.

        This subcell shoreline retains actual film thickness. The normals
        differentiate exactly the height used by the ray marcher.
        """
        outside = incident_surface(x, z, t, self.amplitude[None], self.period[None])
        surface = outside
        if ti.abs(x) <= SEA_HALF and ti.abs(z) <= SEA_HALF:
            u = ti.max(0.0, ti.min(SEA_N - 1.0, (x + SEA_HALF) / SEA_DX))
            v = ti.max(0.0, ti.min(SEA_N - 1.0, (z + SEA_HALF) / SEA_DX))
            i, j = ti.min(SEA_N - 2, ti.cast(u, ti.i32)), ti.min(SEA_N - 2, ti.cast(v, ti.i32))
            wx, dx = _cubic_weights(u - i)
            wz, dz = _cubic_weights(v - j)
            interior = ti.Vector([0.0, 0.0, 0.0])
            for a, b in ti.static(ti.ndrange(4, 4)):
                ii, jj = ti.min(SEA_N - 1, ti.max(0, i + a - 1)), ti.min(SEA_N - 1, ti.max(0, j + b - 1))
                elevation = self.eta_view[ii, jj]
                interior += elevation * ti.Vector([wx[a] * wz[b], dx[a] * wz[b] / SEA_DX,
                                                   wx[a] * dz[b] / SEA_DX])
            # An 8 m collar joins value AND derivative to the analytic sea.
            tx = ti.max(0.0, ti.min(1.0, (SEA_HALF - ti.abs(x)) / 8.0))
            tz = ti.max(0.0, ti.min(1.0, (SEA_HALF - ti.abs(z)) / 8.0))
            bx, bz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
            dbx = -ti.select(x >= 0, 1.0, -1.0) * 6 * tx * (1 - tx) / 8.0
            dbz = -ti.select(z >= 0, 1.0, -1.0) * 6 * tz * (1 - tz) / 8.0
            blend = bx * bz
            difference = interior.x - outside.x
            surface = outside + blend * (interior - outside)
            surface.y += dbx * bz * difference
            surface.z += bx * dbz * difference
        depth = ti.max(0.0, surface.x - shore_terrain(x, z))
        elevation = surface.x
        if depth <= 0.00001:
            elevation = -1e6
        return ti.Vector([elevation, surface.y, surface.z, depth])

    @ti.func
    def sample_depth(self, x, z):
        return self._bilinear(self.h_view, x, z)

    @ti.func
    def _trace(self, field: ti.template(), x, z, footprint):
        value = 0.0
        if ti.abs(x) <= SEA_HALF and ti.abs(z) <= SEA_HALF:
            value = self._bilinear(field, x, z)
            if footprint > SEA_DX:
                radius = ti.min(4.0, footprint * 0.5)
                value = 0.2 * (value + self._bilinear(field, x - radius, z)
                               + self._bilinear(field, x + radius, z)
                               + self._bilinear(field, x, z - radius)
                               + self._bilinear(field, x, z + radius))
        return value

    @ti.func
    def sample_foam(self, x, z, footprint=0.0):
        return self._trace(self.foam_view, x, z, footprint)

    @ti.func
    def sample_strand(self, x, z, footprint=0.0):
        return self._trace(self.strand_view, x, z, footprint)

    @ti.func
    def sample_wetness(self, x, z):
        return self._trace(self.wet_view, x, z, 0.0)


class ShoreCamera(OrbitCamera):
    """Orbit camera widened for the shore: long zoom range, wide pan area."""

    def zoom(self, amount):
        self.distance = float(np.clip(self.distance * math.exp(amount), 4.5, 60.0))

    def pan(self, dx, dy):
        _, _, right, _ = self.basis()
        horizontal_forward = np.array([-right[2], 0.0, right[0]])
        self.target -= (right * dx + horizontal_forward * dy) * self.distance
        self.target[0] = float(np.clip(self.target[0], -24.0, 24.0))
        self.target[1] = float(np.clip(self.target[1], -0.5, 2.0))
        self.target[2] = float(np.clip(self.target[2], -24.0, 24.0))


@ti.data_oriented
class ShoreRenderer(SunsetOceanRenderer):
    """Demo 03 renderer whose sea surface comes from the shallow-water grid."""

    def __init__(self, width, height, samples=4):
        super().__init__(width, height, samples)
        self.sea = ShallowSea()
        self.view_eye = ti.Vector.field(3, ti.f32, shape=())
        self.view_angle = ti.field(ti.f32, shape=())
        self.view_eye[None], self.view_angle[None] = [0, 4, 10], 0.001
        self.detail_modes = ti.Vector.field(4, ti.f32, shape=24)
        self.detail_omega = ti.field(ti.f32, shape=24)
        rng = np.random.default_rng(61)
        direction = rng.uniform(-math.pi, math.pi, 24)
        wavelength = rng.uniform(0.9, 3.4, 24)
        k = 2 * math.pi / wavelength
        modes = np.column_stack((k * np.cos(direction), k * np.sin(direction),
                                 0.008 / (k * math.sqrt(12)), rng.uniform(0, 2 * math.pi, 24)))
        self.detail_modes.from_numpy(modes.astype(np.float32))
        self.detail_omega.from_numpy(np.sqrt(GRAVITY * k).astype(np.float32))
        self.water_controls[None] = [0.06, 0.5, 1.0, 4.0]
        # Sand is matte, not semi-gloss: a rougher terrain keeps the bright
        # sky out of the beach so the swash band and ripples stay readable.
        self.materials[6] = [0.86, 0.0, 0.04]
        # Neutral caustic gate: shade_surface multiplies the terrain sun by
        # 1 - foam + foam * caustic_density; an all-ones map keeps it at 1.
        self.caustic_map.from_numpy(np.ones((128, 96), dtype=np.float32))

    def build_static(self):
        """A few driftwood logs on the beach; no courtyard or lighthouse."""
        def box(center, extent, color, material=1, bevel=0.05):
            i = self.boxes
            self.box_center[i], self.box_extent[i] = center, extent
            self.box_color[i], self.box_material[i] = color, material
            self.box_bevel[i] = min(bevel, min(extent) * 0.8)
            self.boxes += 1

        box((11.5, 0.42, 3.0), (1.6, 0.11, 0.16), (0.36, 0.26, 0.16), 1, 0.05)
        box((9.8, 0.36, -2.2), (1.1, 0.09, 0.13), (0.42, 0.32, 0.20), 1, 0.045)
        box((13.0, 0.55, 0.8), (0.9, 0.10, 0.12), (0.33, 0.24, 0.14), 1, 0.04)

    @ti.func
    def wave_amplitude(self):
        half_range = 0.5 * (self.sea.eta_max[None] - self.sea.eta_min[None])
        return half_range + 0.5 * self.sea.amplitude[None] + 0.02

    @ti.func
    def wave_height(self, x, z, clock):
        return self.sea.sample(x, z, clock).x

    @ti.func
    def wave_surface(self, x, z, clock):
        surface = self.sea.sample(x, z, clock)
        return surface.x, ti.Vector([-surface.y, 1.0, -surface.z]).normalized()

    @ti.func
    def wave_bounds(self):
        amplitude = self.sea.amplitude[None]
        lo = ti.min(self.sea.eta_min[None], -1.02 * amplitude) - 0.02
        hi = ti.max(self.sea.eta_max[None], 1.02 * amplitude) + 0.02
        # Catmull-Rom can overshoot the node range by up to 28.125% of it;
        # the march slab must contain the interpolant, not just the nodes.
        padding = (hi - lo) * 0.28125
        return ti.Vector([lo - padding, hi + padding])

    @ti.func
    def water_depth_limit(self):
        return 12.0

    @ti.func
    def water_body(self, color, thickness, clarity, wave_y, normal, lighting):
        transmission = ti.exp(-ti.Vector([0.34, 0.12, 0.065]) * thickness / clarity)
        # Orientation-dependent illumination, not a height tint that would
        # paint solver cells onto the surface. The bulk color stays in the
        # tropical shallow green-cyan range over the sandy bed.
        illumination = 0.75 + 0.25 * ti.max(0.0, normal.dot(self.sun_direction(lighting)))
        bulk = ti.Vector([0.020, 0.150, 0.135]) * illumination
        return color * transmission + bulk * (1.0 - transmission)

    @ti.func
    def water_ray_offset(self, p, normal):
        film = ti.max(0.0, p.y - shore_terrain(p.x, p.z))
        return ti.min(0.002, ti.max(0.000001, film * 0.15))

    @ti.func
    def _foam_amount(self, p, footprint):
        mobile = self.sea.sample_foam(p.x, p.z, footprint)
        strand = self.sea.sample_strand(p.x, p.z, footprint)
        film = ti.max(0.0, p.y - shore_terrain(p.x, p.z))
        return ti.min(1.0, (mobile + strand * (1 - self.smooth(0.01, 0.12, film))) * self.water_controls[None].z)

    @ti.func
    def foam_filtered(self, p, clock, lighting, footprint):
        return ti.Vector([0.93, 0.94, 0.91]) * self._foam_amount(p, footprint)

    @ti.func
    def composite_foam(self, color, p, clock, lighting, footprint):
        amount = self._foam_amount(p, footprint)
        coverage = amount * amount * 0.85
        illumination = 0.55 + 0.30 * lighting
        foam_color = ti.Vector([0.93, 0.94, 0.91]) * illumination
        return color * (1 - coverage) + foam_color * coverage

    @ti.func
    def foam(self, p, clock, lighting):
        return self.foam_filtered(p, clock, lighting, 0.0)

    @ti.func
    def detail_gradient(self, x, z, clock, footprint):
        gradient = ti.Vector([0.0, 0.0])
        depth = self.sea.sample_depth(x, z)
        fade_depth = self.smooth(0.01, 0.30, depth)
        for band in range(24):
            mode = self.detail_modes[band]
            magnitude = ti.sqrt(mode.x * mode.x + mode.y * mode.y)
            wavelength = 2 * math.pi / magnitude
            fade = 1 - self.smooth(wavelength * 0.12, wavelength * 0.5, footprint)
            phase = mode.x * x + mode.y * z - self.detail_omega[band] * clock + mode.w
            gradient += mode.z * ti.cos(phase) * ti.Vector([mode.x, mode.y]) * fade
        return gradient * self.water_controls[None].y * fade_depth

    @ti.func
    def sand_sample(self, x, z, footprint):
        """Rotated quintic value noise, with analytic slopes and pixel filtering."""
        result = ti.Vector([0.0, 0.0, 0.0])
        for octave in ti.static(range(3)):
            scale, angle, weight = 0.65, 0.37, 0.55
            if ti.static(octave == 1):
                scale, angle, weight = 2.1, 1.14, 0.30
            elif ti.static(octave == 2):
                scale, angle, weight = 6.0, -0.63, 0.15
            c, sn = ti.cos(angle), ti.sin(angle)
            uv = ti.Vector([c * x - sn * z, sn * x + c * z]) * scale + 17.3 * octave
            cell, t = ti.floor(uv), uv - ti.floor(uv)
            f = t * t * t * (t * (6 * t - 15) + 10)
            df = 30 * t * t * (t - 1) * (t - 1)
            a, b = self.hash2(cell), self.hash2(cell + ti.Vector([1., 0.]))
            c0, d = self.hash2(cell + ti.Vector([0., 1.])), self.hash2(cell + ti.Vector([1., 1.]))
            value = (a * (1 - f.x) + b * f.x) * (1 - f.y) + (c0 * (1 - f.x) + d * f.x) * f.y
            gx = ((b - a) * (1 - f.y) + (d - c0) * f.y) * df.x
            gz = ((c0 - a) * (1 - f.x) + (d - b) * f.x) * df.y
            fade = 1 - self.smooth(0.15, 0.65, footprint * scale)
            result.x += weight * (0.5 + (value - 0.5) * fade)
            result.y += weight * scale * (c * gx + sn * gz) * fade
            result.z += weight * scale * (-sn * gx + c * gz) * fade
        return result

    @ti.func
    def sand_noise(self, x, z):
        return self.sand_sample(x, z, 0.0).x

    @ti.func
    def ground_footprint(self, p):
        delta = p - self.view_eye[None]
        distance = ti.sqrt(delta.dot(delta))
        grad = shore_terrain_grad(p.x, p.z)
        normal = ti.Vector([-grad.x, 1., -grad.y]).normalized()
        incidence = ti.abs(normal.dot(delta / ti.max(distance, 0.001)))
        return distance * self.view_angle[None] / ti.max(0.15, incidence)

    @ti.func
    def surface_properties(self, p, material, properties):
        result = properties
        if material == 6:
            wet = self.smooth(0.20, 0.98, self.sea.sample_wetness(p.x, p.z))
            reef = reef_factor(p.x, p.z)
            roughness = 0.86 * (1 - wet) + 0.38 * wet
            roughness = roughness * (1 - reef) + (0.76 * (1 - wet) + 0.42 * wet) * reef
            strand = self.sea.sample_strand(p.x, p.z) * self.water_controls[None].z
            submerged = self.smooth(0.002, 0.02, self.sea.sample_depth(p.x, p.z))
            # A submerged bed must not add a second air/water-like highlight.
            # Sand/rock versus water has much lower dielectric contrast.
            roughness = roughness * (1 - submerged) + (0.86 - 0.10 * reef) * submerged
            f0 = 0.04 * (1 - submerged) + 0.003 * submerged
            result = ti.Vector([ti.min(0.92, roughness + strand * 0.3), 0.0, f0])
        return result

    @ti.func
    def surface_sun_visibility(self, p, normal, material, lighting, detail):
        return self.visibility(p, normal, self.sun_direction(lighting), detail)

    @ti.func
    def floor_material(self, p, clock, amplitude):
        x, z = p.x, p.z
        footprint = self.ground_footprint(p)
        wet = self.smooth(0.20, 0.98, self.sea.sample_wetness(x, z))
        strand = self.sea.sample_strand(x, z, footprint) * self.water_controls[None].z
        noise = self.sand_sample(x, z, footprint)
        submerged = self.smooth(0.02, 0.3, self.sea.sample_depth(x, z))
        grain_strength = 0.12 * (1 - submerged) + 0.04 * submerged
        base = ti.Vector([0.48, 0.40, 0.29]) * (1 + grain_strength * (noise.x - 0.5))
        base *= (1 - wet) + ti.Vector([0.50, 0.46, 0.43]) * wet
        reef = reef_factor(x, z)
        rock = ti.Vector([0.31, 0.33, 0.35]) * (0.92 + 0.16 * noise.x)
        base = base * (1 - reef) + rock * reef
        wash = ti.min(1.0, strand) * 0.75
        base = base * (1 - wash) + ti.Vector([0.76, 0.78, 0.73]) * wash
        grad = shore_terrain_grad(x, z)
        relief = 0.012 * (1 - submerged) + 0.002 * submerged
        normal = ti.Vector([-grad.x - noise.y * relief, 1.0, -grad.y - noise.z * relief]).normalized()
        return base, normal

    def draw(self, camera, clock, clarity=1.4, lighting=0.0, exposure=1.05):
        self.view_eye[None] = camera.basis()[0]
        self.view_angle[None] = 2 * math.tan(math.radians(camera.fov / 2)) / self.height
        super().draw(camera, clock, clarity, lighting, exposure)

    @ti.func
    def terrain_hit(self, origin, direction, limit):
        near, far = 0.000001, limit
        if ti.abs(direction.y) > 1e-8:
            a, b = (TERRAIN_Y_HI - origin.y) / direction.y, (TERRAIN_Y_LO - origin.y) / direction.y
            near, far = ti.max(near, ti.min(a, b)), ti.min(far, ti.max(a, b))
        elif origin.y < TERRAIN_Y_LO or origin.y > TERRAIN_Y_HI:
            far = -1.0
        hit = 1e6
        if near <= far:
            current = near
            point = origin + direction * current
            bed = shore_terrain(point.x, point.z)
            if point.y <= bed:
                hit = current
            else:
                count = 0
                horizontal = ti.sqrt(direction.x ** 2 + direction.z ** 2)
                while hit == 1e6 and current < far and count < MARCH_MAX_STEPS:
                    error = point.y - bed
                    precision = 0.000001 + 0.0000002 * current
                    if error <= precision:
                        hit = current
                    else:
                        # Outside every 4-sigma reef support, the planar beach
                        # and Gaussian tails have slope below 0.06. Never let
                        # a fast step enter a support without switching bounds.
                        clearance = 1e6
                        for reef in ti.static(REEFS):
                            distance = ti.sqrt((point.x - reef[0]) ** 2 + (point.z - reef[1]) ** 2)
                            clearance = ti.min(clearance, distance - 4.0 * reef[3])
                        slope_bound = MAX_BED_SLOPE
                        if clearance > 0.0001:
                            slope_bound = 0.06
                        decay = -direction.y + slope_bound * horizontal
                        step = far - current
                        if decay > 1e-8:
                            # Shadow/refraction rays can start micrometres above
                            # the surface: a fixed minimum step would cross it.
                            step = ti.max(0.000001, ti.min(0.9 * error / decay, MARCH_MIN_STEP + MARCH_GROWTH * current))
                        if clearance > 0.0001 and horizontal > 1e-8:
                            step = ti.min(step, 0.9 * clearance / horizontal)
                        next_t = ti.min(current + step, far)
                        point = origin + direction * next_t
                        bed = shore_terrain(point.x, point.z)
                        if point.y <= bed:
                            lo, hi = current, next_t
                            for _ in range(BISECT_STEPS):
                                mid = 0.5 * (lo + hi)
                                mid_point = origin + direction * mid
                                if mid_point.y <= shore_terrain(mid_point.x, mid_point.z):
                                    hi = mid
                                else:
                                    lo = mid
                            hit = 0.5 * (lo + hi)
                        else:
                            current = next_t
                    count += 1
        return hit

    @ti.func
    def geometry(self, origin, direction):
        closest = self.terrain_hit(origin, direction, 600.0)
        material = -1
        normal, color = ti.Vector([0., 1., 0.]), ti.Vector([0.5, 0.5, 0.5])
        if closest < 1e5:
            hit_point = origin + direction * closest
            grad = shore_terrain_grad(hit_point.x, hit_point.z)
            normal = ti.Vector([-grad.x, 1., -grad.y]).normalized()
            material = 6
        for i in range(self.boxes):
            t, n = self.rounded_box_hit(origin, direction, self.box_center[i], self.box_extent[i], self.box_bevel[i])
            if t < closest:
                closest, normal, color, material = t, n, self.box_color[i], self.box_material[i]
        return closest, normal, color, material

    @ti.func
    def occlusion_distance(self, origin, direction, limit):
        closest = ti.min(limit, self.terrain_hit(origin, direction, limit))
        for i in range(self.boxes):
            bound, _ = self.box_hit(origin, direction, self.box_center[i], self.box_extent[i])
            inside = (ti.abs(origin - self.box_center[i]) < self.box_extent[i]).all()
            if bound < closest or inside:
                t, _ = self.rounded_box_hit(origin, direction, self.box_center[i], self.box_extent[i], self.box_bevel[i])
                closest = ti.min(closest, t)
        return closest


def water_pick(camera, u, v, width, height):
    """Intersect the cursor ray (normalized, v from the top) with y = 0."""
    eye, forward, right, up = camera.basis()
    scale = math.tan(math.radians(camera.fov / 2.0))
    px = (2.0 * u - 1.0) * width / height * scale
    py = (2.0 * (1.0 - v) - 1.0) * scale
    direction = forward + right * px + up * py
    direction /= np.linalg.norm(direction)
    point = None
    if direction[1] < -1e-5:
        t = -eye[1] / direction[1]
        hit = eye + direction * t
        if abs(hit[0]) < SEA_HALF - 0.5 and abs(hit[2]) < SEA_HALF - 0.5:
            point = (float(hit[0]), float(hit[2]))
    return point


def simulation_budget(accumulator, elapsed, speed):
    """Catch up ordinary slow frames; cap long stalls to avoid a spiral."""
    budget = min(accumulator + min(elapsed, 0.25) * speed, 8 * SIM_DT)
    steps = int((budget + 1e-10) / SIM_DT)
    return steps, max(0.0, budget - steps * SIM_DT)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 06: shallow-water shore with beach and reefs")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=4, help="Spatial samples per pixel (default: 4)")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="sunset")
    parser.add_argument("--shadow-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--water-roughness", type=float, default=0.06)
    parser.add_argument("--reflection-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-water-detail", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--sea", choices=("auto", "calm", "surf", "storm"), default="auto",
                        help="Sea preset; auto uses the amplitude/period values below")
    parser.add_argument("--wave-amplitude", type=float, default=0.3, help="Incident wave amplitude in metres")
    parser.add_argument("--wave-period", type=float, default=10.0, help="Incident wave period in seconds")
    parser.add_argument("--preset", choices=("overview", "waterline", "top"), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_06.png"))
    parser.add_argument("--time", type=float, default=20.0, help="Simulation warmup in seconds before the first frame")
    parser.add_argument("--frames", type=int, default=1, help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0, help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache for troubleshooting")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0 or not math.isfinite(args.time):
        parser.error("width/height must be >= 32, frames >= 1, and time finite")
    if not math.isfinite(args.water_roughness) or not 0.0 <= args.water_roughness <= 0.45:
        parser.error("water-roughness must be finite and between 0 and 0.45")
    if not math.isfinite(args.wave_amplitude) or not 0.05 <= args.wave_amplitude <= 0.6:
        parser.error("wave-amplitude must be finite and between 0.05 and 0.6")
    if not math.isfinite(args.wave_period) or not 6.0 <= args.wave_period <= 16.0:
        parser.error("wave-period must be finite and between 6 and 16")
    return args


def sea_parameters(args):
    if args.sea == "auto":
        return args.wave_amplitude, args.wave_period
    return SHORE_PRESETS_SEA[args.sea]


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = ShoreRenderer(args.width, args.height, args.samples)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 1.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = ShoreCamera()
    apply_preset(camera, args.preset)
    amplitude, period = sea_parameters(args)
    renderer.sea.set_wave(amplitude, period)
    wave = [amplitude, period]
    renderer._wave_key = (amplitude, period)
    speed = 1.0
    lighting = 0.0 if args.lighting == "sunset" else 1.0
    clarity, exposure = 1.4, 1.05
    clock = 0.0
    sim_accumulator = 0.0

    def advance(substeps):
        nonlocal clock
        if substeps > 0:
            renderer.sea.advance(substeps)
            clock = renderer.sea.time

    # Warmup lets the swell cross the domain before the first visible frame.
    warmup_steps = int(max(0.0, args.time) / SIM_DT)
    advance(warmup_steps)

    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            frame_steps, sim_accumulator = simulation_budget(sim_accumulator, 1.0 / 60.0, 1.0)
            advance(frame_steps)
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
    window = ti.ui.Window("06 / Shore Reef", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    sim_accumulator, frame_ms, window_frames = 0.0, 0.0, 1
    drag_in_scene, click_start, click_camera_moved = False, None, False
    print("Click water: splash | Drag LMB orbit | RMB pan | W/S zoom | 1/2/3 views | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
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
            elif key == "h":
                show_panel = not show_panel
            elif key in ("1", "2", "3"):
                apply_preset(camera, {"1": "overview", "2": "waterline", "3": "top"}[key])
                auto_orbit = False
            elif key == "r":
                apply_preset(camera, "overview")
                renderer.samples = args.samples
                renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 1.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                amplitude, period = sea_parameters(args)
                wave = [amplitude, period]
                renderer.sea.reset()
                renderer.sea.set_wave(amplitude, period)
                renderer._wave_key = (amplitude, period)
                lighting = 0.0 if args.lighting == "sunset" else 1.0
                clarity, exposure, clock = 1.4, 1.05, 0.0
                sim_accumulator = 0.0
                paused, auto_orbit = False, False
            elif key == "p":
                save_frame = True
            elif key in (ti.ui.LMB, ti.ui.RMB):
                drag_in_scene = not in_panel
        if not window.running:
            break
        # A short click splashes; a longer drag keeps orbiting.
        if drag_in_scene and window.is_pressed(ti.ui.LMB):
            if click_start is None:
                click_start, click_camera_moved = mouse.copy(), False
            elif not click_camera_moved and np.linalg.norm(mouse - click_start) > 0.012:
                click_camera_moved = True
        elif click_start is not None:
            if drag_in_scene and not click_camera_moved:
                point = water_pick(camera, click_start[0], 1.0 - click_start[1], args.width, args.height)
                if point is not None:
                    renderer.sea.inject(point[0], point[1], 0.8, 0.3)
            click_start = None
        if previous_mouse is not None and drag_in_scene and (click_camera_moved or window.is_pressed(ti.ui.RMB)):
            delta = mouse - previous_mouse
            if window.is_pressed(ti.ui.LMB):
                camera.orbit(*delta)
                auto_orbit = False
            elif window.is_pressed(ti.ui.RMB):
                camera.pan(*delta)
                auto_orbit = False
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
            with gui.sub_window("SHORE / FINISH", 0.72, 0.335, 0.26, 0.52):
                controls = renderer.water_controls[None]
                controls[0] = gui.slider_float("Water roughness", controls[0], 0.0, 0.45)
                controls[1] = gui.slider_float("Fine waves", controls[1], 0.0, 2.0)
                controls[2] = gui.slider_float("Foam amount", controls[2], 0.0, 2.0)
                controls[3] = 4 if gui.checkbox("4x reflections", controls[3] == 4) else 1
                renderer.water_controls[None] = controls
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
            with gui.sub_window("BEACH / 06", 0.02, 0.025, 0.30, 0.78):
                gui.text("Shallow water / finite volume / shore")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                if gui.button("Single step [N]"):
                    step = True
                wave[0] = gui.slider_float("Wave amplitude", wave[0], 0.05, 0.6)
                wave[1] = gui.slider_float("Wave period", wave[1], 6.0, 16.0)
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
                gui.text("Click water to splash")
                gui.text("Sea presets: --sea calm/surf/storm")
                gui.text("Drag LMB orbit / RMB pan")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        # The incident wave follows the sliders on the host.
        current = (float(wave[0]), float(wave[1]))
        if current != renderer._wave_key:
            renderer.sea.set_wave(*current)
            renderer._wave_key = current
        if paused:
            if step:
                advance(1)
        else:
            substeps, sim_accumulator = simulation_budget(sim_accumulator, dt, speed)
            advance(substeps)
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
