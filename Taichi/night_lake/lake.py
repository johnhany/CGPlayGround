"""Lake water: small wind waves plus a rain-ripple wave-equation field.

LakeWaves sums a fixed set of low-wavenumber sine components with analytic
height and gradient; amplitudes are a few centimetres, directed mostly
toward the camera along the lake. RainRipples runs a leapfrog
finite-difference wave equation on its own grid covering the water ahead
of the camera. Rain arrives as landing events from the streak simulation
(event-driven: streaks fall, land, and their landing points inject
velocity impulses with zero net volume); bulk damping and an absorbing
border dissipate outgoing rings. The scene marches the sum of both height
fields and shades with the night environment.
"""

import math

import numpy as np
import taichi as ti

__all__ = ["LakeWaves", "RainRipples", "RIPPLE_NX", "RIPPLE_NZ",
           "LAKE_MIN_X", "LAKE_MAX_X", "LAKE_MIN_Z", "LAKE_MAX_Z", "RIPPLE_RADIUS",
           "MAX_RAIN_EVENTS", "BASE_INJECTION_STRENGTH"]

# Ripple grid over the water ahead of the camera; cells are square. The
# grid covers a 48 m by 40 m window in front of the camera at 10 cm cells,
# dense enough to resolve raindrop rings; farther water leans on the
# filtered procedural wave normal instead.
LAKE_MIN_X, LAKE_MAX_X = -24.0, 24.0
LAKE_MIN_Z, LAKE_MAX_Z = 8.0, 48.0
RIPPLE_NX, RIPPLE_NZ = 480, 400
CELL_X = (LAKE_MAX_X - LAKE_MIN_X) / RIPPLE_NX
CELL_Z = (LAKE_MAX_Z - LAKE_MIN_Z) / RIPPLE_NZ
RIPPLE_RADIUS = 0.25
# Injection impulses are velocity kicks; the CELL_X^2 factor in the
# disturbance keeps this base scale independent of the grid resolution.
BASE_INJECTION_STRENGTH = 1.1
SUBSTEP_DT = 1.0 / 240.0
WAVE_SPEED = 0.8
DAMPING = 0.996
# Cells within this distance of the border get extra damping so outgoing
# rings die instead of reflecting back into the field.
ABSORB_CELLS = 12
# At most one landing event per streak can arrive in a frame; the staging
# buffer matches the streak table capacity.
MAX_RAIN_EVENTS = 256
WAVE_COUNT = 4
WAVE_AMP_TOTAL = 0.035


@ti.data_oriented
class LakeWaves:
    """Small-amplitude directional wave set with analytic derivatives.

    Wavenumbers follow a descending ladder with seeded jitter; directions
    cluster around the wind blowing along the lake toward the camera.
    """

    def __init__(self, seed=17):
        rng = np.random.default_rng(seed)
        ladder = 2.0 * math.pi / np.array([6.2, 3.6, 2.2, 1.35])
        ladder *= rng.uniform(0.88, 1.12, WAVE_COUNT)
        k = np.sort(ladder)[::-1]
        weights = k ** -1.5 * rng.uniform(0.8, 1.2, WAVE_COUNT)
        amp = WAVE_AMP_TOTAL * weights / weights.sum()
        # Wave travel is mostly toward the camera: wavevectors point -Z.
        base = math.pi
        angles = base + (rng.random(WAVE_COUNT) * 2.0 - 1.0) * 0.45
        omega = np.sqrt(9.81 * k)
        phase = rng.random(WAVE_COUNT) * 2.0 * math.pi
        direction = np.stack([np.sin(angles), np.cos(angles)], axis=1)
        self.dir = ti.Vector.field(2, ti.f32, shape=WAVE_COUNT)
        self.amp = ti.field(ti.f32, shape=WAVE_COUNT)
        self.k = ti.field(ti.f32, shape=WAVE_COUNT)
        self.omega = ti.field(ti.f32, shape=WAVE_COUNT)
        self.phase = ti.field(ti.f32, shape=WAVE_COUNT)
        self.dir.from_numpy(direction.astype(np.float32))
        self.amp.from_numpy(amp.astype(np.float32))
        self.k.from_numpy(k.astype(np.float32))
        self.omega.from_numpy(omega.astype(np.float32))
        self.phase.from_numpy(phase.astype(np.float32))
        self.table = {"dir": direction, "amp": amp, "k": k,
                      "omega": omega, "phase": phase}

    @ti.func
    def height(self, x, z, t):
        y = 0.0
        for i in ti.static(range(WAVE_COUNT)):
            f = self.k[i] * (self.dir[i].x * x + self.dir[i].y * z) - self.omega[i] * t + self.phase[i]
            y += self.amp[i] * ti.sin(f)
        return y

    @ti.func
    def gradient(self, x, z, t):
        gx, gz = 0.0, 0.0
        for i in ti.static(range(WAVE_COUNT)):
            f = self.k[i] * (self.dir[i].x * x + self.dir[i].y * z) - self.omega[i] * t + self.phase[i]
            c = self.amp[i] * self.k[i] * ti.cos(f)
            gx += c * self.dir[i].x
            gz += c * self.dir[i].y
        return ti.Vector([gx, gz])


@ti.data_oriented
class RainRipples:
    """2D wave equation rain-ripple field with an absorbing border.

    Leapfrog finite difference on three rotating buffers at a fixed
    substep. Rain lands as explicit events from the streak simulation;
    each impact deposits a velocity impulse with zero net volume. Bulk
    damping plus a border ramp absorbs outgoing rings.
    """

    def __init__(self, seed=17):
        if abs(CELL_X - CELL_Z) > 1e-9:
            raise ValueError("Ripple cells must be square for an isotropic wave speed")
        self.h = ti.field(ti.f32, shape=(RIPPLE_NX, RIPPLE_NZ))
        self.h_prev = ti.field(ti.f32, shape=(RIPPLE_NX, RIPPLE_NZ))
        self.h_next = ti.field(ti.f32, shape=(RIPPLE_NX, RIPPLE_NZ))
        self.drop_count = ti.field(ti.f32, shape=())
        self.drop_energy = ti.field(ti.f32, shape=())
        self.rng = np.random.default_rng(seed + 1)
        # Event staging: one packed host-to-device copy per frame, unpacked
        # on the device by inject_events_kernel.
        self.event_stage = ti.Vector.field(3, ti.f32, shape=MAX_RAIN_EVENTS)
        self.drop_count[None] = 0.0
        self.drop_energy[None] = 0.0

    def event_strengths(self, count, base=1.0):
        """Fixed-seed per-drop jitter around the caller's base strength."""
        return self.rng.uniform(0.7, 1.3, count).astype(np.float32) * base

    def inject_events(self, positions, strengths):
        """Deposit velocity impulses at explicit landing points.

        positions is an (n, 2) numpy array of water-surface points;
        strengths the per-drop impulse scale. Landings outside the grid
        windows are skipped by the bounds check inside the disturbance.
        """
        count = min(int(len(positions)), MAX_RAIN_EVENTS)
        if count <= 0:
            return
        data = np.zeros((MAX_RAIN_EVENTS, 3), dtype=np.float32)
        data[:count, 0] = positions[:count, 0]
        data[:count, 1] = positions[:count, 1]
        data[:count, 2] = strengths[:count]
        self.event_stage.from_numpy(data)
        self.inject_events_kernel(count)

    @ti.kernel
    def inject_events_kernel(self, count: ti.i32):
        for k in range(count):
            event = self.event_stage[k]
            self.disturbance(event.x, event.y, RIPPLE_RADIUS, event.z, True)
            ti.atomic_add(self.drop_count[None], 1.0)
            ti.atomic_add(self.drop_energy[None], event.z)

    @ti.func
    def disturbance(self, cx, cz, radius, strength, velocity):
        # Deposit a discrete negative Laplacian of a smooth Gaussian so the
        # impulse carries no net volume; support is three radii wide.
        gx = (cx - LAKE_MIN_X) / CELL_X - 0.5
        gz = (cz - LAKE_MIN_Z) / CELL_Z - 0.5
        reach = ti.cast(ti.ceil(3 * radius / CELL_X), ti.i32)
        ci, cj = ti.cast(ti.floor(gx), ti.i32), ti.cast(ti.floor(gz), ti.i32)
        for a, b in ti.ndrange((-reach, reach + 1), (-reach, reach + 1)):
            i, j = ci + a, cj + b
            if 0 <= i < RIPPLE_NX and 0 <= j < RIPPLE_NZ:
                dx, dz = (i - gx) * CELL_X, (j - gz) * CELL_Z
                r2 = (dx * dx + dz * dz) / (radius * radius)
                value = 0.0
                if r2 < 9.0:
                    value = strength * (ti.exp(-r2) - ti.exp(-9.0) * (10.0 - r2)) * radius * radius / (4 * CELL_X * CELL_X)
                for tap in ti.static(range(5)):
                    ii, jj, weight = i, j, 4.0
                    if ti.static(tap == 1):
                        ii, weight = ti.max(0, i - 1), -1.0
                    elif ti.static(tap == 2):
                        ii, weight = ti.min(RIPPLE_NX - 1, i + 1), -1.0
                    elif ti.static(tap == 3):
                        jj, weight = ti.max(0, j - 1), -1.0
                    elif ti.static(tap == 4):
                        jj, weight = ti.min(RIPPLE_NZ - 1, j + 1), -1.0
                    delta = value * weight
                    if velocity:
                        ti.atomic_add(self.h_prev[ii, jj], -delta * SUBSTEP_DT)
                    else:
                        ti.atomic_add(self.h[ii, jj], delta)
                        ti.atomic_add(self.h_prev[ii, jj], delta)

    def advance(self, dt):
        """Advance the field by dt of wave-equation time.

        Rain enters only through inject_events, so the stepping rate is
        independent of the weather.
        """
        self.step_n(max(1, int(round(dt / SUBSTEP_DT))))

    @ti.kernel
    def step(self, courant2: ti.f32, damping: ti.f32):
        for i, j in self.h:
            center = self.h[i, j]
            left = self.h[ti.max(0, i - 1), j]
            right = self.h[ti.min(RIPPLE_NX - 1, i + 1), j]
            down = self.h[i, ti.max(0, j - 1)]
            up = self.h[i, ti.min(RIPPLE_NZ - 1, j + 1)]
            laplacian = left + right + down + up - 4.0 * center
            edge = ti.min(ti.min(i, RIPPLE_NX - 1 - i), ti.min(j, RIPPLE_NZ - 1 - j))
            absorb = 0.90 + 0.10 * ti.min(1.0, edge / ABSORB_CELLS)
            value = center + damping * absorb * (center - self.h_prev[i, j]) + courant2 * laplacian
            # Fresh impulses overshoot before dispersing; bound the spike so
            # a recent drop reads as a rain ring, not a splash column.
            self.h_next[i, j] = ti.max(-0.04, ti.min(0.04, value))

    @ti.kernel
    def commit_step(self):
        for i, j in self.h:
            self.h_prev[i, j] = self.h[i, j]
            self.h[i, j] = self.h_next[i, j]

    def step_n(self, substeps):
        courant2 = (WAVE_SPEED * SUBSTEP_DT / CELL_X) ** 2
        for _ in range(substeps):
            self.step(courant2, DAMPING)
            self.commit_step()

    @ti.func
    def sample(self, x, z):
        """Bilinear height with the cell-exact bilinear gradient."""
        u = ti.max(0.0, ti.min(RIPPLE_NX - 1.0, (x - LAKE_MIN_X) / CELL_X - 0.5))
        v = ti.max(0.0, ti.min(RIPPLE_NZ - 1.0, (z - LAKE_MIN_Z) / CELL_Z - 0.5))
        i0, j0 = ti.min(RIPPLE_NX - 2, ti.cast(u, ti.i32)), ti.min(RIPPLE_NZ - 2, ti.cast(v, ti.i32))
        fu, fv = u - i0, v - j0
        h00 = self.h[i0, j0]
        h10 = self.h[i0 + 1, j0]
        h01 = self.h[i0, j0 + 1]
        h11 = self.h[i0 + 1, j0 + 1]
        height = (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv
        dhdx = ((h10 - h00) * (1.0 - fv) + (h11 - h01) * fv) / CELL_X
        dhdz = ((h01 - h00) * (1.0 - fu) + (h11 - h10) * fu) / CELL_Z
        return ti.Vector([height, dhdx, dhdz])

    def stats(self):
        return int(self.drop_count[None]), float(self.drop_energy[None])

    def heights(self):
        return self.h.to_numpy()
