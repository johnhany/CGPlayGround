"""Rain: screen-space streak splats and near-field splash bursts.

RainStreaks keeps a fixed-capacity CPU particle table advected with plain
Euler steps. Spawns are Poisson births spread through the air volume above
the lake, each drop carrying STREAKS_PER_DROP streaks; a streak dies when
it reaches the terrain or the water surface, and its landing point is
returned as an event for the caller to route (ripple impulses on water,
splash crowns on land near the camera). Rendering projects each streak to
a screen-space line, depth-tested against the scene hit buffer, and
splats a fading segment into an accumulation buffer composited over the
HDR frame after fog.

SplashBursts is a strictly capped pool of crown droplets triggered when a
streak lands close to the camera; far landings never spawn particles.
"""

import numpy as np
import taichi as ti

from night_lake.terrain import terrain_height_np

__all__ = ["RAIN_SEED", "MAX_STREAKS", "MAX_SPLASH_DROPS", "STREAKS_PER_DROP",
           "SPAWN_MIN_X", "SPAWN_MAX_X", "SPAWN_MIN_Z", "SPAWN_MAX_Z",
           "RainStreaks", "SplashBursts"]

RAIN_SEED = 17
# Streak table: rain rate 40/s times four streaks per drop with a fall
# time near a second keeps the steady-state population around 150.
MAX_STREAKS = 256
# Splash pool: only near-camera crowns, so a small cap suffices.
MAX_SPLASH_DROPS = 160
STREAKS_PER_DROP = 4
# Splash triggers only inside this radius of the camera; the raised value
# keeps crowns reachable from the elevated production camera.
SPLASH_RANGE = 18.0
# Streaks are born throughout this volume above the lake and shore; the
# x/z box matches the ripple grid in x and extends past it in z so rain
# lands along the whole visible water and beach strip.
SPAWN_MIN_X, SPAWN_MAX_X = -24.0, 24.0
SPAWN_MIN_Y, SPAWN_MAX_Y = 7.0, 13.0
SPAWN_MIN_Z, SPAWN_MAX_Z = 6.0, 60.0
# Terminal fall speed and wind drift (fraction of the wind control).
FALL_SPEED = (7.5, 10.5)
WIND_DRIFT = 0.3
# One streak splat at most this many pixels long.
MAX_SPLAT_STEPS = 96
# Cool moonlit tint shared by streaks and splash crowns.
RAIN_TINT = (0.58, 0.65, 0.78)


def surface_height_np(x, z):
    """Ground or water level at a landing point (water sits at y = 0)."""
    ground = float(terrain_height_np(np.float32(x), np.float32(z)))
    return max(ground, 0.0)


@ti.data_oriented
class RainStreaks:
    """Fixed-capacity rain streaks spawned from the ripple landing table."""

    def __init__(self, width, height, seed=RAIN_SEED):
        self.width, self.height = width, height
        self.rng = np.random.default_rng(seed)
        # CPU advection state.
        self.pos = np.zeros((MAX_STREAKS, 3), dtype=np.float64)
        self.prev = np.zeros((MAX_STREAKS, 3), dtype=np.float64)
        self.vel = np.zeros((MAX_STREAKS, 3), dtype=np.float64)
        self.alive = np.zeros(MAX_STREAKS, dtype=bool)
        self.gain = np.zeros(MAX_STREAKS, dtype=np.float64)
        # Bookkeeping for conservation checks.
        self.spawned = 0
        self.landed = 0
        # GPU splat state.
        self.streak_head = ti.Vector.field(3, ti.f32, shape=MAX_STREAKS)
        self.streak_tail = ti.Vector.field(3, ti.f32, shape=MAX_STREAKS)
        self.streak_gain = ti.field(ti.f32, shape=MAX_STREAKS)
        self.count = ti.field(ti.i32, shape=())
        self.accum = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.weight = ti.field(ti.f32, shape=(width, height))
        self.count[None] = 0
        # Packed upload staging: one host-to-device copy per frame, [0] is
        # the alive count followed by count records of head.xyz, tail.xyz,
        # gain unpacked on the device.
        self.streak_stage = ti.field(ti.f32, shape=MAX_STREAKS * 7 + 1)

    def spawn(self, rain_rate, wind, dt):
        """Poisson births into free slots; one drop is a bundle of
        STREAKS_PER_DROP streaks born at nearly the same point."""
        drops = int(self.rng.poisson(rain_rate * dt))
        need = drops * STREAKS_PER_DROP
        if need == 0:
            return
        free = np.nonzero(~self.alive)[0]
        for i in free[: min(len(free), need)]:
            self._spawn(i, wind)

    def _spawn(self, i, wind):
        self.pos[i] = (self.rng.uniform(SPAWN_MIN_X, SPAWN_MAX_X),
                       self.rng.uniform(SPAWN_MIN_Y, SPAWN_MAX_Y),
                       self.rng.uniform(SPAWN_MIN_Z, SPAWN_MAX_Z))
        self.prev[i] = self.pos[i]
        self.vel[i] = (wind * WIND_DRIFT * self.rng.uniform(0.6, 1.3),
                       -self.rng.uniform(*FALL_SPEED),
                       wind * WIND_DRIFT * self.rng.uniform(-0.4, 0.4))
        self.gain[i] = self.rng.uniform(0.5, 1.0)
        self.alive[i] = True
        self.spawned += 1

    def advance(self, dt, wind=0.0):
        """Advect alive streaks; returns this frame's landing points.

        The update is a plain Euler step, mirrored by the tests. Surface
        heights are gathered in one vectorized terrain call, since a
        scalar call per streak costs milliseconds per frame. A landing
        streak simply dies; spawned == landed + alive holds at every
        frame because spawn() is accounted per streak actually born.
        """
        indices = np.nonzero(self.alive)[0]
        landings = []
        if len(indices):
            self.prev[indices] = self.pos[indices]
            self.pos[indices] = self.pos[indices] + self.vel[indices] * dt
            xs, zs = self.pos[indices, 0], self.pos[indices, 2]
            ground = np.maximum(terrain_height_np(xs.astype(np.float32),
                                                  zs.astype(np.float32)), 0.0)
            landed = indices[self.pos[indices, 1] <= ground]
            for i in landed:
                landings.append((float(self.pos[i, 0]), float(self.pos[i, 2])))
            self.landed += len(landed)
            self.alive[landed] = False
        return landings

    # Rendered tails trail the head by this many advection steps times the
    # exposure control, so a streak reads as a line instead of a two-pixel
    # blip at 60 fps.
    TAIL_STRETCH = 3.0

    def upload(self, exposure):
        indices = np.nonzero(self.alive)[0]
        n = len(indices)
        packed = np.zeros(MAX_STREAKS * 7 + 1, dtype=np.float32)
        packed[0] = n
        if n:
            head = self.pos[indices]
            tail = head + (head - self.prev[indices]) \
                * (self.TAIL_STRETCH * exposure)
            records = np.concatenate(
                [head, tail, self.gain[indices][:, None]], axis=1)
            packed[1: 1 + n * 7] = records.reshape(-1)
        self.streak_stage.from_numpy(packed)
        self._unpack_streaks()

    @ti.kernel
    def _unpack_streaks(self):
        count = ti.cast(self.streak_stage[0], ti.i32)
        self.count[None] = count
        for k in range(count):
            base = 1 + k * 7
            self.streak_head[k] = ti.Vector(
                [self.streak_stage[base], self.streak_stage[base + 1],
                 self.streak_stage[base + 2]])
            self.streak_tail[k] = ti.Vector(
                [self.streak_stage[base + 3], self.streak_stage[base + 4],
                 self.streak_stage[base + 5]])
            self.streak_gain[k] = self.streak_stage[base + 6]

    @ti.kernel
    def splat(self, eye: ti.types.vector(3, ti.f32),
              forward: ti.types.vector(3, ti.f32),
              right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
              scale: ti.f32, strength: ti.f32, fog_distance: ti.f32,
              hit_depth: ti.template()):
        tint = ti.Vector(RAIN_TINT)
        for i in range(self.count[None]):
            head = self.streak_head[i]
            tail = self.streak_tail[i]
            dh = head - eye
            th = tail - eye
            ch, ct = dh.dot(forward), th.dot(forward)
            if ch > 0.4 and ct > 0.4:
                hx = 0.5 * self.width + dh.dot(right) / ch * 0.5 * self.height / scale
                hy = 0.5 * self.height + dh.dot(up) / ch * 0.5 * self.height / scale
                tx = 0.5 * self.width + th.dot(right) / ct * 0.5 * self.height / scale
                ty = 0.5 * self.height + th.dot(up) / ct * 0.5 * self.height / scale
                span = ti.max(ti.abs(hx - tx), ti.abs(hy - ty))
                steps = ti.cast(ti.min(MAX_SPLAT_STEPS, ti.ceil(span)), ti.i32)
                if steps > 0:
                    distance = dh.norm()
                    # Near streaks read larger and brighter; distant ones
                    # fade into the fog instead of popping off.
                    near = ti.min(1.6, ti.max(0.45, 6.0 / distance))
                    fog = ti.exp(-2.5 * (distance / fog_distance) ** 2)
                    gain = self.streak_gain[i] * strength * near * fog
                    # Energy per pixel stays bounded when the projected
                    # segment exceeds the step cap, so very near streaks
                    # neither blow out nor get skipped for length.
                    energy = ti.min(1.0, MAX_SPLAT_STEPS / span)
                    for k in range(steps + 1):
                        fraction = k / steps
                        x = ti.cast(hx + (tx - hx) * fraction, ti.i32)
                        y = ti.cast(hy + (ty - hy) * fraction, ti.i32)
                        # Only splat where the streak is in front of the
                        # scene surface along this pixel's ray.
                        if 0 <= x < self.width and 0 <= y < self.height:
                            if distance < hit_depth[x, y] + 0.5:
                                # Brightness decreases toward the older
                                # tail.
                                amount = (gain * energy
                                          * (1.0 - 0.55 * fraction) * 0.5)
                                ti.atomic_add(self.accum[x, y], tint * amount)
                                ti.atomic_add(self.weight[x, y], amount)

    @ti.kernel
    def composite(self, hdr: ti.template(), strength: ti.f32):
        for x, y in self.accum:
            weight = self.weight[x, y]
            if weight > 1e-8:
                alpha = 1.0 - ti.exp(-weight * strength)
                color = self.accum[x, y] / weight
                base = hdr[x, y]
                hdr[x, y] = base * (1.0 - alpha) + color * alpha
            self.accum[x, y] = ti.Vector([0.0, 0.0, 0.0])
            self.weight[x, y] = 0.0


@ti.data_oriented
class SplashBursts:
    """Micro crown splashes near the camera; a small fixed droplet pool."""

    def __init__(self, seed=RAIN_SEED + 1):
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros((MAX_SPLASH_DROPS, 3), dtype=np.float64)
        self.vel = np.zeros((MAX_SPLASH_DROPS, 3), dtype=np.float64)
        self.age = np.zeros(MAX_SPLASH_DROPS, dtype=np.float64)
        self.life = np.zeros(MAX_SPLASH_DROPS, dtype=np.float64)
        self.alive = np.zeros(MAX_SPLASH_DROPS, dtype=bool)
        self.cursor = 0
        self.drop_pos = ti.Vector.field(3, ti.f32, shape=MAX_SPLASH_DROPS)
        self.drop_gain = ti.field(ti.f32, shape=MAX_SPLASH_DROPS)
        self.count = ti.field(ti.i32, shape=())
        self.count[None] = 0
        # Packed upload staging, same layout idea as RainStreaks: [0] is
        # the alive count followed by count records of x, y, z, gain.
        self.drop_stage = ti.field(ti.f32, shape=MAX_SPLASH_DROPS * 4 + 1)

    def spawn_burst(self, x, y, z):
        """Ring of upward droplets; callers gate by distance and rain."""
        base_angle = self.rng.uniform(0.0, 2.0 * np.pi)
        for k in range(6):
            i = self.cursor
            self.cursor = (self.cursor + 1) % MAX_SPLASH_DROPS
            angle = base_angle + k * (2.0 * np.pi) / 6.0
            radial = self.rng.uniform(0.4, 0.9)
            self.pos[i] = (x + 0.03 * np.cos(angle), y + 0.02,
                           z + 0.03 * np.sin(angle))
            self.vel[i] = (np.cos(angle) * radial,
                           self.rng.uniform(1.1, 1.9),
                           np.sin(angle) * radial)
            self.age[i] = 0.0
            self.life[i] = self.rng.uniform(0.22, 0.34)
            self.alive[i] = True

    def maybe_spawn(self, x, z, intensity, eye):
        """Probability gate: near landings only, scaled by the splash knob."""
        distance = float(np.hypot(x - eye[0], z - eye[2]))
        if distance > SPLASH_RANGE or self.rng.random() > 0.5 * intensity:
            return
        self.spawn_burst(x, surface_height_np(x, z), z)

    def advance(self, dt):
        for i in range(MAX_SPLASH_DROPS):
            if not self.alive[i]:
                continue
            self.age[i] += dt
            self.vel[i, 1] -= 9.8 * dt
            self.pos[i] = self.pos[i] + self.vel[i] * dt
            if (self.age[i] >= self.life[i]
                    or self.pos[i, 1] <= surface_height_np(self.pos[i, 0],
                                                           self.pos[i, 2])):
                self.alive[i] = False

    def upload(self):
        indices = np.nonzero(self.alive)[0]
        n = len(indices)
        packed = np.zeros(MAX_SPLASH_DROPS * 4 + 1, dtype=np.float32)
        packed[0] = n
        if n:
            gain = 1.0 - self.age[indices] / self.life[indices]
            records = np.concatenate(
                [self.pos[indices], gain[:, None]], axis=1)
            packed[1: 1 + n * 4] = records.reshape(-1)
        self.drop_stage.from_numpy(packed)
        self._unpack_drops()

    @ti.kernel
    def _unpack_drops(self):
        count = ti.cast(self.drop_stage[0], ti.i32)
        self.count[None] = count
        for k in range(count):
            base = 1 + k * 4
            self.drop_pos[k] = ti.Vector(
                [self.drop_stage[base], self.drop_stage[base + 1],
                 self.drop_stage[base + 2]])
            self.drop_gain[k] = self.drop_stage[base + 3]

    @ti.kernel
    def splat(self, accum: ti.template(), weight: ti.template(),
              width: ti.i32, height: ti.i32, eye: ti.types.vector(3, ti.f32),
              forward: ti.types.vector(3, ti.f32),
              right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
              scale: ti.f32, strength: ti.f32):
        tint = ti.Vector(RAIN_TINT)
        for i in range(self.count[None]):
            delta = self.drop_pos[i] - eye
            depth = delta.dot(forward)
            if depth > 0.4:
                px = 0.5 * width + delta.dot(right) / depth \
                    * 0.5 * height / scale
                py = 0.5 * height + delta.dot(up) / depth \
                    * 0.5 * height / scale
                if 0 <= px < width and 0 <= py < height:
                    distance = delta.norm()
                    near = ti.min(1.2, ti.max(0.2, 3.0 / distance))
                    gain = self.drop_gain[i] * strength * near
                    cx, cy = ti.cast(px, ti.i32), ti.cast(py, ti.i32)
                    for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        x, y = cx + ox, cy + oy
                        if 0 <= x < width and 0 <= y < height:
                            falloff = 1.0 if ox == 0 and oy == 0 else 0.35
                            amount = gain * falloff * 0.5
                            ti.atomic_add(accum[x, y], tint * amount)
                            ti.atomic_add(weight[x, y], amount)
