"""Rain pond: interactive ripples from a simulated 2D wave-equation height field.

Raindrops inject ripples into a finite-difference wave simulation; clicking
the water surface adds a ripple at the picked point. The water material
(Fresnel reflection/refraction, absorption, glints, caustics) is shared with
the courtyard demo through CourtyardScene.

Run: uv run python water/demo_02.py
Export: uv run python water/demo_02.py --headless --output output/rain.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .pbr import fresnel_schlick, ggx_brdf
    from .finish import water_reflection
    from .scene import CourtyardScene, OrbitCamera, POOL_HALF_X, POOL_HALF_Z, BED_HEIGHT
else:
    from pbr import fresnel_schlick, ggx_brdf
    from finish import water_reflection
    from scene import CourtyardScene, OrbitCamera, POOL_HALF_X, POOL_HALF_Z, BED_HEIGHT

__all__ = ["RainPondRenderer", "RainSystem", "WaveSimulation", "main", "parse_args",
           "SIM_NX", "SIM_NZ", "CELL_X", "SIM_DT", "WAVE_SPEED_MAX"]

# Simulation grid over the pool interior; cells are square.
SIM_NX, SIM_NZ = 256, 172
CELL_X = 2.0 * POOL_HALF_X / SIM_NX
CELL_Z = 2.0 * POOL_HALF_Z / SIM_NZ
SIM_DT = 1.0 / 120.0
# Explicit scheme stays stable while c*dt/dx stays below 1/sqrt(2); the wave
# speed panel range keeps a safety margin.
WAVE_SPEED_MIN, WAVE_SPEED_MAX = 0.3, 1.8
DAMPING_MIN, DAMPING_MAX = 0.990, 0.9999
MAX_DROPS = 400
RAIN_TABLE = 1024
SPLASH_RADIUS = 0.06


@ti.data_oriented
class WaveSimulation:
    """2D wave equation on a regular grid with reflective pool walls.

    Leapfrog finite-difference update with damping at a fixed time step.
    Three buffers rotate through current / previous / next without rebinding
    any field reference, so kernels compiled against self.h stay valid.
    """

    def __init__(self):
        if abs(CELL_X - CELL_Z) > 1e-9:
            raise ValueError("Simulation cells must be square for an isotropic wave speed")
        self.h = ti.field(ti.f32, shape=(SIM_NX, SIM_NZ))
        self.h_prev = ti.field(ti.f32, shape=(SIM_NX, SIM_NZ))
        self.h_next = ti.field(ti.f32, shape=(SIM_NX, SIM_NZ))
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
            # Clamped neighbor reads give a zero-gradient (reflective) wall.
            left = self.h[ti.max(0, i - 1), j]
            right = self.h[ti.min(SIM_NX - 1, i + 1), j]
            down = self.h[i, ti.max(0, j - 1)]
            up = self.h[i, ti.min(SIM_NZ - 1, j + 1)]
            laplacian = left + right + down + up - 4.0 * center
            self.h_next[i, j] = (2.0 * center - self.h_prev[i, j] + courant2 * laplacian) * damping

    @ti.kernel
    def inject(self, cx: ti.f32, cz: ti.f32, radius: ti.f32, strength: ti.f32):
        gx = (cx + POOL_HALF_X) / CELL_X - 0.5
        gz = (cz + POOL_HALF_Z) / CELL_Z - 0.5
        reach = ti.cast(radius / CELL_X + 1.5, ti.i32)
        ci, cj = ti.cast(gx, ti.i32), ti.cast(gz, ti.i32)
        for a, b in ti.ndrange((-reach, reach + 1), (-reach, reach + 1)):
            i, j = ci + a, cj + b
            if 0 <= i < SIM_NX and 0 <= j < SIM_NZ:
                dx = (ti.cast(i, ti.f32) - gx) * CELL_X
                dz = (ti.cast(j, ti.f32) - gz) * CELL_Z
                falloff = ti.exp(-(dx * dx + dz * dz) / (radius * radius))
                self.h[i, j] += strength * falloff

    @ti.func
    def sample(self, x, z):
        """Bilinear height with the cell-exact bilinear gradient."""
        u = ti.max(0.0, ti.min(SIM_NX - 1.001, (x + POOL_HALF_X) / CELL_X - 0.5))
        v = ti.max(0.0, ti.min(SIM_NZ - 1.001, (z + POOL_HALF_Z) / CELL_Z - 0.5))
        i0, j0 = ti.cast(u, ti.i32), ti.cast(v, ti.i32)
        fu, fv = u - i0, v - j0
        h00 = self.h[i0, j0]
        h10 = self.h[i0 + 1, j0]
        h01 = self.h[i0, j0 + 1]
        h11 = self.h[i0 + 1, j0 + 1]
        height = (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv
        dhdx = ((h10 - h00) * (1.0 - fv) + (h11 - h01) * fv) / CELL_X
        dhdz = ((h01 - h00) * (1.0 - fu) + (h11 - h10) * fu) / CELL_Z
        return ti.Vector([height, dhdx, dhdz])

    def advance(self, wave_speed, damping, substeps):
        courant2 = (wave_speed * SIM_DT / CELL_X) ** 2
        for _ in range(substeps):
            self.step(courant2, damping)
            self.h_prev.copy_from(self.h)
            self.h.copy_from(self.h_next)
        self.steps += substeps


@ti.data_oriented
class RainSystem:
    """Falling drops with fixed-seed spawn tables; impacts feed the wave field.

    The spawn region sits above the pool interior, so streaks never overlap
    scene occluders and splashes always land inside the water.
    """

    def __init__(self, sim, seed=17):
        self.sim = sim
        rng = np.random.default_rng(seed)
        self.spawn_x = ti.field(ti.f32, shape=RAIN_TABLE)
        self.spawn_y = ti.field(ti.f32, shape=RAIN_TABLE)
        self.spawn_z = ti.field(ti.f32, shape=RAIN_TABLE)
        self.spawn_speed = ti.field(ti.f32, shape=RAIN_TABLE)
        self.drift_x = ti.field(ti.f32, shape=RAIN_TABLE)
        self.drift_z = ti.field(ti.f32, shape=RAIN_TABLE)
        self.spawn_x.from_numpy(rng.uniform(-POOL_HALF_X + 0.15, POOL_HALF_X - 0.15, RAIN_TABLE).astype(np.float32))
        self.spawn_y.from_numpy(rng.uniform(0.3, 3.2, RAIN_TABLE).astype(np.float32))
        self.spawn_z.from_numpy(rng.uniform(-POOL_HALF_Z + 0.15, POOL_HALF_Z - 0.15, RAIN_TABLE).astype(np.float32))
        self.spawn_speed.from_numpy(rng.uniform(3.0, 4.2, RAIN_TABLE).astype(np.float32))
        self.drift_x.from_numpy(rng.uniform(-0.25, 0.25, RAIN_TABLE).astype(np.float32))
        self.drift_z.from_numpy(rng.uniform(-0.25, 0.25, RAIN_TABLE).astype(np.float32))
        self.pos = ti.Vector.field(3, ti.f32, shape=MAX_DROPS)
        self.vel = ti.Vector.field(3, ti.f32, shape=MAX_DROPS)
        self.slot = ti.field(ti.i32, shape=MAX_DROPS)
        self.count = ti.field(ti.i32, shape=())

    @ti.kernel
    def reset(self):
        for i in range(MAX_DROPS):
            k = i % RAIN_TABLE
            self.slot[i] = k
            self.pos[i] = ti.Vector([self.spawn_x[k], self.spawn_y[k], self.spawn_z[k]])
            self.vel[i] = ti.Vector([self.drift_x[k], -self.spawn_speed[k], self.drift_z[k]])
        self.count[None] = MAX_DROPS

    @ti.kernel
    def update(self, dt: ti.f32, impact: ti.f32, active: ti.i32, gravity: ti.f32):
        for i in range(MAX_DROPS):
            if i < active:
                p = self.pos[i]
                v = self.vel[i]
                v.y -= gravity * dt
                p += v * dt
                if p.y <= 0.0:
                    speed = v.norm()
                    strength = impact * (0.35 + 0.18 * speed) * 0.014
                    gx = (p.x + POOL_HALF_X) / CELL_X - 0.5
                    gz = (p.z + POOL_HALF_Z) / CELL_Z - 0.5
                    ci, cj = ti.cast(gx, ti.i32), ti.cast(gz, ti.i32)
                    for a, b in ti.ndrange((-2, 3), (-2, 3)):
                        ii, jj = ci + a, cj + b
                        if 0 <= ii < SIM_NX and 0 <= jj < SIM_NZ:
                            dx = (ti.cast(ii, ti.f32) - gx) * CELL_X
                            dz = (ti.cast(jj, ti.f32) - gz) * CELL_Z
                            splash = strength * ti.exp(-(dx * dx + dz * dz) / (2.0 * SPLASH_RADIUS * SPLASH_RADIUS))
                            ti.atomic_add(self.sim.h[ii, jj], splash)
                    k = (self.slot[i] + MAX_DROPS) % RAIN_TABLE
                    self.slot[i] = k
                    p = ti.Vector([self.spawn_x[k], self.spawn_y[k], self.spawn_z[k]])
                    v = ti.Vector([self.drift_x[k], -self.spawn_speed[k], self.drift_z[k]])
                self.pos[i] = p
                self.vel[i] = v


@ti.data_oriented
class RainPondRenderer(CourtyardScene):
    """Ray-traced rain pond over the simulated height field.

    Reuses the courtyard scene, PBR shading, and HDR finishing; the water
    surface, rain streaks, and caustics come from the wave simulation.
    """

    def __init__(self, width, height, samples=4):
        if samples not in (1, 4):
            raise ValueError("samples must be 1 or 4")
        super().__init__(width, height)
        self.samples = samples
        self.sim = WaveSimulation()
        self.rain = RainSystem(self.sim)
        self.rain_controls = ti.Vector.field(2, ti.f32, shape=())
        self.rain_controls[None] = [0.6, 0.55]
        self._caustic_key = None

    @ti.func
    def wave(self, x, z):
        return self.sim.sample(x, z)

    @ti.func
    def detail_gradient(self, x, z, clock, footprint):
        gradient = ti.Vector([0.0, 0.0])
        edge = self.smooth(0.0, 0.45, POOL_HALF_X - ti.abs(x)) * self.smooth(0.0, 0.45, POOL_HALF_Z - ti.abs(z))
        for band in ti.static(range(2)):
            kx, kz, strength, speed = 32.0, -24.0, 0.00063, 2.2
            if ti.static(band == 1):
                kx, kz, strength, speed = 59.0, 43.0, 0.00032, 3.1
            wavelength = 2.0 * math.pi / ti.sqrt(kx * kx + kz * kz)
            fade = 1.0 - self.smooth(wavelength * 0.12, wavelength * 0.5, footprint)
            phase = kx * x + kz * z - speed * clock
            gradient += strength * ti.cos(phase) * ti.Vector([kx, kz]) * fade
        return gradient * edge * self.water_controls[None].y

    @ti.kernel
    def trace_caustics(self, clock: ti.f32, lighting: ti.f32, shadows: ti.i32):
        for x, y in self.caustic_raw:
            self.caustic_raw[x, y] = 0.0
        for x, y in ti.ndrange(256, 192):
            px, pz = (x + 0.5) / 256 * 2 * POOL_HALF_X - POOL_HALF_X, (y + 0.5) / 192 * 2 * POOL_HALF_Z - POOL_HALF_Z
            wave = self.wave(px, pz)
            fine = self.detail_gradient(px, pz, clock, 2 * POOL_HALF_X / 128)
            normal = ti.Vector([-wave.y - fine.x, 1.0, -wave.z - fine.y]).normalized()
            p = ti.Vector([px, wave.x, pz])
            sun = self.sun_direction(lighting)
            cosine = ti.max(0.0, sun.dot(normal))
            eta = 1.0 / 1.333
            ray = (-eta * sun + (eta * cosine - ti.sqrt(ti.max(0.0, 1.0 - eta * eta * (1.0 - cosine * cosine)))) * normal).normalized()
            distance = (BED_HEIGHT - p.y) / ray.y
            target = p + distance * ray
            visible = 1
            if shadows == 1:
                visible = ti.cast(self.occlusion_distance(p + normal * 0.006, sun, 18.0) >= 18.0, ti.i32)
            if visible and cosine > 0.0 and distance > 0.0 and ti.abs(target.x) < POOL_HALF_X and ti.abs(target.z) < POOL_HALF_Z:
                f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
                transmitted = 1.0 - fresnel_schlick(cosine, f0)
                # Source area / receiver cell area = 1/4. Correct tilted flux.
                energy = transmitted * cosine / ti.max(0.001, normal.y * sun.y) * 0.25
                u, v = (target.x + POOL_HALF_X) / (2 * POOL_HALF_X) * 128 - 0.5, (target.z + POOL_HALF_Z) / (2 * POOL_HALF_Z) * 96 - 0.5
                ix, iy = ti.cast(ti.floor(u), ti.i32), ti.cast(ti.floor(v), ti.i32)
                fx, fy = u - ti.floor(u), v - ti.floor(v)
                for a, b in ti.static(ti.ndrange(2, 2)):
                    if 0 <= ix + a < 128 and 0 <= iy + b < 96:
                        weight = (fx if a else 1.0 - fx) * (fy if b else 1.0 - fy)
                        ti.atomic_add(self.caustic_raw[ix + a, iy + b], energy * weight)

    @ti.func
    def water_hit(self, origin, direction):
        t = 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        if direction.y < -0.0001:
            candidate = -origin.y / direction.y
            for _ in ti.static(range(8)):
                p = origin + direction * candidate
                wave = self.wave(p.x, p.z)
                derivative = direction.y - wave.y * direction.x - wave.z * direction.z
                if ti.abs(derivative) > 0.005:
                    step = (p.y - wave.x) / derivative
                    candidate -= ti.max(-0.25, ti.min(0.25, step))
            p = origin + direction * candidate
            wave = self.wave(p.x, p.z)
            if candidate > 0.001 and ti.abs(p.x) < POOL_HALF_X and ti.abs(p.z) < POOL_HALF_Z and ti.abs(p.y - wave.x) < 0.005:
                t = candidate
                normal = ti.Vector([-wave.y, 1.0, -wave.z]).normalized()
        return t, normal

    @ti.func
    def rain_radiance(self, origin, direction):
        # Analytic ray-streak proximity: each drop draws a short faint streak.
        color = ti.Vector([0.0, 0.0, 0.0])
        count = self.rain.count[None]
        for i in range(count):
            p = self.rain.pos[i]
            v = self.rain.vel[i]
            speed = v.norm()
            if speed > 0.5 and p.y > 0.03:
                s = v / speed
                w0 = origin - (p + s * 0.11)
                b = direction.dot(s)
                d = direction.dot(w0)
                e = s.dot(w0)
                denom = 1.0 - b * b
                if ti.abs(denom) > 1e-3:
                    seg_v = e + b * (b * e - d) / denom
                    seg_v = ti.max(0.0, ti.min(0.13, seg_v))
                    ray_t = seg_v * b - d
                    if ray_t > 0.05:
                        dist = (w0 + direction * ray_t - s * seg_v).norm()
                        glow = ti.exp(-(dist / 0.0085) ** 2)
                        if glow > 0.003:
                            near_surface = ti.min(1.0, p.y * 5.0)
                            attenuation = 1.0 / (1.0 + 0.008 * ray_t * ray_t)
                            color += ti.Vector([0.62, 0.68, 0.76]) * (glow * 0.16 * near_surface
                                                                    * attenuation * (0.7 + 0.3 * speed / 4.0))
        return color

    @ti.func
    def radiance(self, origin, direction, clock, clarity, lighting, pixel_angle):
        opaque_t, opaque_normal, opaque_base, opaque_material = self.geometry(origin, direction)
        water_t, normal = self.water_hit(origin, direction)
        has_water = water_t < opaque_t
        p = origin + direction * water_t
        if has_water:
            footprint = water_t * pixel_angle / ti.max(0.15, ti.abs(direction.y))
            fine = self.detail_gradient(p.x, p.z, clock, footprint)
            normal = (normal + ti.Vector([-fine.x, 0.0, -fine.y])).normalized()
        cosine = ti.max(0.0, ti.min(1.0, -direction.dot(normal)))
        eta = 1.0 / 1.333
        root = ti.sqrt(ti.max(0.0, 1.0 - eta * eta * (1.0 - cosine * cosine)))
        refracted = (eta * direction + (eta * cosine - root) * normal).normalized()
        color, reflection = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        thickness = 0.0
        reflection_count = 0
        if has_water:
            reflection_count = ti.cast(self.water_controls[None].w, ti.i32)
            if self.water_controls[None].x <= 0.001:
                reflection_count = 1
        # Share one shading body across primary and every water secondary ray.
        for bounce in range(1 + reflection_count):
            ray_origin, ray_direction = origin, direction
            t, n, base, material = opaque_t, opaque_normal, opaque_base, opaque_material
            weight = 1.0
            if has_water:
                if bounce < reflection_count:
                    ray_direction, weight = water_reflection(-direction, normal, self.water_controls[None].x, bounce, reflection_count)
                    ray_origin = p + normal * 0.004
                else:
                    ray_origin, ray_direction = p - normal * 0.004, refracted
                t, n, base, material = self.geometry(ray_origin, ray_direction)
            shaded = ti.Vector([0.0, 0.0, 0.0])
            if weight > 0.0 or bounce == reflection_count:
                shaded, _ = self.shade_surface(ray_origin, ray_direction, t, n, base, material, clock, lighting, 0.0, ti.cast(not has_water, ti.i32))
            color = shaded
            if has_water and bounce < reflection_count:
                reflection += shaded * weight / reflection_count
            elif has_water:
                thickness = ti.min(t, 8.0)
        if has_water:
            transmission = ti.exp(-ti.Vector([0.48, 0.18, 0.095]) * thickness / clarity)
            through_water = color * transmission + ti.Vector([0.035, 0.22, 0.20]) * (1.0 - transmission)
            f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
            fresnel = f0 + (1.0 - f0) * (1.0 - cosine) ** 5
            color = through_water * (1.0 - fresnel) + reflection
            glint = ggx_brdf(ti.Vector([0.0, 0.0, 0.0]), ti.max(0.045, self.water_controls[None].x), 0.0,
                             normal, -direction, self.sun_direction(lighting), f0)
            if glint.max() > 0.005:
                shade = self.visibility(p, normal, self.sun_direction(lighting), 0)
                color += glint * ti.Vector([1.0, 0.80, 0.52]) * 3.2 * shade
        if self.rain_controls[None].x > 0.0:
            color += self.rain_radiance(origin, direction) * self.rain_controls[None].x
        return color

    @ti.kernel
    def render(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
               right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
               clock: ti.f32, clarity: ti.f32, lighting: ti.f32,
               exposure: ti.f32, scale: ti.f32, samples: ti.i32):
        for x, y in self.image:
            color = ti.Vector([0.0, 0.0, 0.0])
            for sample in range(samples):
                offset = ti.Vector([0.5, 0.5])
                if samples == 4:
                    offset = ti.Vector([0.25 + 0.5 * (sample % 2), 0.25 + 0.5 * (sample // 2)])
                sx = (2.0 * (x + offset.x) / self.width - 1.0) * self.width / self.height * scale
                sy = (2.0 * (y + offset.y) / self.height - 1.0) * scale
                direction = (forward + right * sx + up * sy).normalized()
                color += self.radiance(eye, direction, clock, clarity, lighting, 2.0 * scale / self.height)
            color /= samples
            self.post.hdr[x, y] = color

    def draw(self, camera, clock, clarity=1.4, lighting=1.0, exposure=1.05):
        key = (self.sim.steps, lighting)
        # Retrace only when the simulation or the sun has changed.
        if self._caustic_key is None or (self.water_controls[None].z > 0.0 and key != self._caustic_key):
            self.trace_caustics(clock, lighting, 1)
            self.filter_caustics()
            self._caustic_key = key
        self.render(*camera.basis(), clock, clarity, lighting,
                    exposure, math.tan(math.radians(camera.fov / 2)), self.samples)
        bloom, threshold, vignette = self.post_controls[None]
        self.post.apply(exposure, bloom, threshold, vignette)


def water_pick(camera, u, v, width, height):
    """Intersect the cursor ray (normalized, v from the top) with y = 0."""
    eye, forward, right, up = camera.basis()
    scale = math.tan(math.radians(camera.fov / 2.0))
    # u, v are continuous cursor positions: no pixel-center offset here.
    px = (2.0 * u - 1.0) * width / height * scale
    py = (2.0 * (1.0 - v) - 1.0) * scale
    direction = forward + right * px + up * py
    direction /= np.linalg.norm(direction)
    point = None
    if direction[1] < -1e-5:
        t = -eye[1] / direction[1]
        hit = eye + direction * t
        if abs(hit[0]) < POOL_HALF_X and abs(hit[2]) < POOL_HALF_Z:
            point = (float(hit[0]), float(hit[2]))
    return point


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 02: rain pond with interactive simulated ripples")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=4, help="Spatial samples per pixel (default: 4)")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="day")
    parser.add_argument("--shadow-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--water-roughness", type=float, default=0.16)
    parser.add_argument("--reflection-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-water-detail", action="store_true")
    parser.add_argument("--no-caustics", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--rain", type=float, default=0.55, help="Rain amount from 0 to 1")
    parser.add_argument("--impact", type=float, default=0.45, help="Drop impact strength")
    parser.add_argument("--wave-speed", type=float, default=1.3, help="Wave speed in m/s")
    parser.add_argument("--damping", type=float, default=0.9975, help="Per-step amplitude damping")
    parser.add_argument("--preset", choices=("overview", "waterline", "top"), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_02.png"))
    parser.add_argument("--time", type=float, default=2.0, help="Rain warmup in seconds before the first frame")
    parser.add_argument("--frames", type=int, default=1, help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0, help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache for troubleshooting")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0 or not math.isfinite(args.time):
        parser.error("width/height must be >= 32, frames >= 1, and time finite")
    if not math.isfinite(args.water_roughness) or not 0.0 <= args.water_roughness <= 0.45:
        parser.error("water-roughness must be finite and between 0 and 0.45")
    if not math.isfinite(args.wave_speed) or not WAVE_SPEED_MIN <= args.wave_speed <= WAVE_SPEED_MAX:
        parser.error(f"wave-speed must be finite and between {WAVE_SPEED_MIN} and {WAVE_SPEED_MAX}")
    if not math.isfinite(args.damping) or not DAMPING_MIN <= args.damping <= DAMPING_MAX:
        parser.error(f"damping must be finite and between {DAMPING_MIN} and {DAMPING_MAX}")
    if not math.isfinite(args.rain) or not 0.0 <= args.rain <= 1.0:
        parser.error("rain must be finite and between 0 and 1")
    if not math.isfinite(args.impact) or not 0.0 <= args.impact <= 1.5:
        parser.error("impact must be finite and between 0 and 1.5")
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = RainPondRenderer(args.width, args.height, args.samples)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 1.0, 0.0 if args.no_caustics else 1.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    renderer.rain_controls[None] = [0.6, args.impact]
    camera = OrbitCamera()
    camera.preset(args.preset)
    rain_amount, wave_speed, damping, speed = args.rain, args.wave_speed, args.damping, 1.0
    active_drops = max(0, int(round(rain_amount * MAX_DROPS)))
    lighting = 1.0 if args.lighting == "day" else 0.0
    clarity, exposure = 1.4, 1.05
    clock = 0.0

    def advance(dt, substeps):
        nonlocal clock
        renderer.rain.count[None] = max(0, min(MAX_DROPS, int(round(rain_amount * MAX_DROPS))))
        if renderer.rain.count[None] > 0:
            renderer.rain.update(dt, renderer.rain_controls[None].y, renderer.rain.count[None], 1.0)
        renderer.sim.advance(wave_speed, damping, substeps)
        clock += dt * speed

    # Warmup reaches steady rain before the first visible frame.
    warmup_steps = int(max(0.0, args.time) * 120.0)
    for _ in range(warmup_steps):
        advance(SIM_DT, 1)
    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            advance(1.0 / 60.0, 2)
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
    window = ti.ui.Window("02 / Rain Pond", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    sim_accumulator, frame_ms, window_frames = 0.0, 0.0, 1
    drag_in_scene, click_start, click_camera_moved = False, None, False
    print("Click water: ripple | Drag LMB orbit | RMB pan | W/S zoom | 1/2/3 views | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
    while window.running and (not args.window_frames or window_frames < args.window_frames):
        now = time.perf_counter()
        dt = min(now - previous_time, 0.05)
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
                camera.preset({"1": "overview", "2": "waterline", "3": "top"}[key])
                auto_orbit = False
            elif key == "r":
                camera.preset("overview")
                renderer.samples = args.samples
                renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 1.0, 0.0 if args.no_caustics else 1.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                renderer.rain_controls[None] = [0.6, args.impact]
                renderer.sim.clear()
                renderer.rain.reset()
                rain_amount, wave_speed, damping, speed = args.rain, args.wave_speed, args.damping, 1.0
                lighting = 1.0 if args.lighting == "day" else 0.0
                clarity, exposure, clock, sim_accumulator = 1.4, 1.05, 0.0, 0.0
                paused, auto_orbit = False, False
            elif key == "p":
                save_frame = True
            elif key in (ti.ui.LMB, ti.ui.RMB):
                drag_in_scene = not in_panel
        if not window.running:
            break
        # A short click injects a ripple; a longer drag keeps orbiting.
        if drag_in_scene and window.is_pressed(ti.ui.LMB):
            if click_start is None:
                click_start, click_camera_moved = mouse.copy(), False
            elif not click_camera_moved and np.linalg.norm(mouse - click_start) > 0.012:
                click_camera_moved = True
        elif click_start is not None:
            if drag_in_scene and not click_camera_moved:
                point = water_pick(camera, click_start[0], click_start[1], args.width, args.height)
                if point is not None:
                    renderer.sim.inject(point[0], point[1], 0.06, 0.05)
            click_start = None
        if previous_mouse is not None and drag_in_scene and click_camera_moved:
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
            with gui.sub_window("WATER / FINISH", 0.72, 0.335, 0.26, 0.61):
                controls = renderer.water_controls[None]
                controls[0] = gui.slider_float("Water roughness", controls[0], 0.0, 0.45)
                controls[1] = gui.slider_float("Fine waves", controls[1], 0.0, 2.0)
                controls[3] = 4 if gui.checkbox("4x reflections", controls[3] == 4) else 1
                controls[2] = gui.slider_float("Caustics", controls[2], 0.0, 1.0)
                renderer.water_controls[None] = controls
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
                gui.text("No temporal accumulation")
            with gui.sub_window("RAIN POND / 02", 0.02, 0.025, 0.30, 0.82):
                gui.text("Wave equation / finite difference / rain")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                if gui.button("Single step [N]"):
                    step = True
                if gui.button("Calm water"):
                    renderer.sim.clear()
                rain_amount = gui.slider_float("Rain amount", rain_amount, 0.0, 1.0)
                controls = renderer.rain_controls[None]
                controls[1] = gui.slider_float("Drop impact", controls[1], 0.0, 1.5)
                controls[0] = gui.slider_float("Rain visibility", controls[0], 0.0, 1.5)
                renderer.rain_controls[None] = controls
                wave_speed = gui.slider_float("Wave speed", wave_speed, WAVE_SPEED_MIN, WAVE_SPEED_MAX)
                damping = gui.slider_float("Damping", damping, DAMPING_MIN, DAMPING_MAX)
                speed = gui.slider_float("Time speed", speed, 0.0, 2.0)
                clarity = gui.slider_float("Water clarity", clarity, 0.35, 3.0)
                lighting = gui.slider_float("Day / sunset", lighting, 0.0, 1.0)
                exposure = gui.slider_float("Exposure", exposure, 0.5, 1.8)
                renderer.samples = 4 if gui.checkbox("4x spatial AA", renderer.samples == 4) else 1
                if gui.button("Overview [1]"):
                    camera.preset("overview")
                    auto_orbit = False
                if gui.button("Waterline [2]"):
                    camera.preset("waterline")
                    auto_orbit = False
                if gui.button("Top view [3]"):
                    camera.preset("top")
                    auto_orbit = False
                gui.text("Click water to add a ripple")
                gui.text("Drag LMB orbit / RMB pan")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        if not paused:
            sim_accumulator = min(sim_accumulator + dt * speed, 8 * SIM_DT)
            substeps = int(sim_accumulator / SIM_DT)
            sim_accumulator -= substeps * SIM_DT
            advance(dt * speed, substeps)
        elif step:
            advance(1.0 / 60.0, 2)
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
