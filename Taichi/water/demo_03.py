"""Sunset sea: Gerstner swell with a low sun, distance fog, and a lighthouse skerry.

Directional Gerstner waves are summed analytically: deep-water dispersion
relates wavelength and frequency, per-wave steepness is bounded so crests
sharpen without looping, and pixel-footprint filtering fades fine detail at
distance. The ocean is ray-marched against the summed displacement with
error-controlled steps and bisection refinement; shading reuses the shared
water material (Fresnel reflection/refraction, moderate absorption, GGX
glints) and the courtyard PBR framework through the build_static hook.

Run: uv run python water/demo_03.py
Export: uv run python water/demo_03.py --headless --output output/sunset.png
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
    from .scene import CourtyardScene, OrbitCamera
else:
    from pbr import fresnel_schlick, ggx_brdf
    from finish import water_reflection
    from scene import CourtyardScene, OrbitCamera

__all__ = ["GerstnerSpectrum", "SunsetOceanRenderer", "main", "parse_args",
           "GRAVITY", "WAVE_COUNT", "SEABED", "FAR_PLANE", "FOG_DISTANCE",
           "apply_preset", "OCEAN_PRESETS"]

GRAVITY = 9.81
WAVE_COUNT = 12
# The last few components form a cross swell from a very different direction;
# their share of the total amplitude breaks the aligned-crest look.
CROSS_COUNT = 3
CROSS_SHARE = 0.2
ENVELOPE_COUNT = 2
SEABED = -7.0
FAR_PLANE = 1500.0
FOG_DISTANCE = 550.0
# Marching: the step grows with distance so far water costs logarithmic work;
# near-field accuracy comes from the surface-error term.
MARCH_MIN_STEP = 0.005
MARCH_GROWTH = 0.08
MARCH_MAX_STEPS = 160
BISECT_STEPS = 10

OCEAN_PRESETS = {
    "overview": (0.72, 0.16, 13.0),
    "waterline": (0.55, 0.13, 9.0),
    "top": (0.20, 1.42, 10.0),
}


def apply_preset(camera, name):
    camera.target = np.array([0.0, -0.08, 0.0])
    camera.yaw, camera.pitch, camera.distance = OCEAN_PRESETS[name]


@ti.data_oriented
class GerstnerSpectrum:
    """Fixed-count directional Gerstner set with deep-water dispersion.

    Wavelengths descend from the wind-driven primary swell with randomized
    ratios, so no pair of components shares a common beat period. The tail of
    the set is a cross swell travelling well away from the wind direction,
    and two very long envelope waves modulate the amplitude in space, which
    keeps the field from repeating at a glance. Per-wave steepness
    Q_i = steepness / (k_i A_i n) keeps the sum of Q k A at the slider value,
    which is the standard constraint against self-intersecting crests.
    """

    def __init__(self, seed=17):
        self.seed = seed
        self.dir = ti.Vector.field(2, ti.f32, shape=WAVE_COUNT)
        self.amp = ti.field(ti.f32, shape=WAVE_COUNT)
        self.k = ti.field(ti.f32, shape=WAVE_COUNT)
        self.omega = ti.field(ti.f32, shape=WAVE_COUNT)
        self.phase = ti.field(ti.f32, shape=WAVE_COUNT)
        self.steep = ti.field(ti.f32, shape=WAVE_COUNT)
        self.env_k = ti.Vector.field(2, ti.f32, shape=ENVELOPE_COUNT)
        self.env_amp = ti.field(ti.f32, shape=ENVELOPE_COUNT)
        self.env_phase = ti.field(ti.f32, shape=ENVELOPE_COUNT)
        self.amp_total = ti.field(ti.f32, shape=())
        self.table = None
        self.rebuild(wind_speed=8.0, wind_direction=35.0, wave_height=1.3,
                     steepness=0.6, scale=0.8)

    def rebuild(self, wind_speed, wind_direction, wave_height, steepness, scale=1.0):
        if not 0.0 <= wind_speed <= 24.0 or not 0.0 <= steepness <= 1.0:
            raise ValueError("wind speed and steepness out of range")
        rng = np.random.default_rng(self.seed)
        lam0 = scale * (4.0 + 3.2 * wind_speed ** 0.75)
        ladder = WAVE_COUNT - CROSS_COUNT
        # Randomized ratios avoid the visible beating of a regular ladder.
        ratios = rng.uniform(0.62, 0.82, ladder - 1)
        wavelengths = lam0 * np.concatenate([[1.0], np.cumprod(ratios)])
        wavelengths = np.concatenate([wavelengths, lam0 * rng.uniform(0.30, 0.85, CROSS_COUNT)])
        k = 2.0 * math.pi / wavelengths
        omega = np.sqrt(GRAVITY * k)
        base = math.radians(wind_direction)
        spread = 0.22 + 0.35 / (1.0 + wind_speed * 0.15)
        order = np.arange(ladder) / max(1, ladder - 1)
        angles = base + (rng.random(ladder) * 2.0 - 1.0) * spread * (0.3 + 0.7 * order)
        signs = rng.choice([-1.0, 1.0], CROSS_COUNT)
        angles = np.concatenate([angles,
                                 base + signs * rng.uniform(0.9, 2.3, CROSS_COUNT)])
        weights = k ** -1.25 * rng.uniform(0.75, 1.25, WAVE_COUNT)
        amp = np.empty(WAVE_COUNT)
        ladder_w = weights[:ladder] / weights[:ladder].sum()
        cross_w = weights[ladder:] / weights[ladder:].sum()
        amp[:ladder] = wave_height * (1.0 - CROSS_SHARE) * ladder_w
        amp[ladder:] = wave_height * CROSS_SHARE * cross_w
        steep = steepness / (k * amp * WAVE_COUNT)
        phase = rng.random(WAVE_COUNT) * 2.0 * math.pi
        direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        # Slow spatial amplitude modulation: two noncommensurate long waves.
        env_angles = rng.random(ENVELOPE_COUNT) * 2.0 * math.pi
        env_lam = lam0 * rng.uniform(5.0, 9.0, ENVELOPE_COUNT)
        env_k = (2.0 * math.pi / env_lam)[:, None] * np.stack([np.cos(env_angles), np.sin(env_angles)], axis=1)
        env_amp = np.array([0.30, 0.18])
        env_phase = rng.random(ENVELOPE_COUNT) * 2.0 * math.pi
        self.dir.from_numpy(direction.astype(np.float32))
        self.amp.from_numpy(amp.astype(np.float32))
        self.k.from_numpy(k.astype(np.float32))
        self.omega.from_numpy(omega.astype(np.float32))
        self.phase.from_numpy(phase.astype(np.float32))
        self.steep.from_numpy(steep.astype(np.float32))
        self.env_k.from_numpy(env_k.astype(np.float32))
        self.env_amp.from_numpy(env_amp.astype(np.float32))
        self.env_phase.from_numpy(env_phase.astype(np.float32))
        self.amp_total[None] = float(amp.sum())
        self.table = {"dir": direction, "amp": amp, "k": k, "omega": omega,
                      "phase": phase, "steep": steep,
                      "env_k": env_k, "env_amp": env_amp, "env_phase": env_phase}

    @ti.func
    def envelope(self, x, z):
        value = 1.0
        for e in ti.static(range(ENVELOPE_COUNT)):
            f = self.env_k[e].dot(ti.Vector([x, z])) + self.env_phase[e]
            value += self.env_amp[e] * ti.sin(f)
        return value

    @ti.func
    def height(self, x, z, t):
        # Intersection treats the sum as a height field; horizontal
        # displacement is accounted in the analytic normal only.
        env = self.envelope(x, z)
        y = 0.0
        for i in ti.static(range(WAVE_COUNT)):
            f = self.k[i] * (self.dir[i].x * x + self.dir[i].y * z) - self.omega[i] * t + self.phase[i]
            y += self.amp[i] * env * ti.sin(f)
        return y

    @ti.func
    def vertical_velocity(self, x, z, t):
        """Time derivative of the height field (envelope varies slowly)."""
        env = self.envelope(x, z)
        vy = 0.0
        for i in ti.static(range(WAVE_COUNT)):
            f = self.k[i] * (self.dir[i].x * x + self.dir[i].y * z) - self.omega[i] * t + self.phase[i]
            vy -= self.amp[i] * env * self.omega[i] * ti.cos(f)
        return vy

    @ti.func
    def surface(self, x, z, t):
        """Vertical displacement and the Gerstner normal with envelope."""
        env = self.envelope(x, z)
        env_dx, env_dz = 0.0, 0.0
        for e in ti.static(range(ENVELOPE_COUNT)):
            f = self.env_k[e].dot(ti.Vector([x, z])) + self.env_phase[e]
            c = ti.cos(f)
            env_dx += self.env_amp[e] * self.env_k[e].x * c
            env_dz += self.env_amp[e] * self.env_k[e].y * c
        y, y0 = 0.0, 0.0
        tx, ty, tz = 0.0, 0.0, 0.0
        bx, by, bz = 0.0, 0.0, 0.0
        for i in ti.static(range(WAVE_COUNT)):
            dx, dz = self.dir[i].x, self.dir[i].y
            f = self.k[i] * (dx * x + dz * z) - self.omega[i] * t + self.phase[i]
            s, c = ti.sin(f), ti.cos(f)
            a = self.amp[i] * env
            wa = self.k[i] * a
            q = self.steep[i]
            y += a * s
            y0 += self.amp[i] * s
            tx -= q * wa * dx * dx * s
            ty += wa * dx * c
            tz -= q * wa * dx * dz * s
            bx -= q * wa * dx * dz * s
            by += wa * dz * c
            bz -= q * wa * dz * dz * s
        tangent = ti.Vector([1.0 + tx, ty, tz])
        binormal = ti.Vector([bx, by, 1.0 + bz])
        normal = binormal.cross(tangent)
        # Envelope slope contribution beyond the per-wave amplitude scaling.
        normal.x -= y0 * env_dx
        normal.z -= y0 * env_dz
        return y, normal.normalized()


@ti.data_oriented
class SunsetOceanRenderer(CourtyardScene):
    """Ray-marched open ocean against the Gerstner sum, with horizon fog."""

    def __init__(self, width, height, samples=4):
        if samples not in (1, 4):
            raise ValueError("samples must be 1 or 4")
        super().__init__(width, height)
        self.samples = samples
        self.spectrum = GerstnerSpectrum(seed=17)
        self.water_controls[None] = [0.05, 0.5, 0.0, 4.0]

    def build_static(self):
        """A lighthouse skerry near the orbit center, two far sea stacks."""
        def box(center, extent, color, material=4, bevel=0.05):
            i = self.boxes
            self.box_center[i], self.box_extent[i] = center, extent
            self.box_color[i], self.box_material[i] = color, material
            self.box_bevel[i] = min(bevel, min(extent) * 0.8)
            self.boxes += 1

        basalt = (0.10, 0.095, 0.09)
        basalt2 = (0.125, 0.115, 0.105)
        plaster = (0.85, 0.82, 0.76)
        metal = (0.05, 0.055, 0.055)
        box((4.5, -0.20, -0.4), (2.0, 0.55, 1.5), basalt, bevel=0.18)
        box((4.15, 0.30, -0.75), (1.25, 0.50, 0.95), basalt2, bevel=0.13)
        box((4.75, 0.62, 0.05), (0.75, 0.35, 0.60), basalt, bevel=0.10)
        box((4.75, 1.50, 0.05), (0.36, 0.90, 0.36), plaster, bevel=0.055)
        box((4.75, 1.78, 0.05), (0.368, 0.14, 0.368), (0.55, 0.12, 0.08), bevel=0.02)
        box((4.75, 2.44, 0.05), (0.50, 0.07, 0.50), metal, 7, 0.02)
        box((4.75, 2.62, 0.05), (0.28, 0.17, 0.28), (0.85, 0.50, 0.18), 3, 0.035)
        box((4.75, 2.80, 0.05), (0.36, 0.09, 0.36), (0.35, 0.08, 0.06), bevel=0.03)
        # Half-submerged foreground rocks give the swell a scale reference.
        box((1.9, -0.12, 3.4), (0.95, 0.50, 0.75), basalt, bevel=0.16)
        box((-2.8, -0.15, -2.0), (0.80, 0.45, 0.65), basalt2, bevel=0.14)
        # Fogged silhouettes on the horizon.
        box((-95.0, 1.5, -170.0), (14.0, 12.0, 14.0), basalt, bevel=3.0)
        box((130.0, 0.5, -110.0), (10.0, 8.0, 10.0), basalt2, bevel=2.5)

    @ti.func
    def extra_height(self, x, z, clock):
        """Extra height-field displacement; subclasses override (demo 04 wake)."""
        return 0.0

    @ti.func
    def extra_slope(self, x, z, clock):
        """Extra height-field gradient; subclasses override (demo 04 wake)."""
        return ti.Vector([0.0, 0.0])

    @ti.func
    def foam(self, p, clock, lighting):
        """Additive foam color at a water-surface point; zero by default."""
        return ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def wave_amplitude(self):
        """Height-field amplitude used for march bounds and upwelling lift."""
        return self.spectrum.amp_total[None]

    @ti.func
    def wave_height(self, x, z, clock):
        """Intersection height field; subclasses override (demo 05 FFT)."""
        return self.spectrum.height(x, z, clock)

    @ti.func
    def wave_surface(self, x, z, clock):
        """Height and shading normal at a surface point; subclass override."""
        return self.spectrum.surface(x, z, clock)

    @ti.func
    def wave_bounds(self):
        amplitude = self.wave_amplitude()
        return ti.Vector([-1.02 * amplitude, 1.02 * amplitude])

    @ti.func
    def seabed_depth(self):
        return SEABED

    @ti.func
    def shading_normal(self, p, clock, footprint, geometric_normal):
        fine = self.detail_gradient(p.x, p.z, clock, footprint) + self.extra_slope(p.x, p.z, clock)
        return (geometric_normal + ti.Vector([-fine.x, 0.0, -fine.y])).normalized()

    @ti.func
    def water_body(self, color, thickness, clarity, wave_y, normal, lighting):
        transmission = ti.exp(-ti.Vector([0.34, 0.12, 0.065]) * thickness / clarity)
        through_water = color * transmission + ti.Vector([0.008, 0.030, 0.042]) * (1.0 - transmission)
        lift = 1.0 + 0.5 * wave_y / ti.max(0.05, self.wave_amplitude())
        return through_water * lift

    @ti.func
    def water_depth_limit(self):
        return 12.0

    @ti.func
    def foam_filtered(self, p, clock, lighting, footprint):
        return self.foam(p, clock, lighting)

    @ti.func
    def floor_material(self, p, clock, amplitude):
        # Dark volcanic sand; mostly seen through a long absorbing water path.
        ripple = 0.5 + 0.5 * ti.sin(p.x * 2.1 + 1.3 * ti.sin(p.z * 1.7))
        base = ti.Vector([0.10, 0.11, 0.10]) * (0.8 + 0.3 * ripple)
        normal = ti.Vector([0.05 * ti.sin(p.z * 1.9), 1.0, 0.05 * ti.sin(p.x * 1.6)]).normalized()
        return base, normal

    @ti.func
    def geometry(self, origin, direction):
        closest, material = 1e6, -1
        normal, color = ti.Vector([0.0, 1.0, 0.0]), ti.Vector([0.5, 0.5, 0.5])
        if direction.y < -1e-6:
            t = (self.seabed_depth() - origin.y) / direction.y
            if t > 0.001:
                closest, material = t, 6
        for i in range(self.boxes):
            t, n = self.rounded_box_hit(origin, direction, self.box_center[i],
                                        self.box_extent[i], self.box_bevel[i])
            if t < closest:
                closest, normal = t, n
                color, material = self.box_color[i], self.box_material[i]
        return closest, normal, color, material

    @ti.func
    def water_hit(self, origin, direction, clock):
        """First entering root via error-controlled marching and bisection."""
        t = 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        surf_y = 0.0
        bounds = self.wave_bounds()
        lo, hi = bounds.x, bounds.y
        near, far = 0.001, FAR_PLANE
        if ti.abs(direction.y) < 1e-8:
            if origin.y < lo or origin.y > hi:
                far = -1.0
        else:
            a, b = (lo - origin.y) / direction.y, (hi - origin.y) / direction.y
            near, far = ti.max(near, ti.min(a, b)), ti.min(far, ti.max(a, b))
        if near <= far:
            current = near
            p = origin + direction * current
            h = self.wave_height(p.x, p.z, clock) + self.extra_height(p.x, p.z, clock)
            if p.y <= h:
                t = current
            else:
                count = 0
                while t == 1e6 and current < far and count < MARCH_MAX_STEPS:
                    error = p.y - h
                    limit = MARCH_MIN_STEP + MARCH_GROWTH * current
                    step = ti.min(ti.max(error * 1.4, MARCH_MIN_STEP), limit)
                    next_t = ti.min(current + step, far)
                    p = origin + direction * next_t
                    h = self.wave_height(p.x, p.z, clock) + self.extra_height(p.x, p.z, clock)
                    if p.y <= h:
                        lo_t, hi_t = current, next_t
                        for _ in ti.static(range(BISECT_STEPS)):
                            mid = 0.5 * (lo_t + hi_t)
                            pm = origin + direction * mid
                            hm = self.wave_height(pm.x, pm.z, clock) + self.extra_height(pm.x, pm.z, clock)
                            if pm.y <= hm:
                                hi_t = mid
                            else:
                                lo_t = mid
                        t = 0.5 * (lo_t + hi_t)
                    else:
                        current = next_t
                    count += 1
            if t < 1e6:
                # Bisection changes t: sample at the refined hit, not at the
                # last marching endpoint (which can lie inside the water).
                hit_point = origin + direction * t
                surf_y, normal = self.wave_surface(hit_point.x, hit_point.z, clock)
        return t, normal, surf_y

    @ti.func
    def detail_gradient(self, x, z, clock, footprint):
        gradient = ti.Vector([0.0, 0.0])
        for band in ti.static(range(6)):
            kx, kz, strength, speed = 4.2, 2.6, 0.0060, 1.6
            if ti.static(band == 1):
                kx, kz, strength, speed = 7.8, -5.4, 0.0030, 2.4
            if ti.static(band == 2):
                kx, kz, strength, speed = 13.0, 9.2, 0.0016, 3.2
            if ti.static(band == 3):
                kx, kz, strength, speed = 5.6, -3.1, 0.0022, 2.0
            if ti.static(band == 4):
                kx, kz, strength, speed = 10.5, 3.3, 0.0013, 2.9
            if ti.static(band == 5):
                kx, kz, strength, speed = 18.0, 12.5, 0.0007, 3.6
            wavelength = 2.0 * math.pi / ti.sqrt(kx * kx + kz * kz)
            fade = 1.0 - self.smooth(wavelength * 0.12, wavelength * 0.5, footprint)
            phase = kx * x + kz * z - speed * clock
            gradient += strength * ti.cos(phase) * ti.Vector([kx, kz]) * fade
        return gradient * self.water_controls[None].y

    @ti.func
    def radiance(self, origin, direction, clock, clarity, lighting, pixel_angle):
        opaque_t, opaque_normal, opaque_base, opaque_material = self.geometry(origin, direction)
        water_t, normal, wave_y = self.water_hit(origin, direction, clock)
        has_water = water_t < opaque_t
        p = origin + direction * water_t
        geometric_normal = normal
        footprint = 0.0
        if has_water:
            footprint = water_t * pixel_angle / ti.max(0.15, ti.abs(direction.y))
            normal = self.shading_normal(p, clock, footprint, geometric_normal)
            # A filtered shading normal cannot face away from a visible
            # geometric surface; blend only as much as needed at grazing view.
            view_cos = -direction.dot(normal)
            geom_cos = -direction.dot(geometric_normal)
            if view_cos < 0.001:
                blend = (0.001 - view_cos) / ti.max(1e-6, geom_cos - view_cos)
                normal = ((1 - ti.min(1.0, blend)) * normal + ti.min(1.0, blend) * geometric_normal).normalized()
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
                    ray_origin = p + geometric_normal * 0.004
                else:
                    ray_origin, ray_direction = p - geometric_normal * 0.004, refracted
                t, n, base, material = self.geometry(ray_origin, ray_direction)
            shaded = ti.Vector([0.0, 0.0, 0.0])
            if weight > 0.0 or bounce == reflection_count:
                shaded, _ = self.shade_surface(ray_origin, ray_direction, t, n, base, material, clock, lighting, 0.0, ti.cast(not has_water, ti.i32))
            color = shaded
            if has_water and bounce < reflection_count:
                reflection += shaded * weight / reflection_count
            elif has_water:
                thickness = ti.min(t, self.water_depth_limit())
        if has_water:
            through_water = self.water_body(color, thickness, clarity, wave_y, normal, lighting)
            f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
            fresnel = f0 + (1.0 - f0) * (1.0 - cosine) ** 5
            roughness = ti.max(0.045, self.water_controls[None].x)
            color = through_water * (1.0 - fresnel) + reflection
            glint = ggx_brdf(ti.Vector([0.0, 0.0, 0.0]), roughness, 0.0,
                             normal, -direction, self.sun_direction(lighting), f0)
            if glint.max() > 0.005:
                shade = self.visibility(p, normal, self.sun_direction(lighting), 0)
                color += glint * ti.Vector([1.0, 0.80, 0.52]) * 3.2 * shade
            color += self.foam_filtered(p, clock, lighting, footprint)
        primary_t = ti.min(water_t, opaque_t)
        if primary_t < 1e5:
            # Distance haze toward the sky color. Rays pointing downward fade
            # toward the zenith instead of the azimuth horizon: sampling the
            # horizon pole from a top view sweeps longitude per pixel and
            # paints radial spokes onto the water.
            fog = 1.0 - ti.exp(-3.0 * (primary_t / FOG_DISTANCE) ** 2)
            overhead = ti.max(0.0, -direction.y)
            horizon = ti.Vector([direction.x,
                                 0.02 + 0.65 * overhead * overhead, direction.z])
            fog_color = self.environment.sample_map(self.environment.environment, horizon.normalized(), lighting)
            color = color * (1.0 - fog) + fog_color * fog
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

    def draw(self, camera, clock, clarity=1.4, lighting=0.0, exposure=1.05):
        self.render(*camera.basis(), clock, clarity, lighting,
                    exposure, math.tan(math.radians(camera.fov / 2)), self.samples)
        bloom, threshold, vignette = self.post_controls[None]
        self.post.apply(exposure, bloom, threshold, vignette)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 03: sunset sea with Gerstner waves")
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
    parser.add_argument("--wind", type=float, default=8.0, help="Wind speed in m/s")
    parser.add_argument("--wind-dir", type=float, default=35.0, help="Wind direction in degrees")
    parser.add_argument("--wave-height", type=float, default=1.3, help="Total wave amplitude in metres")
    parser.add_argument("--steepness", type=float, default=0.55, help="Gerstner steepness sum Q k A, 0..1")
    parser.add_argument("--wave-scale", type=float, default=0.8, help="Wavelength scale multiplier")
    parser.add_argument("--preset", choices=("overview", "waterline", "top"), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_03.png"))
    parser.add_argument("--time", type=float, default=2.0, help="Wave warmup in seconds before the first frame")
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
    renderer = SunsetOceanRenderer(args.width, args.height, args.samples)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 0.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = OrbitCamera()
    apply_preset(camera, args.preset)
    wind = (args.wind, args.wind_dir, args.wave_height, args.steepness, args.wave_scale)
    speed = 1.0
    lighting = 0.0 if args.lighting == "sunset" else 1.0
    clarity, exposure = 1.4, 1.05
    clock = max(0.0, args.time)
    renderer.spectrum.rebuild(*wind)

    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            clock += 1.0 / 60.0
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
    window = ti.ui.Window("03 / Sunset Sea", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    frame_ms, window_frames = 0.0, 1
    print("Drag LMB orbit | RMB pan | W/S zoom | 1/2/3 views | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
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
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 0.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                wind = (args.wind, args.wind_dir, args.wave_height, args.steepness, args.wave_scale)
                renderer.spectrum.rebuild(*wind)
                lighting = 0.0 if args.lighting == "sunset" else 1.0
                clarity, exposure, clock = 1.4, 1.05, 0.0
                paused, auto_orbit = False, False
            elif key == "p":
                save_frame = True
        if not window.running:
            break
        if previous_mouse is not None and not in_panel and (window.is_pressed(ti.ui.LMB) or window.is_pressed(ti.ui.RMB)):
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
            with gui.sub_window("SUNSET SEA / 03", 0.02, 0.025, 0.30, 0.78):
                gui.text("Gerstner waves / dispersion / steepness")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
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
                gui.text("Drag LMB orbit / RMB pan")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        if paused:
            if step:
                clock += 1.0 / 60.0
        else:
            clock += dt * speed
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
