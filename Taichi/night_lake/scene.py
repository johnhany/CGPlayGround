"""Night lake scene: night sky, moon, stars, wind waves, rain ripples, shore
terrain, fog.

Primary rays march the combined water height field (small wind waves plus
the rain-ripple simulation) and the procedural beach height field; the
nearer of the two crossings wins. Water shading is reflection dominated:
the prefiltered night environment sampled along the reflected ray, weighted
by Fresnel, a dark water body underneath, and a GGX moon glint forming
the glitter band. Terrain shading zones wet mud, soil, and grass by height
and slope, with a moonlit sheen on the wet band. Sky rays sample the baked
night environment with the sharp star field and moon disk added on top.
Output goes through the shared PostProcess bloom / tone mapping chain.
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image
import taichi as ti

from water.pbr import EnvironmentLighting, convolve_environment, direction_grid
from night_lake.accel import (SRC_BLADE, SRC_GRASS, SRC_LEAF, SRC_STRIDE,
                              TILE_PIXELS, WIND_MARGIN_MAX, VegetationAccel)
from night_lake.camera import FixedCamera
from night_lake.finish import NightPostProcess, hash01
from night_lake.moon import (MOON_DIRECTION, MOON_TINT, MOON_DIRECT_INTENSITY,
                             MOON_FOLIAGE_DIRECT, MOON_BARK_DIRECT,
                             MOON_IBL_BOOST, MOON_GLINT_INTENSITY,
                             MOON_GLOW_WIDE_POWER, MOON_GLOW_WIDE,
                             MOON_GLOW_TIGHT_POWER, MOON_GLOW_TIGHT,
                             night_environment)
from night_lake.shadow import RES_U, RES_V, bake_moon_shadow, shadow_factor
from night_lake.sky import night_sky
from night_lake.lake import BASE_INJECTION_STRENGTH, LakeWaves, RainRipples
from night_lake.grass import GrassField, blade_hit_test, card_hit_test
from night_lake.rain import RainStreaks, SplashBursts
from night_lake.trees import TreeField
from night_lake.terrain import (DEEP_WATER, FOG_COLOR, GRASS_LINE_DEFAULT,
                                SHORE_Z_DEFAULT, WET_SCALE_DEFAULT,
                                terrain_gradient, terrain_height,
                                terrain_height_np, terrain_material,
                                terrain_mottle)

__all__ = ["NightLakeScene"]

# Marching: the step grows with distance; bisection refines the crossing.
MARCH_MIN_STEP = 0.01
MARCH_GROWTH = 0.06
MARCH_MAX_STEPS = 220
BISECT_STEPS = 8
FAR_PLANE = 400.0
# Distance haze toward the horizon fog color.
FOG_DISTANCE = 140.0
# Ripple influence fades with distance to avoid far-field flicker.
RIPPLE_FADE_DISTANCE = 25.0
# Terrain marching: the beach is smooth, so fewer steps at larger growth.
TERRAIN_MIN_STEP = 0.03
TERRAIN_GROWTH = 0.12
TERRAIN_MAX_STEPS = 160
TERRAIN_BISECT_STEPS = 8
TERRAIN_FAR_PLANE = 150.0
# Shading normals amplify the height-field gradient so the fbm relief reads
# under the nearly overhead moon; geometry marching uses the true gradient.
TERRAIN_BUMP = 3.0
# The vegetation near-field is baked for the fixed production viewpoint.
PRODUCTION_EYE = np.array([0.0, 6.0, 0.0], dtype=np.float64)
# Distant rain curtain slab: above the water from just past the waterline
# to the far bank, tall enough to read against the sky and treeline.
CURTAIN_Z0, CURTAIN_Z1 = 20.0, 75.0
CURTAIN_HEIGHT = 7.0
CURTAIN_COLOR = (0.030, 0.038, 0.052)
# Temporal accumulation alphas per object class: sky and terrain hold a
# 12-frame exponential window, swaying vegetation a 3-frame one, and the
# water surface a 2-frame one so waves and ripples never ghost.
ACCUM_ALPHA_STATIC = 1.0 / 12.0
ACCUM_ALPHA_VEGETATION = 1.0 / 3.0
ACCUM_ALPHA_WATER = 0.5
# Low-discrepancy sub-pixel jitter directions, one per grain frame.
JITTER_GX = 0.7548776662
JITTER_GY = 0.5698402909


def _smoothstep(edge0, edge1, x):
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


@ti.data_oriented
class NightLakeScene:
    """Night lake with a ray-marched water surface and sky dome."""

    def __init__(self, width, height):
        self.environment = EnvironmentLighting()
        self._rebake_night_environment()
        self.width, self.height = width, height
        self.post = NightPostProcess(width, height)
        self.image = self.post.image
        self.post_controls = ti.Vector.field(4, ti.f32, shape=())
        # Bloom strength, bloom threshold, vignette, film grain. Tuned for a
        # night photograph: moon and glitter lift into bloom, the sky stays
        # below the threshold, corners fall off like lens shading.
        self.post_controls[None] = [0.12, 0.9, 0.22, 0.025]
        self.camera_controls = ti.Vector.field(4, ti.f32, shape=())
        # Thin-lens depth of field: enabled, aperture radius, focus
        # distance, sample count. Disabled by default: at 129-147 ms per
        # frame it misses the frame budget, so the pinhole path is the
        # production path and the lens path runs only on demand. The focus
        # plane rests on the waterline band (~14 m deep) so a moderate
        # aperture keeps the shore and glitter path crisp while the far
        # bank softens.
        self.camera_controls[None] = [0.0, 0.03, 14.0, 4.0]
        self.water_controls = ti.Vector.field(4, ti.f32, shape=())
        # Wave amplitude scale, ripple strength, rain rate (drops/s), roughness.
        self.water_controls[None] = [1.0, 1.0, 40.0, 0.06]
        self.terrain_controls = ti.Vector.field(4, ti.f32, shape=())
        # Shoreline distance, detail amplitude scale, wetness decay, grass line.
        self.terrain_controls[None] = [SHORE_Z_DEFAULT, 1.0, WET_SCALE_DEFAULT, GRASS_LINE_DEFAULT]
        self.vegetation_controls = ti.Vector.field(4, ti.f32, shape=())
        # Grass density, tree density, wind strength, leaf brightness.
        self.vegetation_controls[None] = [1.0, 1.0, 0.6, 1.0]
        self.rain_controls = ti.Vector.field(4, ti.f32, shape=())
        # Streak strength, splash amount, curtain strength, unused.
        self.rain_controls[None] = [1.0, 1.0, 0.8, 0.0]
        # Staging vector for the whole control block: the UI commits all 24
        # floats in one host-to-device write, draw() reads them back in one
        # device-to-host copy, and one small kernel keeps the staging block
        # and the individual fields in sync on the device.
        self.controls_stage = ti.Vector.field(24, ti.f32, shape=())
        self._controls_dirty = False
        self._sync_controls(0)
        self.waves = LakeWaves()
        self.ripples = RainRipples()
        self.grass = GrassField(PRODUCTION_EYE)
        self.trees = TreeField()
        self.streaks = RainStreaks(width, height)
        self.splash = SplashBursts()
        # Per-pixel classification for tests and debugging: 0 sky, 1 water,
        # 2 terrain, 3 grass, 4 tree.
        self.object_id = ti.field(ti.i32, shape=(width, height))
        # Primary-hit distance per pixel; the rain curtain clamps against it
        # so the veil never crosses visible geometry.
        self.hit_distance = ti.field(ti.f32, shape=(width, height))
        self.accel = self._build_acceleration()
        # Baked moon shadow map (light-space depth, see shadow.py). The
        # bake is cached on its inputs, so re-uploads after a terrain
        # control change are cheap.
        self.shadow_depth = ti.field(ti.f32, shape=(RES_U, RES_V))
        self.shadow_depth.from_numpy(bake_moon_shadow(self.trees, self.grass,
                                                      SHORE_Z_DEFAULT, 1.0))
        self._shadow_key = (SHORE_Z_DEFAULT, 1.0)
        # Quarter-resolution planar reflection cache: the mirrored scene
        # above the water plane, re-rendered every frame (it holds swaying
        # vegetation and rain), sampled by shade_water through the
        # reflected ray. Quarter resolution keeps the depth-of-field path
        # inside the frame budget; the water surface is rippled, so the
        # soft cache reads as roughness rather than missing detail.
        self.refl_width = max(2, width // 4)
        self.refl_height = max(2, height // 4)
        self.refl_color = ti.Vector.field(3, ti.f32,
                                          shape=(self.refl_width,
                                                 self.refl_height))
        self.reflection_gain = 1.0
        # Temporal accumulation (jittered supersampling): the jittered HDR
        # frame blends into this buffer with a per-class alpha and the
        # buffer is copied back over the frame. Static classes get a long
        # window; water and swaying vegetation get short windows so motion
        # never smears. The buffer resets when the clock does not advance
        # (pauses, seeks) or when controls are committed.
        self.accum_color = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.accum_frame_flag = ti.field(ti.i32, shape=())
        self.temporal_aa = True
        self._accum_valid = False
        self._last_accum_clock = None

    def _build_acceleration(self):
        """Screen-tile candidate lists and the world uniform grid over all
        vegetation instances, built for the fixed production camera."""
        entries = []
        cards = self.grass.cards
        centres = cards.card_center.to_numpy()[:cards.count]
        us = cards.card_u.to_numpy()[:cards.count]
        vs = cards.card_v.to_numpy()[:cards.count]
        for i in range(cards.count):
            sway = 0.10 * WIND_MARGIN_MAX * float(np.linalg.norm(vs[i]))
            radius = float(np.linalg.norm(us[i]) + np.linalg.norm(vs[i])) + sway
            entries.append((SRC_GRASS * SRC_STRIDE + i, centres[i], radius))
        if self.grass.blade_count:
            verts = self.grass.blade_vertex.to_numpy()[:self.grass.blade_count]
            for i in range(self.grass.blade_count):
                centre = verts[i].mean(axis=0)
                reach = float(np.linalg.norm(verts[i] - centre, axis=1).max())
                entries.append((SRC_BLADE * SRC_STRIDE + i, centre,
                                reach + 0.05 * WIND_MARGIN_MAX))
        leaves = self.trees.leaves
        centres = leaves.card_center.to_numpy()[:leaves.count]
        us = leaves.card_u.to_numpy()[:leaves.count]
        vs = leaves.card_v.to_numpy()[:leaves.count]
        for i in range(leaves.count):
            sway = 0.10 * WIND_MARGIN_MAX * float(np.linalg.norm(vs[i]))
            radius = float(np.linalg.norm(us[i]) + np.linalg.norm(vs[i])) + sway
            entries.append((SRC_LEAF * SRC_STRIDE + i, centres[i], radius))
        items = np.array([e[0] for e in entries], dtype=np.int32)
        centres = np.array([e[1] for e in entries], dtype=np.float64)
        radii = np.array([e[2] for e in entries], dtype=np.float64)
        camera = FixedCamera()
        eye, forward, right, up = camera.basis()
        return VegetationAccel(self.width, self.height, eye, forward, right,
                               up, camera.fov, items, centres, radii)

    @ti.kernel
    def _sync_controls(self, unpack: ti.i32):
        # One kernel keeps the 24-float staging block and the six 4-float
        # control fields in sync, so a frame costs one small device copy
        # plus one host round trip instead of six. unpack == 1 pushes a
        # committed UI state from the staging block into the fields first.
        if unpack == 1:
            s = self.controls_stage[None]
            self.water_controls[None] = [s[0], s[1], s[2], s[3]]
            self.camera_controls[None] = [s[4], s[5], s[6], s[7]]
            self.post_controls[None] = [s[8], s[9], s[10], s[11]]
            self.rain_controls[None] = [s[12], s[13], s[14], s[15]]
            self.terrain_controls[None] = [s[16], s[17], s[18], s[19]]
            self.vegetation_controls[None] = [s[20], s[21], s[22], s[23]]
        w = self.water_controls[None]
        c = self.camera_controls[None]
        p = self.post_controls[None]
        r = self.rain_controls[None]
        t = self.terrain_controls[None]
        v = self.vegetation_controls[None]
        self.controls_stage[None] = [w.x, w.y, w.z, w.w,
                                     c.x, c.y, c.z, c.w,
                                     p.x, p.y, p.z, p.w,
                                     r.x, r.y, r.z, r.w,
                                     t.x, t.y, t.z, t.w,
                                     v.x, v.y, v.z, v.w]

    def controls_snapshot(self):
        """The whole 24-float control block as one numpy vector."""
        return self.controls_stage[None].to_numpy()

    def commit_controls(self, state):
        """Commit a UI-edited control block; the next draw unpacks it and
        restarts temporal accumulation, since the committed parameters
        change what every pixel should converge to."""
        state = np.asarray(state, dtype=np.float32).reshape(24)
        self.controls_stage[None] = state
        self._controls_dirty = True
        self._accum_valid = False

    def _rebake_night_environment(self):
        """Rebuild the night variant (index 2) of the baked environment maps
        against the production moon direction so IBL, the visible disk, and
        the water glitter path stay aligned. The day and sunset variants,
        and the BRDF lookup table, are reused unchanged."""
        sampler = night_environment
        sky = self.environment.environment.to_numpy()
        sky[2] = sampler(direction_grid(128, 64))
        irradiance = self.environment.irradiance.to_numpy()
        irradiance[2] = convolve_environment(direction_grid(32, 16), sampler=sampler)
        prefiltered = self.environment.prefiltered.to_numpy()
        for level in range(5):
            prefiltered[2, level] = convolve_environment(
                direction_grid(64, 32), roughness=level / 4, sampler=sampler)
        self.environment.environment.from_numpy(sky)
        self.environment.irradiance.from_numpy(irradiance)
        self.environment.prefiltered.from_numpy(prefiltered)

    @ti.func
    def _moon_vec(self):
        return ti.Vector([
            float(MOON_DIRECTION[0]),
            float(MOON_DIRECTION[1]),
            float(MOON_DIRECTION[2]),
        ])

    @ti.func
    def _moon_shadow(self, p, n):
        return shadow_factor(self.shadow_depth, p, n)

    @ti.func
    def _specular_lobe(self, roughness, normal, view, light, f0):
        """Specular-only Cook-Torrance term of water/pbr.ggx_brdf; the
        diffuse part is added separately as hand-tuned moon diffuse, so
        the two must not be summed twice."""
        result = ti.Vector([0.0, 0.0, 0.0])
        nv, nl = normal.dot(view), normal.dot(light)
        if nv > 0.0 and nl > 0.0:
            half_vector = (view + light).normalized()
            nh = ti.max(0.0, normal.dot(half_vector))
            vh = ti.max(0.0, view.dot(half_vector))
            alpha = ti.max(0.045, roughness) ** 2
            alpha2 = alpha * alpha
            denominator = nh * nh * (alpha2 - 1.0) + 1.0
            distribution = alpha2 / (math.pi * denominator * denominator + 1e-8)
            gv = 2.0 * nv / (nv + ti.sqrt(alpha2 + (1.0 - alpha2) * nv * nv))
            gl = 2.0 * nl / (nl + ti.sqrt(alpha2 + (1.0 - alpha2) * nl * nl))
            fresnel = f0 + (1.0 - f0) * (1.0 - vh) ** 5
            result = distribution * gv * gl * fresnel / ti.max(4.0 * nv * nl, 1e-6) * nl
        return result

    @ti.func
    def ground_height(self, x, z):
        return terrain_height(x, z, self.terrain_controls[None].x, self.terrain_controls[None].y)

    @ti.func
    def ripple_fade(self, distance):
        r = distance / RIPPLE_FADE_DISTANCE
        return 1.0 / (1.0 + r * r)

    @ti.func
    def wave_height(self, x, z, t, fade):
        amp_scale = self.water_controls[None].x
        ripple_scale = self.water_controls[None].y
        return (self.waves.height(x, z, t) * amp_scale
                + self.ripples.sample(x, z).x * ripple_scale * fade)

    @ti.func
    def wave_surface(self, x, z, t, fade):
        amp_scale = self.water_controls[None].x
        ripple_scale = self.water_controls[None].y
        grad = self.waves.gradient(x, z, t) * amp_scale
        ripple = self.ripples.sample(x, z)
        grad += ripple.yz * ripple_scale * fade
        height = self.waves.height(x, z, t) * amp_scale + ripple.x * ripple_scale * fade
        return height, grad

    @ti.func
    def filtered_gradient(self, x, z, t, fade, radius):
        # Two-point box filter over the pixel footprint so distant ripples
        # blend into the wave normal instead of popping per pixel.
        total = ti.Vector([0.0, 0.0])
        for a, b in ti.static(ti.ndrange(2, 2)):
            offset = ti.Vector([2.0 * a - 1.0, 2.0 * b - 1.0]) * radius
            total += self.waves.gradient(x + offset.x, z + offset.y, t) * self.water_controls[None].x
            total += self.ripples.sample(x + offset.x, z + offset.y).yz * self.water_controls[None].y * fade
        return total * 0.25

    @ti.func
    def water_hit(self, origin, direction, t):
        """First entering root via a calm-surface bracket and bisection.

        Wave and ripple displacement stays inside a thin slab around the
        calm surface, so the first crossing is always bracketed around the
        plane intersection t0 = -origin.y / direction.y. The previous
        error-controlled march crawled at the minimum step on grazing
        rays and exhausted its step budget, which capped the visible lake
        well inside the far bank and read as sky."""
        hit_t = 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        amp = 0.05 * self.water_controls[None].x + 0.05 * self.water_controls[None].y + 0.005
        lo, hi = -1.02 * amp, 1.02 * amp
        near, far = 0.001, FAR_PLANE
        a, b = (lo - origin.y) / direction.y, (hi - origin.y) / direction.y
        near, far = ti.max(near, ti.min(a, b)), ti.min(far, ti.max(a, b))
        if near <= far and direction.y < -1e-5:
            t0 = (0.0 - origin.y) / direction.y
            margin = 1.05 * amp / ti.max(ti.abs(direction.y), 1e-4) + MARCH_MIN_STEP
            lo_t = ti.max(near, t0 - margin)
            hi_t = ti.min(far, t0 + margin)
            if lo_t <= hi_t:
                found = 0
                root = 1e6
                current = lo_t
                prev_t = lo_t
                prev_f = self._water_error(origin, direction, prev_t, t)
                if prev_f <= 0.0:
                    # Already under the surface at the bracket start; the
                    # entering root sits between the slab entry and here.
                    lo_b, hi_b = near, lo_t
                    for _ in ti.static(range(BISECT_STEPS)):
                        mid = 0.5 * (lo_b + hi_b)
                        if self._water_error(origin, direction, mid, t) <= 0.0:
                            hi_b = mid
                        else:
                            lo_b = mid
                    root = 0.5 * (lo_b + hi_b)
                    found = 1
                step = (hi_t - lo_t) / 8.0
                for _ in range(8):
                    if found == 0:
                        current = prev_t + step
                        next_f = self._water_error(origin, direction, current, t)
                        if next_f <= 0.0:
                            lo_b, hi_b = prev_t, current
                            for _ in ti.static(range(BISECT_STEPS)):
                                mid = 0.5 * (lo_b + hi_b)
                                if self._water_error(origin, direction, mid, t) <= 0.0:
                                    hi_b = mid
                                else:
                                    lo_b = mid
                            root = 0.5 * (lo_b + hi_b)
                            found = 1
                        prev_t = current
                if found == 1:
                    hit_t = root
            if hit_t < 1e6:
                hit_point = origin + direction * hit_t
                _, grad = self.wave_surface(hit_point.x, hit_point.z, t,
                                             self.ripple_fade(hit_t))
                normal = ti.Vector([-grad.x, 1.0, -grad.y]).normalized()
        return hit_t, normal

    @ti.func
    def _water_error(self, origin, direction, tt, t):
        p = origin + direction * tt
        return p.y - self.wave_height(p.x, p.z, t, self.ripple_fade(tt))

    @ti.func
    def shade_water(self, direction, t, normal, footprint, origin,
                    proj_forward, proj_right, proj_up, proj_scale,
                    refl_gain):
        cosine = ti.max(0.0, ti.min(1.0, -direction.dot(normal)))
        f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
        fresnel = f0 + (1.0 - f0) * (1.0 - cosine) ** 5
        roughness = ti.max(0.02, self.water_controls[None].w)
        reflected = (direction - 2.0 * direction.dot(normal) * normal).normalized()
        hit_point = origin + direction * t
        shadow = self._moon_shadow(hit_point, normal)
        # Planar reflection: the quarter-res cache holds the mirrored scene
        # rendered with the same camera but negated up; the reflected ray
        # is mirrored back into cache space and projected with the real
        # camera basis, y negated because the cache image is flipped. The
        # prefiltered environment stays as the rough and distant fallback,
        # blended in as the cache weight fades.
        weight = refl_gain * self.ripple_fade(t) * ti.max(
            0.3, ti.min(1.0, 1.0 - (roughness - 0.04) * 4.0))
        environment_reflection = self.environment.specular(reflected, roughness, 2.0)
        reflection = environment_reflection
        if weight > 0.001 and reflected.y > 1e-4:
            mirrored = ti.Vector([reflected.x, -reflected.y, reflected.z])
            denom = ti.max(1e-4, mirrored.dot(proj_forward))
            tanx = mirrored.dot(proj_right) / denom
            tany = -mirrored.dot(proj_up) / denom
            rw = ti.cast(self.refl_width, ti.f32)
            rh = ti.cast(self.refl_height, ti.f32)
            aspect = rw / rh
            px = (tanx / (aspect * proj_scale) + 1.0) * 0.5 * rw - 0.5
            py = (tany / proj_scale + 1.0) * 0.5 * rh - 0.5
            if 0.0 <= px and px <= rw - 1.0 and 0.0 <= py and py <= rh - 1.0:
                ix, iy = ti.cast(px, ti.i32), ti.cast(py, ti.i32)
                fx = px - ti.cast(ix, ti.f32)
                fy = py - ti.cast(iy, ti.f32)
                x1 = ti.min(ix + 1, self.refl_width - 1)
                y1 = ti.min(iy + 1, self.refl_height - 1)
                cache = (self.refl_color[ix, iy] * ((1.0 - fx) * (1.0 - fy))
                         + self.refl_color[x1, iy] * (fx * (1.0 - fy))
                         + self.refl_color[ix, y1] * ((1.0 - fx) * fy)
                         + self.refl_color[x1, y1] * (fx * fy))
                reflection = cache * weight + environment_reflection * (1.0 - weight)
        # The night water body is nearly black; refraction contributes little.
        deep = ti.Vector([0.004, 0.009, 0.013])
        color = reflection * fresnel + deep * (1.0 - fresnel)
        moon = self._moon_vec()
        moon_tint = ti.Vector(list(MOON_TINT))
        glint = self._specular_lobe(roughness, normal, -direction, moon, f0)
        color += glint * moon_tint * MOON_GLINT_INTENSITY * shadow
        # Residual analytic glow: the broad glitter column now comes from
        # the mirrored scene in the cache, so only a reduced lobe remains,
        # keeping the band readable under ripple chop.
        alignment = ti.max(0.0, reflected.dot(moon))
        moon_glow = (alignment ** MOON_GLOW_WIDE_POWER * MOON_GLOW_WIDE
                     + alignment ** MOON_GLOW_TIGHT_POWER * MOON_GLOW_TIGHT * 0.5)
        color += moon_tint * moon_glow * fresnel * shadow
        fog = 1.0 - ti.exp(-3.0 * (t / FOG_DISTANCE) ** 2)
        fog_color = ti.Vector([0.020, 0.028, 0.042])
        return color * (1.0 - fog) + fog_color * fog

    @ti.func
    def terrain_hit(self, origin, direction):
        """First terrain crossing via error-controlled marching and bisection."""
        hit_t = 1e6
        grad = ti.Vector([0.0, 0.0])
        # The beach tops out below the camera, so rising rays never hit it.
        if direction.y < 0.0:
            current = 0.05
            step = TERRAIN_MIN_STEP
            p = origin + direction * current
            error = p.y - self.ground_height(p.x, p.z)
            count = 0
            while error > 0.0 and current < TERRAIN_FAR_PLANE and count < TERRAIN_MAX_STEPS:
                limit = TERRAIN_MIN_STEP + TERRAIN_GROWTH * current
                step = ti.min(ti.max(error * 1.2, TERRAIN_MIN_STEP), limit)
                current += step
                p = origin + direction * current
                error = p.y - self.ground_height(p.x, p.z)
                count += 1
            # A non-positive error on the very first sample means the eye is
            # already inside the terrain; no crossing exists ahead.
            if error <= 0.0 and count > 0:
                lo_t, hi_t = ti.max(0.001, current - step), current
                for _ in ti.static(range(TERRAIN_BISECT_STEPS)):
                    mid = 0.5 * (lo_t + hi_t)
                    pm = origin + direction * mid
                    if pm.y <= self.ground_height(pm.x, pm.z):
                        hi_t = mid
                    else:
                        lo_t = mid
                hit_t = 0.5 * (lo_t + hi_t)
                hit_point = origin + direction * hit_t
                grad = terrain_gradient(hit_point.x, hit_point.z,
                                        self.terrain_controls[None].x,
                                        self.terrain_controls[None].y)
        return hit_t, grad

    @ti.func
    def shade_terrain(self, direction, t, grad, hit_point):
        height = hit_point.y
        controls = self.terrain_controls[None]
        slope = grad.norm()
        normal = ti.Vector([-grad.x * TERRAIN_BUMP, 1.0, -grad.y * TERRAIN_BUMP]).normalized()
        wet, grass, roughness = terrain_material(height, slope, controls.z, controls.w)
        # Wet ground keeps a minimum roughness so the sheen below spreads
        # into a soft sheet instead of clipping into white patches.
        roughness = ti.max(roughness, 0.30 * wet)
        moon = self._moon_vec()
        moon_tint = ti.Vector(list(MOON_TINT))
        view = -direction
        shadow = self._moon_shadow(hit_point, normal)
        # Bare wet mud below and near the waterline, grass-root soil above.
        mud = ti.Vector([0.075, 0.062, 0.048]) * (1.0 - 0.6 * wet)
        soil = ti.Vector([0.055, 0.058, 0.038])
        base = mud * (1.0 - grass) + soil * grass
        # Large-scale albedo patchiness breaks up the flat tone.
        base *= 0.86 + 0.28 * (terrain_mottle(hit_point.x, hit_point.z) + 0.5)
        depth = ti.max(0.0, -height)
        if depth > 0.0:
            # Shallow water cover: darken and shift toward the water body.
            sunk = 1.0 - ti.exp(-depth * 6.0)
            base = base * (1.0 - sunk) + ti.Vector(list(DEEP_WATER)) * sunk
        irradiance = self.environment.sample_map(self.environment.irradiance, normal, 2.0)
        color = base * irradiance * MOON_IBL_BOOST
        # Directional moon diffuse is what models the relief: the fbm normal
        # variation shows up as light and shade across the beach.
        nl = ti.max(0.0, normal.dot(moon))
        color += base * nl * moon_tint * MOON_DIRECT_INTENSITY * shadow
        f0 = 0.04
        cosine = ti.max(0.0, view.dot(normal))
        # The glint stays modest: at grazing angles a rough GGX lobe wraps
        # around the whole frame and reads as haze instead of moonlight.
        specular = self._specular_lobe(roughness, normal, view, moon, f0)
        color += specular * moon_tint * 0.5 * shadow
        # Wet sheen: the environment and the moon glow reflect off the film
        # of water on the mud. The glow lobe is tight so the sheen reads as
        # a moon column on the wet band rather than a uniform lift.
        reflected = (direction - 2.0 * direction.dot(normal) * normal).normalized()
        sheen = self.environment.specular(reflected, roughness, 2.0)
        fresnel = f0 + (1.0 - f0) * (1.0 - cosine) ** 5
        alignment = ti.max(0.0, reflected.dot(moon))
        # A medium lobe gives the wet band its mirror sheen under the moon
        # column; a tight lobe adds sparkle where the mirror geometry aligns.
        moon_glow = alignment ** 16 * 0.45 \
            + alignment ** MOON_GLOW_TIGHT_POWER * MOON_GLOW_TIGHT
        wet_sheen = 0.35 + 0.65 * wet
        sheen_total = (sheen + moon_tint * moon_glow) * fresnel * wet_sheen * shadow
        # Luminance clamp: grazing wet mud used to blow out into solid
        # white sheets. The cap keeps every channel below clipping while
        # the tight lobe preserves sparkle near the waterline.
        sheen_lum = sheen_total.x * 0.3 + sheen_total.y * 0.6 + sheen_total.z * 0.1
        if sheen_lum > 0.28:
            sheen_total *= 0.28 / sheen_lum
        color += sheen_total
        fog = 1.0 - ti.exp(-3.0 * (t / FOG_DISTANCE) ** 2)
        fog_color = ti.Vector(list(FOG_COLOR))
        return color * (1.0 - fog) + fog_color * fog

    @ti.func
    def vegetation_hit_tiled(self, origin, direction, clock, t_max, tile):
        """Tile-candidate vegetation scan; identical intersection math to
        vegetation_hit, restricted to the pixel tile's candidate list."""
        controls = self.vegetation_controls[None]
        best_t, kind, index, a, b = 1e6, 0, -1, 0.0, 0.0
        tiles_x = self.accel.tiles_x
        tiles_y = self.accel.tiles_y
        if 0 <= tile.x < tiles_x and 0 <= tile.y < tiles_y:
            tid = tile.y * tiles_x + tile.x
            for slot in range(self.accel.tile_start[tid],
                              self.accel.tile_start[tid + 1]):
                item = self.accel.tile_item[slot]
                source = item // SRC_STRIDE
                i = item - source * SRC_STRIDE
                if source == SRC_GRASS and controls.x > 0.0:
                    t, ca, cb = card_hit_test(self.grass.cards, i, origin,
                                              direction, controls.x, best_t)
                    if t < best_t:
                        best_t, kind, index, a, b = t, 1, i, ca, cb
                elif source == SRC_BLADE and controls.x > 0.0:
                    t, bv = blade_hit_test(self.grass, i, origin, direction,
                                           best_t)
                    if t < best_t:
                        best_t, kind, index, a, b = t, 2, i, 0.0, bv
                elif source == SRC_LEAF:
                    t, ca, cb = card_hit_test(self.trees.leaves, i, origin,
                                              direction, controls.y, best_t)
                    if t < best_t:
                        best_t, kind, index, a, b = t, 3, i, ca, cb
        return best_t, kind, index, a, b

    @ti.func
    def vegetation_hit(self, origin, direction, clock, t_max):
        """Closest vegetation surface: grass cards, leaf cards, near blades.

        Returns (t, kind, index, a, b): kind 1 grass card, 2 analytic blade,
        3 leaf card; a and b are the card coordinates for cards and the
        along-blade parameter for blades.
        """
        controls = self.vegetation_controls[None]
        best_t, kind, index, a, b = 1e6, 0, -1, 0.0, 0.0
        if controls.x > 0.0:
            t, i, ca, cb = self.grass.cards.cards_hit(origin, direction, controls.x, t_max)
            if t < best_t:
                best_t, kind, index, a, b = t, 1, i, ca, cb
            t, i, bv = self.grass.blades_hit(origin, direction, best_t)
            if t < best_t:
                best_t, kind, index, a, b = t, 2, i, 0.0, bv
        t, i, ca, cb = self.trees.leaves.cards_hit(origin, direction, controls.y, best_t)
        if t < best_t:
            best_t, kind, index, a, b = t, 3, i, ca, cb
        return best_t, kind, index, a, b

    @ti.func
    def shade_leaf_surface(self, direction, t, normal, v_along, tint,
                           base_color, brightness, origin, ao):
        """Shared night shading for grass blades and leaf cards."""
        moon = self._moon_vec()
        if normal.dot(-direction) < 0.0:
            normal = -normal
        hit_point = origin + direction * t
        shadow = self._moon_shadow(hit_point, normal)
        base = base_color * tint * (0.45 + 0.75 * v_along)
        irradiance = self.environment.sample_map(self.environment.irradiance, normal, 2.0)
        color = base * irradiance * MOON_IBL_BOOST
        moon_tint = ti.Vector(list(MOON_TINT))
        nl = ti.max(0.0, normal.dot(moon))
        color += base * nl * moon_tint * MOON_DIRECT_INTENSITY * MOON_FOLIAGE_DIRECT * shadow
        # Translucent backlit: light bleeds through when the moon is behind.
        color += base * moon_tint * ti.max(0.0, -normal.dot(moon)) * 0.45
        specular = self._specular_lobe(0.6, normal, -direction, moon, 0.04)
        color += specular * moon_tint * 0.25 * shadow
        # Root contact darkening: the baked card occlusion grounds the
        # card; the vertical term keeps the upper half untouched.
        ao_mix = ao * (0.62 + 0.38 * v_along)
        fog = 1.0 - ti.exp(-3.0 * (t / FOG_DISTANCE) ** 2)
        fog_color = ti.Vector(list(FOG_COLOR))
        return color * brightness * ao_mix * (1.0 - fog) + fog_color * fog

    @ti.func
    def shade_grass_card(self, direction, t, index, a, b, origin):
        cards = self.grass.cards
        normal = cards.card_u[index].cross(cards.card_v[index]).normalized()
        v_along = (b + 1.0) * 0.5
        tint_noise = ti.sin(float(index) * 91.7) * 43758.5453
        tint = 0.78 + 0.44 * (tint_noise - ti.floor(tint_noise))
        # Darker midrib near the card centreline, fading toward the tip.
        rib = 1.0 - 0.30 * (1.0 - ti.min(1.0, ti.abs(a) * 3.0)) * (1.0 - 0.6 * v_along)
        dry = cards.card_dry[index]
        lush = ti.Vector([0.042, 0.066, 0.024])
        straw = ti.Vector([0.105, 0.082, 0.030])
        base = (lush * (1.0 - dry) + straw * dry) * rib
        return self.shade_leaf_surface(direction, t, normal, v_along, tint,
                                       base, self.vegetation_controls[None].w,
                                       origin, cards.card_ao[index])

    @ti.func
    def shade_leaf_card(self, direction, t, index, a, b, origin):
        cards = self.trees.leaves
        normal = cards.card_u[index].cross(cards.card_v[index]).normalized()
        v_along = (b + 1.0) * 0.5
        tint_noise = ti.sin(float(index) * 57.3) * 43758.5453
        tint = 0.72 + 0.56 * (tint_noise - ti.floor(tint_noise))
        base = ti.Vector([0.036, 0.064, 0.024])
        return self.shade_leaf_surface(direction, t, normal, v_along, tint,
                                       base, self.vegetation_controls[None].w,
                                       origin, 1.0)

    @ti.func
    def shade_blade(self, direction, t, index, v_along, origin):
        v0 = self.grass.blade_vertex[index, 0]
        v1 = self.grass.blade_vertex[index, 1]
        v2 = self.grass.blade_vertex[index, 2]
        normal = (v1 - v0).cross(v2 - v0).normalized()
        meta = self.grass.blade_meta[index]
        lush = ti.Vector([0.042, 0.066, 0.024])
        straw = ti.Vector([0.105, 0.082, 0.030])
        base = lush * (1.0 - meta.z) + straw * meta.z
        return self.shade_leaf_surface(direction, t, normal, v_along, meta.y,
                                       base, self.vegetation_controls[None].w,
                                       origin, 1.0)

    @ti.func
    def shade_tree(self, direction, t, index, normal, origin):
        p = origin + direction * t
        grain = ti.sin((p.x + p.y * 1.7 + p.z * 0.6) * 34.0) * 43758.5453
        base = ti.Vector([0.062, 0.048, 0.036]) * (0.82 + 0.36 * (grain - ti.floor(grain)))
        # Bark tone drifts over metres, so trunks pick up per-tree
        # variation from the low-frequency terrain mottle.
        base *= 0.78 + 0.5 * (terrain_mottle(p.x * 0.6, p.z * 0.6) + 0.5)
        moon = self._moon_vec()
        shadow = self._moon_shadow(p, normal)
        irradiance = self.environment.sample_map(self.environment.irradiance, normal, 2.0)
        color = base * irradiance * MOON_IBL_BOOST
        moon_tint = ti.Vector(list(MOON_TINT))
        nl = ti.max(0.0, normal.dot(moon))
        color += base * nl * moon_tint * MOON_DIRECT_INTENSITY * MOON_BARK_DIRECT * shadow
        specular = self._specular_lobe(0.85, normal, -direction, moon, 0.04)
        color += specular * moon_tint * 0.3 * shadow
        # Contact darkening toward the ground line so trunks sit in the
        # scene instead of floating above it.
        contact = 0.55 + 0.45 * ti.min(1.0, ti.max(0.0, (p.y - 0.1) / 1.1))
        fog = 1.0 - ti.exp(-3.0 * (t / FOG_DISTANCE) ** 2)
        fog_color = ti.Vector(list(FOG_COLOR))
        return color * contact * (1.0 - fog) + fog_color * fog

    @ti.func
    def rain_curtain(self, origin, direction, t_hit, clock, strength):
        """Noise-modulated translucent veil over the far lake.

        The ray crosses a low slab above the water between the near shore
        band and past the far bank; density is time-scrolled value noise
        with a transparent window around the moon direction.
        """
        veil = 0.0
        if strength > 0.001 and direction.z > 0.02:
            ta = (CURTAIN_Z0 - origin.z) / direction.z
            tb = (CURTAIN_Z1 - origin.z) / direction.z
            lo, hi = ti.min(ta, tb), ti.max(ta, tb)
            if ti.abs(direction.y) > 1e-5:
                ty0 = (0.0 - origin.y) / direction.y
                ty1 = (CURTAIN_HEIGHT - origin.y) / direction.y
                lo = ti.max(lo, ti.min(ty0, ty1))
                hi = ti.min(hi, ti.max(ty0, ty1))
            elif origin.y < 0.0 or origin.y > CURTAIN_HEIGHT:
                hi = lo - 1.0
            lo = ti.max(lo, 0.05)
            hi = ti.min(hi, t_hit)
            if hi > lo:
                mid = origin + direction * (lo + hi) * 0.5
                n1 = terrain_mottle(mid.x * 0.40 + clock * 0.9,
                                    mid.z * 0.25 - clock * 1.6)
                n2 = terrain_mottle(mid.x * 0.90 - clock * 0.5,
                                    mid.z * 0.55 - clock * 2.6)
                density = 0.5 + (n1 + 0.5 * n2) * 0.9
                density = ti.max(0.06, ti.min(1.0, density))
                # Transparent window around the moon disk and its halo.
                moon = ti.Vector([
                    float(MOON_DIRECTION[0]),
                    float(MOON_DIRECTION[1]),
                    float(MOON_DIRECTION[2]),
                ])
                alignment = direction.dot(moon)
                t = ti.min(1.0, ti.max(0.0, (alignment - 0.9955) / 0.0041))
                window = 1.0 - t * t * (3.0 - 2.0 * t)
                depth = strength * density * window * (hi - lo) * 0.016
                veil = 1.0 - ti.exp(-depth)
                veil = ti.min(veil, 0.6)
        return veil

    @ti.func
    def shade_sky(self, direction, fog_strength):
        """Environment dome plus the night sky layer and height fog."""
        color = self.environment.sample_map(self.environment.environment, direction, 2.0)
        color = night_sky(direction, color)
        # Thin height fog, strongest at the horizon in both directions.
        # The cap keeps the distant ridge silhouette readable through it.
        fog = ti.min(0.85, fog_strength) * ti.exp(-ti.abs(direction.y) * 5.0)
        fog_color = ti.Vector([0.020, 0.028, 0.042])
        return color * (1.0 - fog) + fog_color * fog

    @ti.func
    def trace_scene(self, origin, direction, clock, fog_strength, pixel_angle,
                    tile, with_water: ti.template(), proj_forward,
                    proj_right, proj_up, proj_scale, refl_gain):
        """Pinhole-equivalent shading of one ray; returns (color, t, obj)."""
        water_t = 1e6
        water_normal = ti.Vector([0.0, 1.0, 0.0])
        if ti.static(with_water):
            if direction.y < -1e-6:
                water_t, water_normal = self.water_hit(origin, direction, clock)
        terrain_t, terrain_grad = self.terrain_hit(origin, direction)
        best_t = ti.min(water_t, terrain_t)
        veg_t, veg_kind, veg_index, veg_a, veg_b = self.vegetation_hit_tiled(
            origin, direction, clock, best_t, tile)
        if veg_t < best_t:
            best_t = veg_t
        tree_t, tree_index, tree_normal = self.trees.trees_hit(
            origin, direction, self.vegetation_controls[None].y, best_t)
        if tree_t < best_t:
            best_t = tree_t
        obj = 0
        color = ti.Vector([0.0, 0.0, 0.0])
        if water_t == best_t and water_t < 1e5:
            obj = 1
            footprint = water_t * pixel_angle / ti.max(0.15, ti.abs(direction.y))
            hit_point = origin + direction * water_t
            fade = self.ripple_fade(water_t)
            grad = self.filtered_gradient(hit_point.x, hit_point.z, clock, fade,
                                          ti.min(0.5, footprint * 0.28867513))
            water_normal = ti.Vector([-grad.x, 1.0, -grad.y]).normalized()
            color = self.shade_water(direction, water_t, water_normal,
                                     footprint, origin, proj_forward,
                                     proj_right, proj_up, proj_scale,
                                     refl_gain)
        elif terrain_t == best_t and terrain_t < 1e5:
            obj = 2
            hit_point = origin + direction * terrain_t
            color = self.shade_terrain(direction, terrain_t, terrain_grad, hit_point)
        elif veg_t == best_t and veg_t < 1e5:
            if veg_kind == 1:
                obj = 3
                color = self.shade_grass_card(direction, veg_t, veg_index, veg_a, veg_b, origin)
            elif veg_kind == 2:
                obj = 3
                color = self.shade_blade(direction, veg_t, veg_index, veg_b, origin)
            else:
                obj = 4
                color = self.shade_leaf_card(direction, veg_t, veg_index, veg_a, veg_b, origin)
        elif tree_t == best_t and tree_t < 1e5:
            obj = 4
            color = self.shade_tree(direction, tree_t, tree_index, tree_normal, origin)
        if best_t >= 1e5:
            obj = 0
            color = self.shade_sky(direction, fog_strength)
        return color, best_t, obj

    @ti.kernel
    def temporal_accumulate(self):
        """Blend the jittered frame into the accumulation buffer and copy
        the running mean back over the frame, with the blend alpha chosen
        per object class so motion never ghosts into the history."""
        for x, y in self.post.hdr:
            alpha = ACCUM_ALPHA_STATIC
            obj = self.object_id[x, y]
            if obj == 1:
                alpha = ACCUM_ALPHA_WATER
            elif obj == 3 or obj == 4:
                alpha = ACCUM_ALPHA_VEGETATION
            hdr = self.post.hdr[x, y]
            if self.accum_frame_flag[None] == 0:
                self.accum_color[x, y] = hdr
            else:
                self.accum_color[x, y] = (self.accum_color[x, y] * (1.0 - alpha)
                                          + hdr * alpha)
            self.post.hdr[x, y] = self.accum_color[x, y]

    @ti.kernel
    def render(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
               right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
               clock: ti.f32, scale: ti.f32, fog_strength: ti.f32,
               pixel_angle: ti.f32, refl_gain: ti.f32,
               jx: ti.f32, jy: ti.f32):
        for x, y in self.image:
            sx = (2.0 * (x + jx) / self.width - 1.0) * self.width / self.height * scale
            sy = (2.0 * (y + jy) / self.height - 1.0) * scale
            direction = (forward + right * sx + up * sy).normalized()
            tile = ti.Vector([x // TILE_PIXELS, y // TILE_PIXELS])
            color, best_t, obj = self.trace_scene(eye, direction, clock,
                                                  fog_strength, pixel_angle,
                                                  tile, True, forward, right,
                                                  up, scale, refl_gain)
            self.object_id[x, y] = obj
            self.hit_distance[x, y] = best_t
            self.post.hdr[x, y] = color

    @ti.kernel
    def render_reflection(self, eye: ti.types.vector(3, ti.f32),
                          forward: ti.types.vector(3, ti.f32),
                          right: ti.types.vector(3, ti.f32),
                          up: ti.types.vector(3, ti.f32),
                          clock: ti.f32, scale: ti.f32,
                          fog_strength: ti.f32, pixel_angle: ti.f32,
                          jx: ti.f32, jy: ti.f32):
        """Quarter-resolution mirror render of the scene above the water plane.

        Up is negated so the image lands y-flipped, matching the lookup in
        shade_water, which projects the mirrored reflected direction with
        the real camera basis. The water surface itself is skipped; rays
        that miss geometry shade the sky along the mirrored direction so
        the moon disk and stars land in the lower half of the cache."""
        rwf = ti.cast(self.refl_width, ti.f32)
        rhf = ti.cast(self.refl_height, ti.f32)
        for x, y in self.refl_color:
            sx = (2.0 * (x + jx) / rwf - 1.0) * rwf / rhf * scale
            sy = (2.0 * (y + jy) / rhf - 1.0) * scale
            direction = (forward + right * sx - up * sy).normalized()
            tile = ti.Vector([(x * self.width // self.refl_width) // TILE_PIXELS,
                              (y * self.height // self.refl_height) // TILE_PIXELS])
            color, best_t, obj = self.trace_scene(eye, direction, clock,
                                                  fog_strength, pixel_angle,
                                                  tile, False, forward, right,
                                                  up, scale, 0.0)
            if best_t >= 1e5:
                mirrored = ti.Vector([direction.x, -direction.y, direction.z])
                color = self.shade_sky(mirrored, fog_strength)
            self.refl_color[x, y] = color

    @ti.kernel
    def render_dof(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
                   right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
                   clock: ti.f32, scale: ti.f32, fog_strength: ti.f32,
                   pixel_angle: ti.f32, aperture: ti.f32, focus_distance: ti.f32,
                   sample_count: ti.i32, refl_gain: ti.f32,
                   jx: ti.f32, jy: ti.f32):
        """Thin-lens accumulation: jitter the origin on the aperture disk
        and aim every ray at the primary ray's focal point, so the focal
        plane stays sharp and everything else blurs by its defocus.
        """
        golden = 2.399963229728653
        for x, y in self.image:
            tile = ti.Vector([x // TILE_PIXELS, y // TILE_PIXELS])
            sx = (2.0 * (x + jx) / self.width - 1.0) * self.width / self.height * scale
            sy = (2.0 * (y + jy) / self.height - 1.0) * scale
            direction = (forward + right * sx + up * sy).normalized()
            focal_t = focus_distance / ti.max(1e-3, direction.dot(forward))
            focal_point = eye + direction * focal_t
            # Sky samples take the center-ray sky color so the moon and
            # stars stay crisp; evaluating it once per pixel keeps the
            # shading branch out of the sample loop.
            sky_color = self.shade_sky(direction, fog_strength)
            # Spin the aperture pattern per pixel and frame so the few
            # sample replicas of a sharp edge decorrelate instead of
            # lining up into visible ghosts.
            spin = 2.0 * math.pi * hash01(x, y, self.post.frame[None], 41)
            total = ti.Vector([0.0, 0.0, 0.0])
            for s in range(sample_count):
                # Golden-angle spiral keeps the disk coverage even for any
                # sample count without a per-pixel hash.
                u = (ti.cast(s, ti.f32) + 0.5) / ti.cast(sample_count, ti.f32)
                angle = golden * ti.cast(s, ti.f32) + spin
                radius = aperture * ti.sqrt(u)
                origin = eye + right * (radius * ti.cos(angle)) \
                    + up * (radius * ti.sin(angle))
                ray = (focal_point - origin).normalized()
                color, best_t, obj = self.trace_scene(origin, ray, clock,
                                                      fog_strength, pixel_angle,
                                                      tile, True, forward,
                                                      right, up, scale,
                                                      refl_gain)
                if best_t >= 1e5:
                    color = sky_color
                total += color
                if s == 0:
                    # Classification and the curtain clamp follow the
                    # unjittered ray; splats stay sharp on top afterwards.
                    self.object_id[x, y] = obj
                    self.hit_distance[x, y] = best_t
            self.post.hdr[x, y] = total / ti.cast(sample_count, ti.f32)

    @ti.kernel
    def apply_rain_curtain(self, eye: ti.types.vector(3, ti.f32),
                           forward: ti.types.vector(3, ti.f32),
                           right: ti.types.vector(3, ti.f32),
                           up: ti.types.vector(3, ti.f32),
                           scale: ti.f32, clock: ti.f32, strength: ti.f32):
        """Post-shading veil pass; separate from render so its noise and
        branching do not inflate the primary kernel's register pressure."""
        tint = ti.Vector(list(CURTAIN_COLOR))
        for x, y in self.post.hdr:
            sx = (2.0 * (x + 0.5) / self.width - 1.0) * self.width / self.height * scale
            sy = (2.0 * (y + 0.5) / self.height - 1.0) * scale
            direction = (forward + right * sx + up * sy).normalized()
            veil = self.rain_curtain(eye, direction, self.hit_distance[x, y],
                                     clock, strength)
            if veil > 0.0:
                base = self.post.hdr[x, y]
                self.post.hdr[x, y] = base * (1.0 - veil) + tint * veil

    @ti.kernel
    def update_vegetation_sway(self, wind: ti.f32, clock: ti.f32,
                               grass_cards: ti.i32, blades: ti.i32,
                               leaf_cards: ti.i32):
        # One launch per frame instead of one per container; the formulas
        # stay identical so the sway is bit-for-bit unchanged.
        for i in range(grass_cards):
            meta = self.grass.cards.card_meta[i]
            amplitude = self.grass.cards.card_v[i].norm() * 0.10 * wind
            strength = ti.sin(clock * 1.9 + meta.z * 6.2831) * amplitude
            self.grass.cards.card_offset[i] = ti.Vector(
                [ti.cos(meta.z * 39.0), 0.0, ti.sin(meta.z * 39.0)]) * strength
        for i in range(blades):
            meta = self.grass.blade_meta[i]
            strength = ti.sin(clock * 1.9 + meta.x * 6.2831) * 0.05 * wind
            self.grass.blade_offset[i] = ti.Vector(
                [ti.cos(meta.x * 39.0), 0.0, ti.sin(meta.x * 39.0)]) * strength
        for i in range(leaf_cards):
            meta = self.trees.leaves.card_meta[i]
            amplitude = self.trees.leaves.card_v[i].norm() * 0.10 * wind
            strength = ti.sin(clock * 1.9 + meta.z * 6.2831) * amplitude
            self.trees.leaves.card_offset[i] = ti.Vector(
                [ti.cos(meta.z * 39.0), 0.0, ti.sin(meta.z * 39.0)]) * strength

    def draw(self, camera, clock=0.0, exposure=0.92, fog_strength=1.0, dt=1.0 / 60.0):
        if self._controls_dirty:
            self._sync_controls(1)
            self._controls_dirty = False
        else:
            self._sync_controls(0)
        # Block layout: water 0:4, camera 4:8, post 8:12, rain 12:16,
        # terrain 16:20, vegetation 20:24.
        state = self.controls_snapshot()
        rain_rate = state[2]
        wind = state[22]
        streak_gain, splash_gain = state[12], state[13]
        eye = np.asarray(camera.basis()[0], dtype=np.float64)
        self._simulate_rain(rain_rate, wind, splash_gain, eye, dt,
                            spawn=True, want_splash=True)
        self.update_vegetation_sway(wind, clock, self.grass.cards.count,
                                    self.grass.blade_count,
                                    self.trees.leaves.count)
        scale = math.tan(math.radians(camera.fov / 2))
        basis = camera.basis()
        pixel_angle = 2.0 * scale / self.height
        # The shadow bake keys on the terrain controls; a UI change of
        # either one re-bakes (cached, so repeat frames stay free).
        shadow_key = (state[16], state[17])
        if shadow_key != self._shadow_key:
            self.shadow_depth.from_numpy(bake_moon_shadow(
                self.trees, self.grass, shadow_key[0], shadow_key[1]))
            self._shadow_key = shadow_key
        self.post.advance_frame(clock)
        # Temporal accumulation state: a non-advancing clock (pause, seek,
        # repeated test draw) means the history no longer describes the
        # present frame; restart it instead of blending stale content. The
        # restarting frame samples at the pixel center, so the accumulated
        # sequence begins from the classic pinhole sample.
        accum_reset = (not self._accum_valid) or (clock <= self._last_accum_clock)
        if self.temporal_aa and not accum_reset:
            frame_idx = float(self.post.frame[None])
            jx = frame_idx * JITTER_GX % 1.0
            jy = frame_idx * JITTER_GY % 1.0
        else:
            jx, jy = 0.5, 0.5
        if self.reflection_gain > 0.0:
            self.render_reflection(basis[0], basis[1], basis[2], basis[3],
                                   clock, scale, fog_strength, pixel_angle,
                                   jx, jy)
        if state[4] > 0.5:
            self.render_dof(*basis, clock, scale, fog_strength, pixel_angle,
                            state[5], state[6], int(round(state[7])),
                            self.reflection_gain, jx, jy)
        else:
            self.render(*basis, clock, scale, fog_strength, pixel_angle,
                        self.reflection_gain, jx, jy)
        if self.temporal_aa:
            self.accum_frame_flag[None] = 0 if accum_reset else 1
            self._accum_valid = True
            self._last_accum_clock = clock
            self.temporal_accumulate()
        curtain = state[14]
        if curtain > 0.001:
            self.apply_rain_curtain(basis[0], basis[1], basis[2], basis[3],
                                    scale, clock, curtain)
        if streak_gain > 0.0 or splash_gain > 0.0:
            self.streaks.upload(state[15])
            self.streaks.splat(basis[0], basis[1], basis[2], basis[3],
                               scale, streak_gain, FOG_DISTANCE,
                               self.hit_distance)
            self.splash.upload()
            self.splash.splat(self.streaks.accum, self.streaks.weight,
                              self.width, self.height,
                              basis[0], basis[1], basis[2], basis[3],
                              scale, splash_gain)
            self.streaks.composite(self.post.hdr, 1.0)
        self.post.apply(exposure, state[8], state[9], state[10], state[11])

    def _simulate_rain(self, rain_rate, wind, splash_gain, eye, dt,
                       spawn=True, want_splash=True):
        """One rain step shared by draw() and warm_rain().

        Streaks are born in the air volume, advect under gravity and wind,
        and land on the first surface below them. Water landings inject
        velocity impulses into the ripple field; land landings near the
        camera trigger splash crowns. The ripple field steps every frame
        regardless of the weather so rings keep propagating.
        """
        if spawn and rain_rate > 0.0:
            self.streaks.spawn(rain_rate, wind, dt)
        landings = self.streaks.advance(dt, wind)
        if landings:
            points = np.asarray(landings, dtype=np.float64)
            heights = terrain_height_np(points[:, 0].astype(np.float32),
                                        points[:, 1].astype(np.float32))
            on_water = points[heights < 0.0]
            if len(on_water):
                ripple_strength = float(self.water_controls[None].y)
                strength = ripple_strength * BASE_INJECTION_STRENGTH
                self.ripples.inject_events(
                    on_water.astype(np.float32),
                    self.ripples.event_strengths(len(on_water), strength))
            if want_splash:
                for x, z in points[heights >= 0.0]:
                    self.splash.maybe_spawn(float(x), float(z), splash_gain,
                                            eye)
        self.splash.advance(dt)
        self.ripples.advance(dt)

    def warm_rain(self, seconds, dt=1.0 / 60.0):
        """Pre-roll the ripple field so rain looks developed on frame one.

        The streak table advects through the same warmup so airborne rain
        exists from the first rendered frame; splash crowns stay off so the
        pool starts empty.
        """
        _, _, rain_rate, _ = tuple(self.water_controls[None])
        _, _, wind, _ = tuple(self.vegetation_controls[None])
        for _ in range(int(round(seconds / dt))):
            self._simulate_rain(rain_rate, wind, 0.0, PRODUCTION_EYE, dt,
                                spawn=True, want_splash=False)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Taichi images are x-first, with the origin at the bottom left.
        pixels = self.image.to_numpy().transpose(1, 0, 2)[::-1]
        if not np.isfinite(pixels).all():
            raise RuntimeError("Renderer produced a non-finite image")
        Image.fromarray(np.round(np.clip(pixels, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
        return path.resolve()
