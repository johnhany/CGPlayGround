"""Dive: the shallow-water shore seen from inside the water.

The camera of demo 06 drops below the swell: rays leaving the eye travel in
water, so they attenuate toward blue-green with distance (Beer-Lambert plus a
bulk inscatter), the sandy bed lights up with sun caustics refracted by the
waves, and looking up the surface opens the Snell window — sky and beach seen
through refraction inside the critical cone, total internal reflection
outside it. Above-water views keep the demo 06 path unchanged.

Run: uv run python water/demo_07.py
Export: uv run python water/demo_07.py --headless --output output/dive.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .demo_03 import FAR_PLANE, FOG_DISTANCE
    from .demo_06 import (BISECT_STEPS, MARCH_GROWTH, MARCH_MAX_STEPS,
                          MARCH_MIN_STEP, MAX_BED_SLOPE, SEA_DX, SEA_HALF,
                          SEA_N, SHORE_PRESETS_SEA, SIM_DT, ShoreCamera,
                          ShoreRenderer, sea_parameters, shore_terrain,
                          simulation_budget, water_pick)
    from .finish import water_reflection
else:
    from demo_03 import FAR_PLANE, FOG_DISTANCE
    from demo_06 import (BISECT_STEPS, MARCH_GROWTH, MARCH_MAX_STEPS,
                         MARCH_MIN_STEP, MAX_BED_SLOPE, SEA_DX, SEA_HALF,
                         SEA_N, SHORE_PRESETS_SEA, SIM_DT, ShoreCamera,
                         ShoreRenderer, sea_parameters, shore_terrain,
                         simulation_budget, water_pick)
    from finish import water_reflection

__all__ = ["DiveRenderer", "DiveCamera", "bed_height_py", "apply_dive_preset",
           "DIVE_PRESETS", "dive_escape_cpu", "main", "parse_args",
           "sea_parameters", "simulation_budget", "water_pick",
           "SEA_HALF", "SEA_N", "SEA_DX", "SIM_DT", "IOR"]

# Water refractive index. The critical angle is asin(1 / 1.333) ~ 48.6 deg:
# rays from below steeper than that exit into air, shallower ones reflect.
IOR = 1.333
# Local caustic atlas: 0.125 m cells over a 64 m underwater region.
CAUSTIC_N = 512
CAUSTIC_HALF = 32.0
CAUSTIC_CENTER_X = -22.0
CAUSTIC_DX = 2.0 * CAUSTIC_HALF / CAUSTIC_N


def bed_height_py(x, z):
    """Python mirror of demo_06.shore_terrain for camera ground clearance."""
    bounded_x = x
    if x > SEA_HALF:
        bounded_x = SEA_HALF + 8.0 * math.tanh((x - SEA_HALF) / 8.0)
    elif x < -SEA_HALF:
        bounded_x = -SEA_HALF + 8.0 * math.tanh((x + SEA_HALF) / 8.0)
    height = (0.045 * (bounded_x - 4.0)
              + 0.16 * math.sin(0.031 * z + 0.9)
              + 0.10 * math.sin(0.083 * z + 3.94)
              + 0.06 * math.sin(0.017 * z + 4.32))
    for cx, cz, top, sigma in ((-18.0, -10.0, 3.0, 1.6), (-6.0, 8.0, 2.2, 1.2),
                               (-30.0, 18.0, 2.3, 2.0), (20.0, -20.0, 0.9, 1.4)):
        dx, dz = x - cx, z - cz
        crag = 1.0 + 0.06 * math.sin(1.8 * dx) * math.sin(1.3 * dz)
        height += top * math.exp(-(dx * dx + dz * dz) / sigma ** 2) * crag
    return height


@ti.func
def dive_escape(direction, normal_up):
    """Water-to-air interface at a surface point seen from below.

    Returns (tir, transmitted, reflected): tir is 1.0 past the critical
    angle, transmitted the refracted air direction (undefined under TIR),
    reflected the mirror direction about the surface normal. Branchless
    shaping with pre-declared locals: Taichi funcs cannot return out of a
    dynamic if, and plain-Python helpers miscompile inside kernels.
    """
    cos_in = direction.dot(normal_up)
    sin2 = ti.max(0.0, 1.0 - cos_in * cos_in)
    refracted2 = IOR * IOR * sin2
    tir = 0.0
    if refracted2 >= 1.0:
        tir = 1.0
    cos_t = ti.sqrt(ti.max(1e-6, 1.0 - refracted2))
    # Water to air: the tangential component scales by IOR, so the normal
    # offset is (cos_t - IOR * cos_in) — t dot n must equal cos_t.
    transmitted = (IOR * direction + (cos_t - IOR * cos_in) * normal_up).normalized()
    reflected = (direction - 2.0 * direction.dot(normal_up) * normal_up).normalized()
    return tir, transmitted, reflected


def dive_escape_cpu(direction, normal_up):
    """Python mirror of dive_escape for tests and reference checks."""
    direction = np.asarray(direction, dtype=float)
    normal_up = np.asarray(normal_up, dtype=float)
    cos_in = float(np.dot(direction, normal_up))
    sin2 = max(0.0, 1.0 - cos_in * cos_in)
    refracted2 = IOR * IOR * sin2
    tir = refracted2 >= 1.0
    cos_t = math.sqrt(max(1e-6, 1.0 - refracted2))
    transmitted = IOR * direction + (cos_t - IOR * cos_in) * normal_up
    transmitted /= np.linalg.norm(transmitted)
    reflected = direction - 2.0 * np.dot(direction, normal_up) * normal_up
    reflected /= np.linalg.norm(reflected)
    return tir, transmitted, reflected


DIVE_PRESETS = {
    "overview": ((2.0, 0.0, 0.0), (0.50, 0.35, 15.0), 45.0),
    "waterline": ((2.0, 0.0, 0.0), (0.50, 0.12, 9.0), 45.0),
    "top": ((2.0, 0.0, 0.0), (0.50, 1.42, 26.0), 45.0),
    "dive": ((-20.0, -0.75, -8.0), (-1.10, -0.10, 4.0), 62.0),
    "window": ((-28.0, 0.0, -6.0), (2.70, -1.48, 1.6), 110.0),
    "bed": ((-24.0, -1.25, -7.0), (0.9, 0.32, 1.4), 62.0),
}


def apply_dive_preset(camera, name):
    camera.target = np.array(DIVE_PRESETS[name][0])
    camera.yaw, camera.pitch, camera.distance = DIVE_PRESETS[name][1]
    camera.fov = DIVE_PRESETS[name][2]
    camera.lift_above_bed()


class DiveCamera(ShoreCamera):
    """Water/air orbit camera with continuous motion over the full sea."""

    def orbit(self, dx, dy):
        self.yaw -= dx * 3.5
        self.pitch = float(np.clip(self.pitch - dy * 3.0, -1.48, 1.48))
        self.lift_above_bed()

    def pan(self, dx, dy):
        _, _, right, _ = self.basis()
        horizontal_forward = np.array([-right[2], 0.0, right[0]])
        self.target -= (right * dx + horizontal_forward * dy) * self.distance
        self.target[0] = float(np.clip(self.target[0], -58.0, 58.0))
        self.target[2] = float(np.clip(self.target[2], -58.0, 58.0))
        self.target[1] = float(np.clip(self.target[1], -3.5, 2.0))
        self.lift_above_bed()

    def zoom(self, amount):
        self.distance = float(np.clip(self.distance * math.exp(amount), 0.7, 60.0))
        self.lift_above_bed()

    def lift_above_bed(self):
        eye = self.basis()[0]
        deficit = bed_height_py(float(eye[0]), float(eye[2])) + 0.3 - float(eye[1])
        if deficit > 0.0:
            self.target[1] = float(self.target[1] + deficit)


@ti.func
def dielectric_fresnel(cos_in, eta):
    cosine = ti.max(0.0, ti.min(1.0, cos_in))
    sin_t2 = eta * eta * (1.0 - cosine * cosine)
    reflectance = 1.0
    if sin_t2 < 1.0:
        transmitted = ti.sqrt(ti.max(0.0, 1.0 - sin_t2))
        rs = (eta * cosine - transmitted) / ti.max(1e-8, eta * cosine + transmitted)
        rp = (cosine - eta * transmitted) / ti.max(1e-8, cosine + eta * transmitted)
        reflectance = 0.5 * (rs * rs + rp * rp)
    return reflectance


@ti.func
def pack_normal(normal):
    n = normal / ti.max(1e-8, ti.abs(normal).sum())
    uv = ti.Vector([n.x, n.y])
    if n.z < 0.0:
        uv = ti.Vector([(1.0 - ti.abs(n.y)) * ti.select(n.x >= 0, 1., -1.),
                        (1.0 - ti.abs(n.x)) * ti.select(n.y >= 0, 1., -1.)])
    quantized = ti.cast(ti.max(0.0, ti.min(65535.0, (uv * 0.5 + 0.5) * 65535.0 + 0.5)), ti.u32)
    return quantized.x | (quantized.y << 16)


@ti.func
def unpack_normal(packed):
    x = ti.cast(packed & 65535, ti.f32) / 65535.0 * 2 - 1
    y = ti.cast(packed >> 16, ti.f32) / 65535.0 * 2 - 1
    z = 1.0 - ti.abs(x) - ti.abs(y)
    nx, ny = x, y
    if z < 0.0:
        nx = (1.0 - ti.abs(y)) * ti.select(x >= 0, 1., -1.)
        ny = (1.0 - ti.abs(x)) * ti.select(y >= 0, 1., -1.)
    return ti.Vector([nx, ny, z]).normalized()


@ti.func
def pack_color(color):
    c = ti.cast(ti.max(0.0, ti.min(255.0, color * 255.0 + 0.5)), ti.u32)
    return c.x | (c.y << 8) | (c.z << 16)


@ti.func
def unpack_color(packed):
    return ti.Vector([ti.cast(packed & 255, ti.f32), ti.cast((packed >> 8) & 255, ti.f32),
                      ti.cast((packed >> 16) & 255, ti.f32)]) / 255.0


@ti.data_oriented
class DiveRenderer(ShoreRenderer):
    """Demo 06 shore renderer with an underwater medium inside the sea."""

    def __init__(self, width, height, samples=4):
        super().__init__(width, height, samples)
        # x: caustic strength on the submerged bed, y: sun-shaft inscatter,
        # z/w spare for panel extensions.
        self.dive_controls = ti.Vector.field(4, ti.f32, shape=())
        self.dive_controls[None] = [1.0, 0.6, 0.0, 0.0]
        self.dive_caustic_raw = ti.field(ti.f32, shape=(CAUSTIC_N, CAUSTIC_N))
        self.dive_caustic_map = ti.field(ti.f32, shape=(CAUSTIC_N, CAUSTIC_N))
        self.dive_caustic_map.from_numpy(np.ones((CAUSTIC_N, CAUSTIC_N),
                                                 dtype=np.float32))
        # Uniform per-frame medium flag: every ray shares the eye, so "is the
        # eye underwater" decides the whole frame and draw() can pick one of
        # two separately compiled render kernels instead of inlining both
        # shading trees into one giant kernel (which takes >10 min to compile).
        self.medium_flag = ti.field(ti.f32, shape=())
        self.medium_flag[None] = 0.0
        self.sample_capacity = samples
        self._dive_mode = ti.field(ti.i32, shape=(width, height, samples))
        self._dive_surface_t = ti.field(ti.f32, shape=(width, height, samples))
        self._dive_geo_normal = ti.field(ti.u32, shape=(width, height, samples))
        self._dive_shading_normal = ti.field(ti.u32, shape=(width, height, samples))
        self._dive_fresnel = ti.field(ti.f32, shape=(width, height, samples))
        self._dive_hit_t = ti.Vector.field(2, ti.f32, shape=(width, height, samples))
        self._dive_hit_n = ti.Vector.field(2, ti.u32, shape=(width, height, samples))
        self._dive_hit_base = ti.Vector.field(2, ti.u32, shape=(width, height, samples))
        self._dive_hit_m = ti.Vector.field(2, ti.i32, shape=(width, height, samples))
        self.surface_slopes = ti.Vector.field(2, ti.f32, shape=(SEA_N, SEA_N))
        self.surface_range = ti.Vector.field(2, ti.f32, shape=())
        self.medium_depth = ti.field(ti.f32, shape=())
        self.view_clock = ti.field(ti.f32, shape=())
        self.view_clarity = ti.field(ti.f32, shape=())
        self.view_clock[None], self.view_clarity[None] = 0.0, 1.4
        self.volume_size = (max(1, (width + 3) // 4), max(1, (height + 3) // 4))
        self.volume = ti.Vector.field(3, ti.f32, shape=self.volume_size)
        self.volume_depth = ti.field(ti.f32, shape=self.volume_size)
        self.transition_air = ti.Vector.field(3, ti.f32, shape=(width, height))
        rng = np.random.default_rng(707)
        angle = rng.uniform(-math.pi, math.pi, 24)
        wavelength = rng.uniform(0.45, 1.7, 24)
        k = 2 * math.pi / wavelength
        modes = np.column_stack((k * np.cos(angle), k * np.sin(angle),
                                 0.30 / (k * math.sqrt(12)), rng.uniform(0, 2 * math.pi, 24)))
        self.detail_modes.from_numpy(modes.astype(np.float32))
        self.detail_omega.from_numpy(np.sqrt(9.81 * k).astype(np.float32))
        # Small submerged stones supply foreground scale and beam occluders.
        for x, z, extent in [(-24., -8., (.65,.23,.48)), (-22.2,-5.8,(.45,.16,.32)),
                             (-25.3,-6.2,(.32,.13,.28))]:
            i = self.boxes
            self.box_center[i] = [x, bed_height_py(x,z) + extent[1] * .7, z]
            self.box_extent[i], self.box_bevel[i] = extent, min(extent) * .8
            self.box_color[i], self.box_material[i] = [.28,.33,.30], 4
            self.boxes += 1

    @ti.kernel
    def _dive_medium(self, eye: ti.types.vector(3, ti.f32), clock: ti.f32):
        surface = self.wave_height(eye.x, eye.z, clock)
        submerged = 0.0
        if eye.y < surface:
            submerged = 1.0
        self.medium_flag[None] = submerged
        self.medium_depth[None] = surface - eye.y

    @ti.kernel
    def render_above(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
                     right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
                     clock: ti.f32, clarity: ti.f32, lighting: ti.f32,
                     exposure: ti.f32, scale: ti.f32, samples: ti.i32):
        """Same body as the inherited render kernel, but bound to the demo 06
        radiance only — the underwater tree lives in render_dive/dive_shade."""
        for x, y in self.image:
            color = ti.Vector([0.0, 0.0, 0.0])
            for sample in range(samples):
                offset = ti.Vector([0.5, 0.5])
                if samples == 4:
                    offset = ti.Vector([0.25 + 0.5 * (sample % 2), 0.25 + 0.5 * (sample // 2)])
                sx = (2.0 * (x + offset.x) / self.width - 1.0) * self.width / self.height * scale
                sy = (2.0 * (y + offset.y) / self.height - 1.0) * scale
                direction = (forward + right * sx + up * sy).normalized()
                color += ShoreRenderer.radiance(self, eye, direction, clock, clarity,
                                               lighting, 2.0 * scale / self.height)
            color /= samples
            self.post.hdr[x, y] = color

    @ti.func
    def camera_ray(self, x, y, sample, samples, forward, right, up, scale):
        offset = ti.Vector([0.5, 0.5])
        if samples == 4:
            offset = ti.Vector([0.25 + 0.5 * (sample % 2), 0.25 + 0.5 * (sample // 2)])
        sx = (2.0 * (x + offset.x) / self.width - 1.0) * self.width / self.height * scale
        sy = (2.0 * (y + offset.y) / self.height - 1.0) * scale
        return (forward + right * sx + up * sy).normalized()

    @ti.func
    def underwater_reflection(self, direction, normal, sample):
        count = ti.cast(self.water_controls[None].w, ti.i32)
        ray, weight = water_reflection(-direction, -normal,
                                      self.water_controls[None].x, sample % count, count)
        # Preserve a valid water-side ray at near-tangent microfacets.
        if weight <= 0.0:
            ray = (direction - 2 * direction.dot(normal) * normal).normalized()
        return ray

    @ti.kernel
    def render_dive(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
                    right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
                    clock: ti.f32, clarity: ti.f32, lighting: ti.f32,
                    exposure: ti.f32, scale: ti.f32, samples: ti.i32):
        for x, y in self.image:
            for sample in range(samples):
                direction = self.camera_ray(x,y,sample,samples,forward,right,up,scale)
                t0, n0, c0, m0 = self.geometry(eye, direction)
                exit_t, geo = self.water_exit(eye, direction, clock, ti.min(FAR_PLANE,t0), 1)
                mode, reflection = 0, 1.0
                n = geo
                t1, m1 = 1e6, -2
                n1, c1 = ti.Vector([0.,1.,0.]), ti.Vector([0.,0.,0.])
                if exit_t < t0:
                    mode = 1
                    p = eye + direction * exit_t
                    footprint = exit_t * (2.0 * scale / self.height) / ti.max(.15,ti.abs(direction.dot(geo)))
                    shading = self.shading_normal(p,clock,footprint,geo)
                    if shading.dot(direction) > 0:
                        n = shading
                    tir, transmitted, reflected = dive_escape(direction,n)
                    reflected = self.underwater_reflection(direction,n,sample)
                    reflection = dielectric_fresnel(direction.dot(n),IOR)
                    offset = self.water_ray_offset(p,geo)
                    t0,n0,c0,m0 = self.geometry(p-geo*offset,reflected)
                    if tir < .5:
                        t1,n1,c1,m1 = self.geometry(p+geo*offset,transmitted)
                self._dive_mode[x,y,sample] = mode
                self._dive_surface_t[x,y,sample] = exit_t
                self._dive_geo_normal[x,y,sample] = pack_normal(geo)
                self._dive_shading_normal[x,y,sample] = pack_normal(n)
                self._dive_fresnel[x,y,sample] = reflection
                self._dive_hit_t[x,y,sample] = ti.Vector([t0,t1])
                self._dive_hit_m[x,y,sample] = ti.Vector([m0,m1])
                self._dive_hit_n[x,y,sample] = ti.Vector([pack_normal(n0),pack_normal(n1)])
                self._dive_hit_base[x,y,sample] = ti.Vector([pack_color(c0),pack_color(c1)])

    @ti.kernel
    def dive_shade(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
                   right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
                   scale: ti.f32, clock: ti.f32, clarity: ti.f32, lighting: ti.f32,
                   samples: ti.i32):
        # A single shading tree handles both events. Packed geometry keeps
        # complete reflection/transmission without duplicating the PBR inline.
        for x,y in self.image:
            color = ti.Vector([0.,0.,0.])
            for sample in range(samples):
                direction = self.camera_ray(x,y,sample,samples,forward,right,up,scale)
                mode = self._dive_mode[x,y,sample]
                exit_t = self._dive_surface_t[x,y,sample]
                f = self._dive_fresnel[x,y,sample]
                geo = unpack_normal(self._dive_geo_normal[x,y,sample])
                normal = unpack_normal(self._dive_shading_normal[x,y,sample])
                tir, transmitted, reflected = dive_escape(direction,normal)
                reflected = self.underwater_reflection(direction,normal,sample)
                p = eye+direction*exit_t
                offset = self.water_ray_offset(p,geo)
                mixed = ti.Vector([0.,0.,0.])
                for event in range(2):
                    origin, ray = eye, direction
                    if mode == 1:
                        origin, ray = p-geo*offset, reflected
                        if event == 1:
                            origin, ray = p+geo*offset, transmitted
                    t = self._dive_hit_t[x,y,sample][event]
                    material = self._dive_hit_m[x,y,sample][event]
                    n = unpack_normal(self._dive_hit_n[x,y,sample][event])
                    base = unpack_color(self._dive_hit_base[x,y,sample][event])
                    shaded = ti.Vector([0.,0.,0.])
                    weight = 1.0
                    if mode == 1:
                        weight = f
                        if event == 1:
                            weight = 1.0-f
                    elif event == 1:
                        weight = 0.0
                    if weight > .000001:
                        shaded,_ = self.shade_surface(origin,ray,t,n,base,material,clock,lighting,0.,1)
                    if event == 0:
                        # A miss in water is deep-water radiance, never sky.
                        if material < 0:
                            shaded = ti.Vector([0.,0.,0.])
                        shaded = self.water_body(shaded,t,clarity,0.,ti.Vector([0.,1.,0.]),lighting)
                    elif mode == 1:
                        if material >= 0:
                            shaded = self.air_fog(shaded,t,ray,lighting)
                    mixed += shaded*weight
                if mode == 1:
                    footprint = exit_t * (2.0 * scale / self.height) / ti.max(.15,ti.abs(direction.dot(geo)))
                    mixed = self.composite_foam(mixed,p,clock,lighting,footprint)
                    mixed = self.water_body(mixed,exit_t,clarity,0.,ti.Vector([0.,1.,0.]),lighting)
                color += mixed
            self.post.hdr[x,y] = color/samples

    @ti.func
    def detail_gradient(self,x,z,clock,footprint):
        gradient = ti.Vector([0.,0.])
        depth = self.sea.sample_depth(x,z)
        for band in range(24):
            mode = self.detail_modes[band]
            k = ti.sqrt(mode.x*mode.x+mode.y*mode.y)
            wavelength = 2*math.pi/k
            fade = 1-self.smooth(wavelength*.12,wavelength*.5,footprint)
            phase = mode.x*x+mode.y*z-self.detail_omega[band]*clock+mode.w
            gradient += mode.z*ti.cos(phase)*ti.Vector([mode.x,mode.y])*fade
        return gradient*self.water_controls[None].y*self.smooth(.05,.5,depth)*ti.min(1.,clock*.5)

    @ti.func
    def bed_ripple_strength(self):
        return 1.0

    @ti.func
    def floor_material(self,p,clock,amplitude):
        base,normal = ShoreRenderer.floor_material(self,p,clock,amplitude)
        depth = self.sea.sample_depth(p.x,p.z)
        footprint = self.ground_footprint(p)
        fade = self.smooth(.05,.3,depth)*(1-self.smooth(.04,.18,footprint))
        noise = self.sand_sample(p.x*.24,p.z*.24,footprint*.24)
        phase = 14.0*(p.x*.94+p.z*.34)+noise.x*2.0
        reef = ti.min(1.,ti.max(0.,(shore_terrain(p.x,p.z)-.045*(p.x-4.))*.9))
        fade *= (1-reef)*self.bed_ripple_strength()
        base *= 1+fade*.12*ti.sin(phase)
        normal = (normal+ti.Vector([-.025*.94,0.,-.025*.34])*ti.cos(phase)*fade).normalized()
        return base,normal

    @ti.func
    def sun_in_water(self,lighting):
        sun = self.sun_direction(lighting)
        eta = 1.0/IOR
        y = ti.sqrt(ti.max(0.,1-eta*eta*(1-sun.y*sun.y)))
        return ti.Vector([eta*sun.x,y,eta*sun.z])

    @ti.func
    def light_surface(self,p,clock,lighting):
        sun = self.sun_in_water(lighting)
        height = self.wave_height(p.x,p.z,clock)
        distance = ti.max(0.,(height-p.y)/sun.y)
        # Fixed-point refinement follows the varying wave height, instead of
        # treating the water surface as a horizontal slab.
        for _ in range(3):
            q = p+sun*distance
            height = self.wave_height(q.x,q.z,clock)
            distance = ti.max(0.,(height-p.y)/sun.y)
        return p+sun*distance,distance

    @ti.func
    def surface_sun_visibility(self,p,normal,material,lighting,detail):
        shade = 1.0
        depth = self.wave_height(p.x,p.z,self.view_clock[None])-p.y
        if depth > .02:
            q,distance = self.light_surface(p,self.view_clock[None],lighting)
            sun_water = self.sun_in_water(lighting)
            underwater = self.occlusion_distance(p+sun_water*.002,sun_water,ti.max(.001,distance-.004))
            air_sun = self.sun_direction(lighting)
            air = self.occlusion_distance(q+air_sun*.003,air_sun,30.)
            shade = ti.cast(underwater >= distance-.005 and air >= 29.99,ti.f32)
            gate = self.dive_controls[None].x*self.smooth(.02,.30,depth)
            shade *= ti.max(0.,1-gate+gate*self.dive_caustic_density(p))
        else:
            shade = self.visibility(p,normal,self.sun_direction(lighting),detail)
        return shade

    @ti.func
    def surface_sun_color(self,p,material,lighting,color):
        depth = ti.max(0.,self.wave_height(p.x,p.z,self.view_clock[None])-p.y)
        distance = depth/self.sun_in_water(lighting).y
        coeff = ti.Vector([.34,.12,.065])/self.view_clarity[None]
        return color*ti.exp(-coeff*distance)

    @ti.kernel
    def _clear_dive_caustics(self):
        for i, j in self.dive_caustic_raw:
            self.dive_caustic_raw[i, j] = 0.0

    @ti.kernel
    def _trace_dive_caustics(self, clock: ti.f32, lighting: ti.f32):
        """Deposit refracted sun flux onto the bed, demo 02 style.

        Each wet surface cell refracts the sun through its cubic-interpolated
        slope and splats the transmitted flux where the ray lands on the bed.
        A flat surface deposits ~uniformly (mean ~ Fresnel transmission), a
        wavy one focuses into filaments brighter than one.
        """
        f0 = ((IOR - 1.0) / (IOR + 1.0)) ** 2
        for i, j in self.dive_caustic_raw:
            x = CAUSTIC_CENTER_X - CAUSTIC_HALF + (i + 0.5) * CAUSTIC_DX
            z = -CAUSTIC_HALF + (j + 0.5) * CAUSTIC_DX
            surface = self.sea.sample(x, z, clock)
            depth = surface.x - shore_terrain(x, z)
            if depth > 0.05:
                detail = self.detail_gradient(x,z,clock,CAUSTIC_DX*.25)
                normal = ti.Vector([-surface.y-detail.x, 1.0, -surface.z-detail.y]).normalized()
                sun = self.sun_direction(lighting)
                cosine = sun.dot(normal)
                if cosine > 0.0:
                    eta = 1.0 / IOR
                    root = ti.sqrt(ti.max(0.0, 1.0 - eta * eta * (1.0 - cosine * cosine)))
                    ray = (-eta * sun + (eta * cosine - root) * normal).normalized()
                    if -ray.y > 1e-4:
                        origin = ti.Vector([x, surface.x, z])
                        hit = self.terrain_hit(origin+ray*.00001,ray,12.)
                        target = origin+ray*hit
                        if hit < 12. and ti.abs(target.x-CAUSTIC_CENTER_X) < CAUSTIC_HALF-CAUSTIC_DX and ti.abs(target.z) < CAUSTIC_HALF-CAUSTIC_DX:
                            transmitted = 1.0 - dielectric_fresnel(cosine, 1.0/IOR)
                            energy = transmitted * cosine / ti.max(0.02, normal.y * sun.y)
                            u = (target.x - CAUSTIC_CENTER_X + CAUSTIC_HALF) / CAUSTIC_DX - 0.5
                            v = (target.z + CAUSTIC_HALF) / CAUSTIC_DX - 0.5
                            ix, iy = ti.cast(ti.floor(u), ti.i32), ti.cast(ti.floor(v), ti.i32)
                            fx, fy = u - ti.floor(u), v - ti.floor(v)
                            for a, b in ti.static(ti.ndrange(2, 2)):
                                if 0 <= ix + a < CAUSTIC_N and 0 <= iy + b < CAUSTIC_N:
                                    weight = (fx if a else 1.0 - fx) * (fy if b else 1.0 - fy)
                                    ti.atomic_add(self.dive_caustic_raw[ix + a, iy + b],
                                                  energy * weight)

    @ti.kernel
    def _filter_dive_caustics(self):
        for i, j in self.dive_caustic_map:
            value = 0.0
            for a, b in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                weight = (2.0 if a == 0 else 1.0) * (2.0 if b == 0 else 1.0) / 16.0
                value += self.dive_caustic_raw[ti.min(CAUSTIC_N - 1, ti.max(0, i + a)),
                                               ti.min(CAUSTIC_N - 1, ti.max(0, j + b))] * weight
            self.dive_caustic_map[i, j] = ti.min(6.0, value)

    def trace_caustics(self, clock, lighting):
        self._clear_dive_caustics()
        self._trace_dive_caustics(clock, lighting)
        self._filter_dive_caustics()

    @ti.func
    def dive_caustic_density(self,p):
        value = 1.0
        if ti.abs(p.x-CAUSTIC_CENTER_X) < CAUSTIC_HALF-CAUSTIC_DX and ti.abs(p.z) < CAUSTIC_HALF-CAUSTIC_DX:
            u = ti.max(0.,ti.min(CAUSTIC_N-1.,(p.x-CAUSTIC_CENTER_X+CAUSTIC_HALF)/CAUSTIC_DX-.5))
            v = ti.max(0.,ti.min(CAUSTIC_N-1.,(p.z+CAUSTIC_HALF)/CAUSTIC_DX-.5))
            ix,iy = ti.cast(u,ti.i32),ti.cast(v,ti.i32)
            fx,fy = u-ix,v-iy
            value = 0.
            for a,b in ti.static(ti.ndrange(2,2)):
                weight = (fx if a else 1-fx)*(fy if b else 1-fy)
                value += self.dive_caustic_map[ti.min(CAUSTIC_N-1,ix+a),ti.min(CAUSTIC_N-1,iy+b)]*weight
        return value

    @ti.func
    def cell_slopes(self,i,j):
        bound = ti.Vector([0.,0.])
        for row in ti.static(range(4)):
            px = ti.Vector([0.,0.,0.,0.])
            pz = ti.Vector([0.,0.,0.,0.])
            for k in ti.static(range(4)):
                px[k] = self.sea.eta_view[ti.max(0,ti.min(SEA_N-1,i+k-1)),ti.max(0,ti.min(SEA_N-1,j+row-1))]
                pz[k] = self.sea.eta_view[ti.max(0,ti.min(SEA_N-1,i+row-1)),ti.max(0,ti.min(SEA_N-1,j+k-1))]
            for axis in ti.static(range(2)):
                p = px
                if ti.static(axis == 1):
                    p = pz
                m0,m1 = .5*(p.z-p.x),.5*(p.w-p.y)
                derivative = ti.max(ti.abs(m0),ti.max(ti.abs(m1),ti.abs(3*(p.z-p.y)-m0-m1)))
                bound[axis] = ti.max(bound[axis],derivative*1.25/SEA_DX)
        return bound

    @ti.kernel
    def prepare_water_bounds(self):
        self.surface_range[None] = ti.Vector([1e6,-1e6])
        for i,j in self.surface_slopes:
            self.surface_slopes[i,j] = self.cell_slopes(i,j)
            ti.atomic_min(self.surface_range[None].x,self.sea.eta_view[i,j])
            ti.atomic_max(self.surface_range[None].y,self.sea.eta_view[i,j])

    @ti.func
    def ray_cell(self,p,direction):
        # Bias exactly-on-grid points toward the next cell, avoiding tiny
        # stalled steps at a boundary without jumping a finite surface gap.
        q = ti.Vector([p.x,p.z])+ti.Vector([direction.x,direction.z])*0.00002
        uv = (q+SEA_HALF)/SEA_DX
        cell = ti.cast(ti.floor(uv),ti.i32)
        distance = 1e6
        for axis in ti.static(range(2)):
            velocity = direction.x
            coordinate = p.x
            if ti.static(axis == 1):
                velocity,coordinate = direction.z,p.z
            if ti.abs(velocity) > 1e-8:
                edge = -SEA_HALF+(cell[axis]+ti.select(velocity>0,1,0))*SEA_DX
                distance = ti.min(distance,ti.max(.00002,(edge-coordinate)/velocity))
        return cell,distance

    @ti.func
    def water_exit(self,origin,direction,clock,limit=FAR_PLANE,cached=0):
        hit = 1e6
        normal = ti.Vector([0.,1.,0.])
        envelope = self.wave_bounds()
        if cached == 1:
            extrema = self.surface_range[None]
            lo = ti.min(extrema.x,-1.02*self.sea.amplitude[None])-.02
            hi = ti.max(extrema.y,1.02*self.sea.amplitude[None])+.02
            padding = (hi-lo)*.28125
            envelope = ti.Vector([lo-padding,hi+padding])
        near,far = .000001,ti.min(FAR_PLANE,limit)
        if ti.abs(direction.y) > 1e-8:
            a,b = (envelope.x-origin.y)/direction.y,(envelope.y-origin.y)/direction.y
            near,far = ti.max(near,ti.min(a,b)),ti.min(far,ti.max(a,b))
        elif origin.y < envelope.x or origin.y > envelope.y:
            far = -1.
        if near <= far:
            current,count = near,0
            point = origin+direction*current
            height = self.wave_height(point.x,point.z,clock)
            while hit == 1e6 and current <= far and count < 768:
                error = height-point.y
                precision = 0.000002+0.0000002*current
                if error <= precision:
                    surface = self.sea.sample(point.x,point.z,clock)
                    if surface.w > .00001:
                        hit = current
                    else:
                        current = far+1.
                else:
                    cell,edge = self.ray_cell(point,direction)
                    # Analytic exterior derivatives are bounded by the two
                    # incident modes (including alongshore amplitude drift).
                    omega = 2*math.pi/self.sea.period[None]
                    k = omega/ti.sqrt(9.81*3.)
                    amplitude = self.sea.amplitude[None]
                    slopes = ti.Vector([amplitude*k*(.864+.42*1.71),
                                         amplitude*k*(.864*.105+.42*1.71*.141)+amplitude*.0128])
                    if 0 <= cell.x < SEA_N-1 and 0 <= cell.y < SEA_N-1:
                        local = self.cell_slopes(cell.x,cell.y)
                        if cached == 1:
                            local = self.surface_slopes[cell.x,cell.y]
                        if ti.abs(point.x)>SEA_HALF-8. or ti.abs(point.z)>SEA_HALF-8.:
                            local = ti.max(local,slopes)+.1875*(envelope.y-envelope.x)
                        slopes = local
                    else:
                        edge = 4.0
                    decay = direction.y+slopes.x*ti.abs(direction.x)+slopes.y*ti.abs(direction.z)
                    step = ti.min(edge,far-current)
                    if decay > 1e-8:
                        step = ti.min(step,.9*error/decay)
                    step = ti.max(.000002,step)
                    next_t = ti.min(current+step,far)
                    candidate = origin+direction*next_t
                    candidate_height = self.wave_height(candidate.x,candidate.z,clock)
                    if candidate.y >= candidate_height:
                        lo,hi = current,next_t
                        for _ in range(BISECT_STEPS):
                            mid = (lo+hi)*.5
                            q = origin+direction*mid
                            if q.y >= self.wave_height(q.x,q.z,clock):
                                hi = mid
                            else:
                                lo = mid
                        hit = (lo+hi)*.5
                    elif next_t >= far:
                        current = far+1.
                    else:
                        current = next_t
                        point,height = candidate,candidate_height
                count += 1
            if hit < 1e5:
                p = origin+direction*hit
                surface = self.sea.sample(p.x,p.z,clock)
                if surface.w <= .00001:
                    hit = 1e6
                else:
                    normal = ti.Vector([-surface.y,1.,-surface.z]).normalized()
        return hit,normal

    @ti.func
    def air_fog(self, color, t, direction, lighting):
        fog = 1.0 - ti.exp(-3.0 * (t / FOG_DISTANCE) ** 2)
        overhead = ti.max(0.0, direction.y)
        horizon = ti.Vector([direction.x, 0.02 + 0.65 * overhead * overhead, direction.z])
        fog_color = self.environment.sample_map(self.environment.environment,
                                                horizon.normalized(), lighting)
        return color * (1.0 - fog) + fog_color * fog


    @ti.kernel
    def trace_volume(self,eye:ti.types.vector(3,ti.f32),forward:ti.types.vector(3,ti.f32),
                     right:ti.types.vector(3,ti.f32),up:ti.types.vector(3,ti.f32),
                     scale:ti.f32,clock:ti.f32,clarity:ti.f32,lighting:ti.f32,samples:ti.i32):
        for vx,vy in self.volume:
            x,y = ti.min(self.width-1,vx*4+2),ti.min(self.height-1,vy*4+2)
            ray = self.camera_ray(x,y,0,samples,forward,right,up,scale)
            length = self._dive_hit_t[x,y,0].x
            if self._dive_mode[x,y,0] == 1:
                length = self._dive_surface_t[x,y,0]
            self.volume_depth[vx,vy] = length
            length = ti.min(40.,length)
            result = ti.Vector([0.,0.,0.])
            sun_water = self.sun_in_water(lighting)
            sun_air = self.sun_direction(lighting)
            g = .70
            cosine = ti.max(-1.,ti.min(1.,ray.dot(sun_water)))
            phase = (1-g*g)/(4*math.pi*ti.pow(1+g*g-2*g*cosine,1.5))
            coeff = ti.Vector([.34,.12,.065])/clarity
            dt = length/8.
            # Screen-space jitter is deterministic while paused.
            jitter = .2+.6*self.hash2(ti.Vector([ti.cast(vx,ti.f32),ti.cast(vy,ti.f32)]))
            for step in range(8):
                distance = (step+jitter)*dt
                point = eye+ray*distance
                surface,light_length = self.light_surface(point,clock,lighting)
                inside = self.wave_height(point.x,point.z,clock)>point.y
                if inside:
                    bed_light = self.occlusion_distance(point+sun_water*.002,sun_water,ti.max(.001,light_length-.004))
                    air_light = self.occlusion_distance(surface+sun_air*.003,sun_air,30.)
                    visible = ti.cast(bed_light>=light_length-.005 and air_light>=29.99,ti.f32)
                    bed_depth = ti.max(.1,surface.y-shore_terrain(surface.x,surface.z))
                    target = point-sun_water*ti.max(0.,(point.y-shore_terrain(point.x,point.z))/sun_water.y)
                    focus = self.dive_caustic_density(target)
                    progress = ti.min(1.,light_length/(bed_depth/sun_water.y))
                    focus = 1+(focus-1)*progress
                    attenuation = ti.exp(-coeff*(distance+light_length))
                    result += attenuation*ti.Vector([.032,.045,.046])*phase*dt*visible*focus*3.2
            self.volume[vx,vy] = result*self.dive_controls[None].y*(.35+.65*lighting)

    @ti.kernel
    def add_volume(self):
        for x,y in self.image:
            uv = ti.Vector([(x-2.)/4.,(y-2.)/4.])
            uv = ti.max(0.,ti.min(uv,ti.Vector([self.volume_size[0]-1.,self.volume_size[1]-1.])))
            i,j = ti.cast(ti.floor(uv.x),ti.i32),ti.cast(ti.floor(uv.y),ti.i32)
            fx,fy = uv.x-i,uv.y-j
            length = self._dive_hit_t[x,y,0].x
            if self._dive_mode[x,y,0] == 1:
                length = self._dive_surface_t[x,y,0]
            color,weight_sum = ti.Vector([0.,0.,0.]),0.
            for a,b in ti.static(ti.ndrange(2,2)):
                ii,jj = ti.min(self.volume_size[0]-1,i+a),ti.min(self.volume_size[1]-1,j+b)
                weight = (fx if a else 1-fx)*(fy if b else 1-fy)
                weight *= ti.exp(-ti.abs(ti.min(40.,length)-ti.min(40.,self.volume_depth[ii,jj]))*.8)
                color += self.volume[ii,jj]*weight
                weight_sum += weight
            self.post.hdr[x,y] += color/ti.max(1e-6,weight_sum)

    @ti.kernel
    def save_transition_air(self):
        for x,y in self.image:
            self.transition_air[x,y] = self.post.hdr[x,y]

    @ti.kernel
    def blend_waterline(self,depth:ti.f32,right_y:ti.f32,up_y:ti.f32,scale:ti.f32):
        for x,y in self.image:
            sx = (2*(x+.5)/self.width-1)*self.width/self.height*scale
            sy = (2*(y+.5)/self.height-1)*scale
            lens_height = .025*(right_y*sx+up_y*sy)
            signed = depth-lens_height
            coverage = self.smooth(-.003,.003,signed)
            color = self.transition_air[x,y]*(1-coverage)+self.post.hdr[x,y]*coverage
            # A subtle meniscus identifies the waterline without a hard seam.
            self.post.hdr[x,y] = color*(1-.12*ti.exp(-signed*signed/.000006))

    def _draw_underwater(self,eye,forward,right,up,clock,clarity,lighting,exposure,scale):
        self.render_dive(eye,forward,right,up,clock,clarity,lighting,exposure,scale,self.samples)
        self.dive_shade(eye,forward,right,up,scale,clock,clarity,lighting,self.samples)
        if self.dive_controls[None].y > .0001:
            self.trace_volume(eye,forward,right,up,scale,clock,clarity,lighting,self.samples)
            self.add_volume()

    def prepare(self,camera,clock,clarity=1.4,lighting=1.,exposure=1.05):
        """Compile BOTH media before opening a native window."""
        other = DiveCamera()
        apply_dive_preset(other,"overview")
        self.draw(other,clock,clarity,lighting,exposure)
        apply_dive_preset(other,"dive")
        self.draw(other,clock,clarity,lighting,exposure)
        # The transition kernels are small but also precompiled at startup.
        self.save_transition_air()
        self.blend_waterline(0.,0.,1.,.5)
        self.draw(camera,clock,clarity,lighting,exposure)

    def composite_effects(self,eye,forward,right,up,scale,clock,lighting,clarity):
        """Extension point for transparent effects before shared postprocessing."""
        pass

    def draw(self,camera,clock,clarity=1.4,lighting=0.,exposure=1.05):
        if self.samples > self.sample_capacity:
            raise ValueError("samples exceed the renderer's allocated sample capacity")
        self.view_clock[None],self.view_clarity[None] = clock,clarity
        self.prepare_water_bounds()
        self.trace_caustics(clock,lighting)
        eye,forward,right,up = camera.basis()
        self.view_eye[None] = eye
        self.view_angle[None] = 2*math.tan(math.radians(camera.fov/2))/self.height
        self._dive_medium(eye,clock)
        depth = float(self.medium_depth[None])
        scale = math.tan(math.radians(camera.fov/2))
        # A 5 cm finite-lens transition is an optical display approximation.
        # Each half is traced from the appropriate side of the surface.
        if abs(depth)<.03 and eye[1]-bed_height_py(float(eye[0]),float(eye[2]))>.06:
            air_eye = eye.copy();air_eye[1] += depth+.035
            water_eye = eye.copy();water_eye[1] += depth-.035
            self.render_above(air_eye,forward,right,up,clock,clarity,lighting,exposure,scale,self.samples)
            self.save_transition_air()
            self._draw_underwater(water_eye,forward,right,up,clock,clarity,lighting,exposure,scale)
            self.blend_waterline(depth,float(right[1]),float(up[1]),scale)
        elif self.medium_flag[None]>.5:
            self._draw_underwater(eye,forward,right,up,clock,clarity,lighting,exposure,scale)
        else:
            self.render_above(eye,forward,right,up,clock,clarity,lighting,exposure,scale,self.samples)
        self.composite_effects(eye,forward,right,up,scale,clock,lighting,clarity)
        bloom,threshold,vignette = self.post_controls[None]
        self.post.apply(exposure,bloom,threshold,vignette)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 07: dive underwater in the shallow shore")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=4, help="Spatial samples per pixel (default: 4)")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="day",
                        help="Day gives the sun elevation caustics and the Snell window need")
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
    parser.add_argument("--preset", choices=tuple(DIVE_PRESETS), default="dive")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_07.png"))
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


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = DiveRenderer(args.width, args.height, args.samples if args.headless else 4)
    renderer.samples = args.samples
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 0.5, 1.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = DiveCamera()
    apply_dive_preset(camera, args.preset)
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
    renderer.prepare(camera, clock, clarity, lighting, exposure)
    ti.sync()
    print(f"[Startup] First frame ready in {time.perf_counter() - warmup_start:.2f}s. Opening window...", flush=True)
    window = ti.ui.Window("07 / Dive", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    previous_time, previous_mouse = time.perf_counter(), None
    sim_accumulator, frame_ms, window_frames = 0.0, 0.0, 1
    drag_in_scene, click_start, click_camera_moved = False, None, False
    preset_keys = {"1": "overview", "2": "waterline", "3": "top", "4": "dive", "5": "window", "6": "bed"}
    print("Click water: splash | Drag LMB orbit | RMB pan | W/S zoom | 1-6 views (4 dive, 5 window, 6 bed) | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
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
            elif key in preset_keys:
                apply_dive_preset(camera, preset_keys[key])
                auto_orbit = False
            elif key == "r":
                apply_dive_preset(camera, "dive")
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
            camera.lift_above_bed()
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
                dive = renderer.dive_controls[None]
                dive[0] = gui.slider_float("Caustics", dive[0], 0.0, 2.0)
                dive[1] = gui.slider_float("Sun shafts", dive[1], 0.0, 2.0)
                renderer.dive_controls[None] = dive
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
            with gui.sub_window("DIVE / 07", 0.02, 0.025, 0.30, 0.78):
                gui.text("Underwater medium / Snell window")
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
                    apply_dive_preset(camera, "overview")
                    auto_orbit = False
                if gui.button("Waterline [2]"):
                    apply_dive_preset(camera, "waterline")
                    auto_orbit = False
                if gui.button("Top view [3]"):
                    apply_dive_preset(camera, "top")
                    auto_orbit = False
                if gui.button("Dive [4]"):
                    apply_dive_preset(camera, "dive")
                    auto_orbit = False
                if gui.button("Snell window [5]"):
                    apply_dive_preset(camera, "window")
                    auto_orbit = False
                if gui.button("Seabed [6]"):
                    apply_dive_preset(camera, "bed")
                    auto_orbit = False
                gui.text("Drag down to submerge the camera")
                gui.text("Click water to splash")
                gui.text("Sea presets: --sea calm/surf/storm")
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
