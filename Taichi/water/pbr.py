"""GGX material lighting and deterministic procedural HDR environment baking.

The environment contains no direct sun: the area sun is evaluated separately.
IBL uses cosine-convolved irradiance, GGX-prefiltered radiance, and a split-sum
BRDF lookup table. All maps and BRDF calculations use linear HDR values.
"""

import math

import numpy as np
import taichi as ti


@ti.func
def fresnel_schlick(cosine, f0):
    cosine = ti.max(0.0, ti.min(1.0, cosine))
    return f0 + (1.0 - f0) * (1.0 - cosine) ** 5


@ti.func
def ggx_brdf(base, roughness, metallic, normal, view, light, dielectric_f0):
    """Cook-Torrance + energy-weighted Lambert, multiplied by N dot L."""
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
        f0 = ti.Vector([dielectric_f0, dielectric_f0, dielectric_f0]) * (1.0 - metallic) + base * metallic
        fresnel = fresnel_schlick(vh, f0)
        specular = distribution * gv * gl * fresnel / ti.max(4.0 * nv * nl, 1e-6)
        diffuse = base * (1.0 - metallic) * (1.0 - fresnel) / math.pi
        result = (diffuse + specular) * nl
    return result


def hammersley(count):
    """Fixed low-discrepancy samples, without random per-frame noise."""
    indices = np.arange(count, dtype=np.uint32)
    bits = indices.copy()
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return np.stack([(indices + 0.5) / count, bits * 2.3283064365386963e-10], axis=1)


def direction_grid(width, height):
    longitude = (np.arange(width) + 0.5) / width * 2 * math.pi - math.pi
    latitude = (np.arange(height) + 0.5) / height * math.pi
    lon, lat = np.meshgrid(longitude, latitude, indexing="ij")
    return np.stack([np.sin(lat) * np.sin(lon), np.cos(lat), np.sin(lat) * np.cos(lon)], axis=-1)


def procedural_environment(directions, daylight):
    y = directions[..., 1]
    azimuth = np.arctan2(directions[..., 0], directions[..., 2])
    t = np.clip(y, 0, 1)[..., None] ** 0.55
    horizon = np.array([0.88, 1.06, 1.24]) if daylight else np.array([1.35, 0.59, 0.26])
    zenith = np.array([0.17, 0.39, 0.78]) if daylight else np.array([0.16, 0.23, 0.47])
    color = horizon * (1 - t) + zenith * t
    # Broad, soft cloud bands become light sources in rough surface reflections.
    cloud = np.sin(azimuth * 3.0 + y * 8.0) * np.sin(azimuth * 5.0 - y * 4.0)
    cloud = np.clip((cloud - 0.12) * 1.7, 0, 1) * np.clip(y * 6, 0, 1)
    cloud *= np.clip((0.85 - y) * 3, 0, 1)
    cloud_color = np.array([1.7, 1.72, 1.65]) if daylight else np.array([1.8, 0.92, 0.48])
    color = color * (1 - cloud[..., None] * 0.55) + cloud_color * cloud[..., None] * 0.55
    ridge = 0.028 + 0.018 * np.sin(azimuth * 5) + 0.01 * np.sin(azimuth * 13)
    hill = (y >= 0) & (y < ridge)
    color = np.where(hill[..., None], np.array([0.23, 0.32, 0.29]), color)
    ground = np.broadcast_to([0.20, 0.19, 0.16], color.shape)
    return np.where((y < 0)[..., None], ground, color).astype(np.float32)


def _basis(normals):
    reference = np.zeros_like(normals)
    reference[..., 1] = 1
    reference[np.abs(normals[..., 1]) > 0.95] = [1, 0, 0]
    tangent = np.cross(reference, normals)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent, np.cross(normals, tangent)


def convolve_environment(normals, roughness=None, count=64, sampler=None):
    """Cosine diffuse integral, or normalized GGX specular prefilter."""
    if sampler is None:
        sampler = lambda directions: procedural_environment(directions, True)
    tangent, bitangent = _basis(normals)
    accumulated = np.zeros_like(normals, dtype=np.float64)
    weights = np.zeros(normals.shape[:-1], dtype=np.float64)
    for u, v in hammersley(count):
        phi = 2 * math.pi * v
        if roughness is None:
            cos_theta = math.sqrt(1 - u)
        else:
            alpha2 = max(roughness, 0.045) ** 4
            cos_theta = math.sqrt((1 - u) / (1 + (alpha2 - 1) * u))
        sin_theta = math.sqrt(max(0, 1 - cos_theta * cos_theta))
        half_vector = tangent * (math.cos(phi) * sin_theta) + bitangent * (math.sin(phi) * sin_theta) + normals * cos_theta
        direction, weight = half_vector, np.ones(normals.shape[:-1])
        if roughness is not None:
            direction = 2 * cos_theta * half_vector - normals
            weight = np.maximum(np.sum(direction * normals, axis=-1), 0)
        accumulated += sampler(direction) * weight[..., None]
        weights += weight
    result = accumulated / np.maximum(weights[..., None], 1e-8)
    if roughness is None:
        result *= math.pi
    return result.astype(np.float32)


def bake_brdf_lut(size=32, count=128):
    nv, rough = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size, indexing="ij")
    alpha2 = np.maximum(rough, 0.045) ** 4
    vx = np.sqrt(1 - nv * nv)
    a, b = np.zeros_like(nv), np.zeros_like(nv)
    for u, v in hammersley(count):
        phi = 2 * math.pi * v
        nh = np.sqrt((1 - u) / (1 + (alpha2 - 1) * u))
        sin_theta = np.sqrt(np.maximum(0, 1 - nh * nh))
        hx = math.cos(phi) * sin_theta
        vh = np.maximum(vx * hx + nv * nh, 0)
        nl = np.maximum(2 * vh * nh - nv, 0)
        gv = 2 * nv / (nv + np.sqrt(alpha2 + (1 - alpha2) * nv * nv))
        gl = 2 * nl / np.maximum(nl + np.sqrt(alpha2 + (1 - alpha2) * nl * nl), 1e-8)
        visibility = gv * gl * vh / np.maximum(nh * nv, 1e-8)
        fc = (1 - vh) ** 5
        a += (1 - fc) * visibility
        b += fc * visibility
    return np.stack([a / count, b / count], axis=-1).astype(np.float32)


@ti.data_oriented
class EnvironmentLighting:
    def __init__(self):
        self.environment = ti.Vector.field(3, ti.f32, shape=(2, 128, 64))
        self.irradiance = ti.Vector.field(3, ti.f32, shape=(2, 32, 16))
        self.prefiltered = ti.Vector.field(3, ti.f32, shape=(2, 5, 64, 32))
        self.brdf_lut = ti.Vector.field(2, ti.f32, shape=(32, 32))
        sky, irradiance, specular = [], [], []
        for daylight in (False, True):
            sampler = lambda directions, day=daylight: procedural_environment(directions, day)
            sky.append(sampler(direction_grid(128, 64)))
            irradiance.append(convolve_environment(direction_grid(32, 16), sampler=sampler))
            specular.append([convolve_environment(direction_grid(64, 32), roughness=level / 4,
                                                 sampler=sampler) for level in range(5)])
        self.environment.from_numpy(np.array(sky, dtype=np.float32))
        self.irradiance.from_numpy(np.array(irradiance, dtype=np.float32))
        self.prefiltered.from_numpy(np.array(specular, dtype=np.float32))
        self.brdf_lut.from_numpy(bake_brdf_lut())

    @ti.func
    def coordinates(self, direction, width, height):
        longitude = ti.atan2(direction.x, direction.z) / (2 * math.pi) + 0.5
        latitude = ti.acos(ti.min(1.0, ti.max(-1.0, direction.y))) / math.pi
        return longitude * width - 0.5, ti.max(0.0, ti.min(height - 1.0, latitude * height - 0.5))

    @ti.func
    def sample_map(self, image: ti.template(), direction, daylight):
        x, y = self.coordinates(direction, ti.static(image.shape[1]), ti.static(image.shape[2]))
        ix, iy = ti.cast(ti.floor(x), ti.i32), ti.cast(ti.floor(y), ti.i32)
        fx, fy = x - ti.floor(x), y - iy
        result = ti.Vector([0.0, 0.0, 0.0])
        for a, b in ti.static(ti.ndrange(2, 2)):
            weight = (fx if a else 1 - fx) * (fy if b else 1 - fy)
            px = (ix + a + ti.static(image.shape[1])) % ti.static(image.shape[1])
            py = ti.min(iy + b, ti.static(image.shape[2]) - 1)
            result += weight * (image[0, px, py] * (1 - daylight) + image[1, px, py] * daylight)
        return result

    @ti.func
    def specular(self, direction, roughness, daylight):
        x, y = self.coordinates(direction, 64, 32)
        ix, iy = ti.cast(ti.floor(x), ti.i32), ti.cast(ti.floor(y), ti.i32)
        fx, fy = x - ti.floor(x), y - iy
        level = ti.max(0.0, ti.min(4.0, roughness * 4.0))
        low = ti.cast(ti.floor(level), ti.i32)
        fraction = level - low
        result = ti.Vector([0.0, 0.0, 0.0])
        for a, b, c in ti.static(ti.ndrange(2, 2, 2)):
            weight = (fx if a else 1 - fx) * (fy if b else 1 - fy) * (fraction if c else 1 - fraction)
            px, py, mip = (ix + a + 64) % 64, ti.min(iy + b, 31), ti.min(low + c, 4)
            result += weight * (self.prefiltered[0, mip, px, py] * (1 - daylight) + self.prefiltered[1, mip, px, py] * daylight)
        return result

    @ti.func
    def integrated_brdf(self, nv, roughness):
        x = ti.max(0.0, ti.min(31.0, nv * 32 - 0.5))
        y = ti.max(0.0, ti.min(31.0, roughness * 32 - 0.5))
        ix, iy = ti.cast(ti.floor(x), ti.i32), ti.cast(ti.floor(y), ti.i32)
        fx, fy = x - ix, y - iy
        result = ti.Vector([0.0, 0.0])
        for a, b in ti.static(ti.ndrange(2, 2)):
            weight = (fx if a else 1 - fx) * (fy if b else 1 - fy)
            result += self.brdf_lut[ti.min(ix + a, 31), ti.min(iy + b, 31)] * weight
        return result
