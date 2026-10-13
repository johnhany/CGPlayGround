"""Shore terrain: procedural beach height field and material zoning.

The shoreline runs across the view a short distance ahead of the camera;
its plan shape is a sum of incommensurate sines, and the beach profile
flattens toward the waterline so the land-water transition stays gradual.
Value-noise fbm adds surface detail on land only, keeping the waterline
itself clean. Height and gradient are analytic (the gradient uses central
differences of the height function, so the two never disagree), and a
numpy mirror of the height function provides CPU-side testing.

Materials zone by height above the water and local slope: bare wet mud
near and below the waterline, grass-root soil higher up, with wetness
driving darkening and a low roughness sheen.
"""

import numpy as np
import taichi as ti

__all__ = ["SHORE_Z_DEFAULT", "BEACH_SLOPE", "SHORE_FLAT", "SEA_SLOPE",
           "DETAIL_AMP", "WET_SCALE_DEFAULT", "GRASS_LINE_DEFAULT",
           "terrain_height_np", "terrain_gradient_np",
           "terrain_height", "terrain_gradient", "terrain_mottle",
           "terrain_material"]

# Beach profile: slope grows from zero at the waterline to BEACH_SLOPE
# inland over SHORE_FLAT metres; offshore descends linearly at SEA_SLOPE.
# The waterline now crosses the view only ~16 m ahead of the camera, so
# the lake fills the lower half of the frame; the camera eye sits at
# 2.2 m, more than a metre above the local beach.
SHORE_Z_DEFAULT = 16.0
BEACH_SLOPE = 0.04
SHORE_FLAT = 3.0
SEA_SLOPE = 0.05
DETAIL_AMP = 0.18
WET_SCALE_DEFAULT = 1.2
GRASS_LINE_DEFAULT = 0.45
FBM_OCTAVES = 3
NOISE_FREQ = 0.7
# Far bank: the lake closes at a wooded ridge across the water so the
# horizon layer reads as a shoreline instead of open water. The rise is
# additive, starts well past the ripple grid, and its plan shape wiggles
# with low-frequency noise so the far waterline meanders like the near one.
BANK_Z0, BANK_Z1 = 62.0, 86.0
BANK_HEIGHT = 4.5
BANK_NOISE_FREQ = 0.14

# Water and fog colors shared with the scene for a continuous waterline.
FOG_COLOR = (0.020, 0.028, 0.042)
DEEP_WATER = (0.004, 0.009, 0.013)


def _hash_np(ix, iz):
    """Deterministic uint32 lattice hash, identical to the taichi version."""
    h = (ix.astype(np.uint32) * np.uint32(374761393)
         + iz.astype(np.uint32) * np.uint32(668265263))
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    h = h ^ (h >> np.uint32(16))
    return h.astype(np.float32) * np.float32(2.3283064e-10)


@ti.func
def _hash_ti(ix: ti.i32, iz: ti.i32) -> ti.f32:
    h = ti.cast(ix, ti.u32) * ti.u32(374761393) + ti.cast(iz, ti.u32) * ti.u32(668265263)
    h = (h ^ (h >> ti.u32(13))) * ti.u32(1274126177)
    h = h ^ (h >> ti.u32(16))
    return ti.cast(h, ti.f32) * 2.3283064e-10


def _value_noise_np(x, z):
    ix, iz = np.floor(x), np.floor(z)
    fx, fz = (x - ix).astype(np.float32), (z - iz).astype(np.float32)
    sx = (fx * fx * (np.float32(3.0) - np.float32(2.0) * fx)).astype(np.float32)
    sz = (fz * fz * (np.float32(3.0) - np.float32(2.0) * fz)).astype(np.float32)
    i0, j0 = ix.astype(np.int32), iz.astype(np.int32)
    h00 = _hash_np(i0, j0)
    h10 = _hash_np(i0 + 1, j0)
    h01 = _hash_np(i0, j0 + 1)
    h11 = _hash_np(i0 + 1, j0 + 1)
    return ((h00 * (np.float32(1.0) - sx) + h10 * sx) * (np.float32(1.0) - sz)
            + (h01 * (np.float32(1.0) - sx) + h11 * sx) * sz) - np.float32(0.5)


@ti.func
def _value_noise_ti(x: ti.f32, z: ti.f32) -> ti.f32:
    ix, iz = ti.floor(x), ti.floor(z)
    fx, fz = x - ix, z - iz
    sx = fx * fx * (3.0 - 2.0 * fx)
    sz = fz * fz * (3.0 - 2.0 * fz)
    i0, j0 = ti.cast(ix, ti.i32), ti.cast(iz, ti.i32)
    h00 = _hash_ti(i0, j0)
    h10 = _hash_ti(i0 + 1, j0)
    h01 = _hash_ti(i0, j0 + 1)
    h11 = _hash_ti(i0 + 1, j0 + 1)
    return ((h00 * (1.0 - sx) + h10 * sx) * (1.0 - sz)
            + (h01 * (1.0 - sx) + h11 * sx) * sz) - 0.5


def _fbm_np(x, z):
    total, amplitude, frequency, norm = np.zeros_like(x, dtype=np.float32), 1.0, NOISE_FREQ, 0.0
    for _ in range(FBM_OCTAVES):
        total = (total + np.float32(amplitude) * _value_noise_np(x * np.float32(frequency),
                                                                 z * np.float32(frequency))).astype(np.float32)
        norm += amplitude
        amplitude *= 0.5
        frequency *= 2.03
    return total / np.float32(norm)


@ti.func
def _fbm_ti(x: ti.f32, z: ti.f32) -> ti.f32:
    total, amplitude, frequency, norm = 0.0, 1.0, NOISE_FREQ, 0.0
    for _ in ti.static(range(FBM_OCTAVES)):
        total += amplitude * _value_noise_ti(x * frequency, z * frequency)
        norm += amplitude
        amplitude *= 0.5
        frequency *= 2.03
    return total / norm


def _smooth_np(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def _smooth_ti(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.max(0.0, ti.min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _shoreline_np(x):
    # Zero-mean wiggle: the waterline sits at shore_z plus a few metres of
    # meander, so shore_z reads directly as the across-view distance.
    return (1.00 * np.sin(0.35 * x + 1.2)
            + 0.60 * np.sin(0.83 * x + 4.0)
            + 0.35 * np.sin(1.85 * x + 2.3))


@ti.func
def _shoreline_ti(x: ti.f32) -> ti.f32:
    return (1.00 * ti.sin(0.35 * x + 1.2)
            + 0.60 * ti.sin(0.83 * x + 4.0)
            + 0.35 * ti.sin(1.85 * x + 2.3))


def _bank_rise_np(x, z):
    t = np.clip((z - np.float32(BANK_Z0)) / np.float32(BANK_Z1 - BANK_Z0),
                np.float32(0.0), np.float32(1.0))
    profile = (t * t * (np.float32(3.0) - np.float32(2.0) * t)).astype(np.float32)
    shape = np.float32(0.7) + np.float32(0.6) * (_fbm_np(x * np.float32(BANK_NOISE_FREQ),
                                                        z * np.float32(BANK_NOISE_FREQ))
                                                 + np.float32(0.5))
    return (np.float32(BANK_HEIGHT) * profile * shape).astype(np.float32)


@ti.func
def _bank_rise_ti(x: ti.f32, z: ti.f32) -> ti.f32:
    t = ti.max(0.0, ti.min(1.0, (z - BANK_Z0) / (BANK_Z1 - BANK_Z0)))
    profile = t * t * (3.0 - 2.0 * t)
    shape = 0.7 + 0.6 * (_fbm_ti(x * BANK_NOISE_FREQ, z * BANK_NOISE_FREQ) + 0.5)
    return BANK_HEIGHT * profile * shape


def terrain_height_np(x, z, shore_z=SHORE_Z_DEFAULT, detail=1.0):
    """Analytic terrain height; the CPU mirror of the taichi function."""
    x = np.asarray(x, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    d = (np.float32(shore_z) + _shoreline_np(x)) - z
    land = BEACH_SLOPE * (d - np.float32(SHORE_FLAT) * (np.float32(1.0) - np.exp(-d / np.float32(SHORE_FLAT))))
    sea = np.float32(SEA_SLOPE) * d
    height = np.where(d >= 0.0, land, sea).astype(np.float32)
    detail_gain = (_smooth_np(np.float32(0.0), np.float32(1.2), d)
                   * np.float32(DETAIL_AMP) * np.float32(detail))
    height = (height + detail_gain * _fbm_np(x, z)
              + _bank_rise_np(x, z)).astype(np.float32)
    return height


@ti.func
def terrain_height(x: ti.f32, z: ti.f32, shore_z: ti.f32, detail: ti.f32) -> ti.f32:
    """Analytic terrain height: beach profile, sine shoreline, fbm detail."""
    d = (shore_z + _shoreline_ti(x)) - z
    height = SEA_SLOPE * d
    if d >= 0.0:
        height = BEACH_SLOPE * (d - SHORE_FLAT * (1.0 - ti.exp(-d / SHORE_FLAT)))
    return (height + _smooth_ti(0.0, 1.2, d) * DETAIL_AMP * detail * _fbm_ti(x, z)
            + _bank_rise_ti(x, z))


def vegetation_patch_noise(x, z):
    """Low-frequency fbm in [-0.5, 0.5] driving vegetation patchiness:
    meadows, lodged swathes, and withered stands read as large-scale
    structure instead of uniform scatter."""
    return _fbm_np(np.asarray(x, dtype=np.float32) * np.float32(0.13),
                   np.asarray(z, dtype=np.float32) * np.float32(0.13))


def terrain_gradient_np(x, z, shore_z=SHORE_Z_DEFAULT, detail=1.0, eps=0.06):
    dx = (terrain_height_np(x + eps, z, shore_z, detail)
          - terrain_height_np(x - eps, z, shore_z, detail)) / (2 * eps)
    dz = (terrain_height_np(x, z + eps, shore_z, detail)
          - terrain_height_np(x, z - eps, shore_z, detail)) / (2 * eps)
    return np.stack([dx, dz], axis=-1)


@ti.func
def terrain_gradient(x: ti.f32, z: ti.f32, shore_z: ti.f32, detail: ti.f32) -> ti.types.vector(2, ti.f32):
    """Central-difference gradient of the height function (eps in metres)."""
    eps = 0.06
    gx = (terrain_height(x + eps, z, shore_z, detail)
          - terrain_height(x - eps, z, shore_z, detail)) / (2 * eps)
    gz = (terrain_height(x, z + eps, shore_z, detail)
          - terrain_height(x, z - eps, shore_z, detail)) / (2 * eps)
    return ti.Vector([gx, gz])


@ti.func
def terrain_mottle(x: ti.f32, z: ti.f32) -> ti.f32:
    """Low-frequency fbm in [-0.5, 0.5] for albedo patchiness."""
    return _fbm_ti(x * 0.22, z * 0.22)


@ti.func
def terrain_material(h: ti.f32, slope: ti.f32, wet_scale: ti.f32,
                     grass_line: ti.f32) -> ti.types.vector(3, ti.f32):
    """Zone the surface: wetness, grass-soil weight, roughness."""
    wet = 1.0
    if h > 0.0:
        wet = ti.exp(-h / ti.max(0.05, wet_scale))
    grass = _smooth_ti(grass_line * 0.4, grass_line * 1.4, h)
    grass *= 1.0 - 0.5 * _smooth_ti(0.3, 0.7, slope)
    grass *= 1.0 - 0.7 * wet
    roughness = (1.0 - grass) * ((1.0 - wet) * 0.8 + wet * 0.22) + grass * 0.9
    return ti.Vector([wet, grass, roughness])
