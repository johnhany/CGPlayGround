"""Night sky additions: hash-grid stars, the visible moon disk, and fbm
clouds that mask both.

Stars and the moon disk are evaluated per pixel on top of the sampled
environment color, not baked into the equirect maps, so they stay sharp.
The cloud layer shares the same fate: it is a per-pixel mask over the sky
dome, dimming the environment color inside the cloud body and blocking
stars and the moon according to coverage, with a faint silver lining on
the moon-facing side. The moon direction is the production one from
night_lake.moon, which sits low enough to stay inside the pitched-down
lake frame; scene.py re-bakes the night environment maps against the same
direction.
"""

import numpy as np
import taichi as ti

from night_lake.moon import (MOON_DIRECTION, MOON_DISK_BASE, MOON_DISK_LIMB,
                             MOON_DISK_PHASE)
from night_lake.terrain import _fbm_np, _fbm_ti, _smooth_ti

__all__ = ["MOON_DIRECTION", "night_sky", "cloud_mask_np"]

# Cloud body dims the dome by up to this fraction; the lining term adds a
# small moonlit rim back on. Stars are nearly extinguished inside a cloud,
# the moon disk only strongly attenuated so a covered moon still glows
# through as a bright patch.
CLOUD_DOME_DARKEN = 0.55
CLOUD_STAR_BLOCK = 0.90
CLOUD_MOON_BLOCK = 0.85


@ti.func
def _hash3(cell: ti.types.vector(3, ti.f32)) -> ti.f32:
    value = cell.dot(ti.Vector([127.1, 311.7, 74.7]))
    value = ti.sin(value) * 43758.5453
    return value - ti.floor(value)


@ti.func
def star_radiance(direction):
    """One point-like star per occupied hash cell, fading near the horizon."""
    grid = direction * 96.0
    cell = ti.floor(grid)
    local = grid - cell
    rnd = _hash3(cell)
    star = ti.Vector([0.0, 0.0, 0.0])
    if rnd > 0.982:
        center = ti.Vector([
            _hash3(cell + ti.Vector([13.0, 57.0, 91.0])),
            _hash3(cell + ti.Vector([91.0, 13.0, 57.0])),
            _hash3(cell + ti.Vector([57.0, 91.0, 13.0])),
        ])
        distance2 = (local - center).norm_sqr()
        falloff = ti.max(0.0, 1.0 - distance2 * 14.0)
        magnitude = (rnd - 0.982) / 0.018
        # Coarse patchiness keeps the star field from being uniform.
        patch = 0.35 + 0.65 * _hash3(ti.floor(direction * 7.0) + 271.0)
        horizon_fade = ti.min(1.0, ti.max(0.0, direction.y * 6.0))
        brightness = falloff * falloff * (0.4 + 5.0 * magnitude * magnitude) * patch * horizon_fade
        star = brightness * ti.Vector([0.88, 0.92, 1.0])
    return star


@ti.func
def moon_radiance(direction):
    """Small bright disk with a gentle phase shading and a narrow halo."""
    moon = ti.Vector([
        float(MOON_DIRECTION[0]),
        float(MOON_DIRECTION[1]),
        float(MOON_DIRECTION[2]),
    ])
    cosine = direction.dot(moon)
    result = ti.Vector([0.0, 0.0, 0.0])
    disk_cos = 0.9999619
    if cosine > disk_cos:
        # Projected offset from the disk center drives a subtle phase term.
        offset = direction - moon * cosine
        phase = 0.90
        if offset.norm_sqr() > 1e-12:
            axis = moon.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
            phase += 0.10 * offset.normalized().dot(axis)
        limb = ti.min(1.0, ti.sqrt(ti.max(0.0, 1.0 - cosine * cosine))
                      / ti.sqrt(1.0 - disk_cos * disk_cos))
        result = ti.Vector([1.0, 0.97, 0.88]) * (MOON_DISK_BASE + MOON_DISK_PHASE * phase) \
            * (1.0 - MOON_DISK_LIMB * limb)
    elif cosine > 0.9993:
        halo = (cosine - 0.9993) / (disk_cos - 0.9993)
        result = ti.Vector([0.55, 0.58, 0.66]) * halo * halo * 0.8
    return result


@ti.func
def cloud_mask(direction):
    """fbm cloud coverage in [0, 1]: blocky patches with thickness variation.

    Directions are projected onto a virtual cloud plane so the layer sits
    at a fixed apparent height; the horizon fade keeps the mask off the
    skyline and the lake reflection stays clean near the far bank.
    """
    mask = 0.0
    if direction.y > 0.02:
        inv = 1.0 / (direction.y + 0.30)
        u = direction.x * inv * 0.55 + 18.0
        v = direction.z * inv * 0.55 - 12.0
        base = _fbm_ti(u, v)
        detail = _fbm_ti(u * 2.3, v * 2.3)
        coverage = base * 0.84 + detail * 0.52
        mask = _smooth_ti(0.06, 0.32, coverage)
        mask *= ti.min(1.0, (direction.y - 0.02) * 4.0)
    return mask


def cloud_mask_np(direction):
    """CPU mirror of cloud_mask for tests and coverage checks."""
    d = np.asarray(direction, dtype=np.float32)
    y = d[..., 1]
    inv = np.float32(1.0) / (y + np.float32(0.30))
    u = d[..., 0] * inv * np.float32(0.55) + np.float32(18.0)
    v = d[..., 2] * inv * np.float32(0.55) - np.float32(12.0)
    base = _fbm_np(u, v)
    detail = _fbm_np(u * np.float32(2.3), v * np.float32(2.3))
    coverage = base * np.float32(0.84) + detail * np.float32(0.52)
    t = np.clip((coverage - np.float32(0.06)) / np.float32(0.26), 0.0, 1.0)
    mask = t * t * (np.float32(3.0) - np.float32(2.0) * t)
    horizon = np.minimum(np.float32(1.0), (y - np.float32(0.02)) * np.float32(4.0))
    mask = mask * horizon
    return np.where(y > 0.02, mask, np.float32(0.0)).astype(np.float32)


@ti.func
def _cloud_lining(direction, mask) -> ti.f32:
    """Faint moonlit rim on the cloud body, strongest toward the moon."""
    moon = ti.Vector([
        float(MOON_DIRECTION[0]),
        float(MOON_DIRECTION[1]),
        float(MOON_DIRECTION[2]),
    ])
    alignment = ti.max(0.0, direction.dot(moon))
    edge = mask * (1.0 - mask) * 4.0
    return edge * alignment * alignment * alignment * alignment


@ti.func
def night_sky(direction, color):
    """Apply clouds, stars, and the moon disk to a sampled dome color."""
    mask = cloud_mask(direction)
    result = color * (1.0 - mask * CLOUD_DOME_DARKEN)
    result += star_radiance(direction) * (1.0 - mask * CLOUD_STAR_BLOCK)
    result += moon_radiance(direction) * (1.0 - mask * CLOUD_MOON_BLOCK)
    lining = _cloud_lining(direction, mask)
    result += ti.Vector([0.55, 0.58, 0.66]) * (lining * 0.10)
    return result
