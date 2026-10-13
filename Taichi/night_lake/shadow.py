"""Baked orthographic moon shadow map.

The scene geometry is static (vegetation sways a little, which the bake
treats as resting pose plus a conservative radius margin) and the moon
direction is fixed, so the shadow map is baked once: for every texel of a
light-space depth map we find the nearest occluder along the moon ray,
marching the terrain height field and testing tree cones and grass and
leaf cards analytically, and store its depth. Shading samples the map with
a 3x3 percentage-closer filter and a slope-scaled bias.

The bake runs in numpy and is cached: the inputs (terrain controls and the
fixed-seed vegetation) are constant for the whole run, and every test
scene then only uploads the depth map once. RebaKing happens only if the
terrain controls change.
"""

import numpy as np
import taichi as ti

from night_lake.accel import WIND_MARGIN_MAX
from night_lake.moon import MOON_DIRECTION
from night_lake.terrain import terrain_height_np

__all__ = ["RES_U", "RES_V", "bake_moon_shadow", "shadow_factor"]

RES_U, RES_V = 384, 224
BIAS_BASE = 0.25
BIAS_SLOPE = 0.6
# Texels start this far before the scene along the light ray.
DEPTH_MARGIN = 12.0

# Light-space frame: FORWARD is the moon ray travel direction, RIGHT and
# UP span the orthographic window.
_FORWARD = -MOON_DIRECTION / np.linalg.norm(MOON_DIRECTION)
_RIGHT = np.cross(_FORWARD, np.array([0.0, 1.0, 0.0]))
_RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_FORWARD, _RIGHT)
_UP /= np.linalg.norm(_UP)


def _frame():
    corners = np.array([[x, y, z] for x in (-48.0, 48.0)
                        for y in (-4.0, 15.0) for z in (-14.0, 96.0)])
    u = corners @ _RIGHT
    v = corners @ _UP
    d = corners @ _FORWARD
    u0, u1 = float(u.min()), float(u.max())
    v0, v1 = float(v.min()), float(v.max())
    d0 = float(d.min()) - DEPTH_MARGIN
    span_d = float(d.max()) - d0
    origin = _RIGHT * (u0 + u1) / 2 + _UP * (v0 + v1) / 2 + _FORWARD * d0
    return origin, u1 - u0, v1 - v0, span_d


_ORIGIN, _SPAN_U, _SPAN_V, _SPAN_D = _frame()

OX, OY, OZ = (float(v) for v in _ORIGIN)
FX, FY, FZ = (float(v) for v in _FORWARD)
RX, RY, RZ = (float(v) for v in _RIGHT)
UX, UY, UZ = (float(v) for v in _UP)
MX, MY, MZ = (float(v) for v in MOON_DIRECTION)

_CACHE = {}


def bake_moon_shadow(trees, grass, shore_z, detail):
    """Depth map over the scene window, cached on the bake inputs."""
    key = (round(float(shore_z), 4), round(float(detail), 4),
           int(trees.count), int(grass.cards.count), int(grass.blade_count))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _bake(trees, grass, float(shore_z), float(detail))
    return _CACHE[key]


def _bake(trees, grass, shore_z, detail):
    nu, nv = RES_U, RES_V
    us = _SPAN_U * (np.arange(nu) + 0.5) / nu - _SPAN_U / 2
    vs = _SPAN_V * (np.arange(nv) + 0.5) / nv - _SPAN_V / 2
    grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
    origins = (_ORIGIN[None, None, :]
               + _RIGHT[None, None, :] * grid_u[..., None]
               + _UP[None, None, :] * grid_v[..., None]).reshape(-1, 3)
    depth = np.full(len(origins), np.inf)
    depth = np.minimum(depth, _march_terrain(origins, shore_z, detail))
    depth = np.minimum(depth, _intersect_segments(origins, trees))
    depth = np.minimum(depth, _intersect_cards(origins, grass.cards))
    depth = np.minimum(depth, _intersect_cards(origins, trees.leaves))
    far = np.float32(_SPAN_D + 10.0)
    depth = np.where(np.isfinite(depth), depth, far).astype(np.float32)
    return depth.reshape(nu, nv)


def _march_terrain(origins, shore_z, detail):
    """Error-controlled march of all texel rays through the terrain.

    Heights come from a lookup table so the fbm is evaluated once per grid
    vertex instead of once per march step.
    """
    step = 0.25
    xs = np.arange(-48.0, 48.0 + step, step)
    zs = np.arange(-14.0, 96.0 + step, step)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    table = terrain_height_np(gx.ravel(), gz.ravel(), shore_z, detail)
    table = table.reshape(len(xs), len(zs))

    def height_at(x, z):
        fx = np.clip((x - xs[0]) / step, 0.0, len(xs) - 1.001)
        fz = np.clip((z - zs[0]) / step, 0.0, len(zs) - 1.001)
        ix, iz = np.floor(fx).astype(np.int64), np.floor(fz).astype(np.int64)
        ax, az = fx - ix, fz - iz
        return ((table[ix, iz] * (1 - ax) + table[ix + 1, iz] * ax) * (1 - az)
                + (table[ix, iz + 1] * (1 - ax) + table[ix + 1, iz + 1] * ax) * az)

    t = np.zeros(len(origins))
    err = origins[:, 1] - height_at(origins[:, 0], origins[:, 2])
    depth = np.full(len(origins), np.inf)
    active = (err > 0.0) & (t < _SPAN_D)
    for _ in range(240):
        if not active.any():
            break
        idx = np.nonzero(active)[0]
        stride = np.clip(err[idx] * 1.2, 0.03, 0.03 + 0.12 * t[idx])
        t[idx] += stride
        p = origins[idx] + _FORWARD[None, :] * t[idx, None]
        err[idx] = p[:, 1] - height_at(p[:, 0], p[:, 2])
        crossed = idx[err[idx] <= 0.0]
        depth[crossed] = t[crossed]
        done = np.zeros(len(origins), dtype=bool)
        done[crossed] = True
        active = active & ~done
        active[idx] &= t[idx] < _SPAN_D
    return depth


def _intersect_segments(origins, trees):
    """Nearest cone hit per texel ray, mirroring TreeField.cone_hit."""
    if trees.count == 0:
        return np.full(len(origins), np.inf)
    a_all = trees.seg_a.to_numpy()[:trees.count]
    b_all = trees.seg_b.to_numpy()[:trees.count]
    r_all = trees.seg_radius.to_numpy()[:trees.count]
    best = np.full(len(origins), np.inf)
    u_o = origins @ _RIGHT
    v_o = origins @ _UP
    for k in range(trees.count):
        a, b = a_all[k], b_all[k]
        r0, r1 = float(r_all[k][0]), float(r_all[k][1])
        axis = b - a
        length = float(np.linalg.norm(axis))
        axis = axis / length
        slope = (r1 - r0) / length
        centre = (a + b) / 2
        radius = length / 2 + max(r0, r1) + 0.02
        cu = float(centre @ _RIGHT)
        cv = float(centre @ _UP)
        mask = (u_o - cu) ** 2 + (v_o - cv) ** 2 < radius * radius
        if not mask.any():
            continue
        oc = origins[mask] - a[None, :]
        s0 = oc @ axis
        d0 = oc - axis[None, :] * s0[:, None]
        du = float(_FORWARD @ axis)
        dd = _FORWARD - axis * du
        r0v = r0 + slope * s0
        qa = float(dd @ dd) - slope * slope * du * du
        qb = 2.0 * (d0 @ dd - r0v * slope * du)
        qc = np.einsum("ij,ij->i", d0, d0) - r0v * r0v
        disc = qb * qb - 4.0 * qa * qc
        ok = disc > 0.0
        if not ok.any():
            continue
        root = np.sqrt(np.where(ok, disc, 0.0))
        qa_safe = qa if abs(qa) > 1e-9 else 1e-9
        for sign in (-1.0, 1.0):
            cand = (-qb + sign * root) / (2.0 * qa_safe)
            s = s0 + cand * du
            hit = (ok & (cand > 0.001) & (s >= 0.0) & (s <= length))
            flat = np.nonzero(mask)[0]
            better = hit & (cand < best[flat])
            best[flat[better]] = cand[better]
    return best


def _intersect_cards(origins, cards):
    """Nearest card quad hit per texel ray, in resting pose with the wind
    sway margin added to the pruning radius so baked shadows stay
    conservative when vegetation swings."""
    if cards.count == 0:
        return np.full(len(origins), np.inf)
    c_all = cards.card_center.to_numpy()[:cards.count]
    u_all = cards.card_u.to_numpy()[:cards.count]
    v_all = cards.card_v.to_numpy()[:cards.count]
    best = np.full(len(origins), np.inf)
    u_o = origins @ _RIGHT
    v_o = origins @ _UP
    for k in range(cards.count):
        c, u, v = c_all[k], u_all[k], v_all[k]
        normal = np.cross(u, v)
        radius = (np.linalg.norm(u) + np.linalg.norm(v)
                  + 0.10 * WIND_MARGIN_MAX * np.linalg.norm(v) + 0.02)
        cu = float(c @ _RIGHT)
        cv = float(c @ _UP)
        mask = (u_o - cu) ** 2 + (v_o - cv) ** 2 < radius * radius
        if not mask.any():
            continue
        denom = float(_FORWARD @ normal)
        if abs(denom) < 1e-6:
            continue
        flat = np.nonzero(mask)[0]
        t = ((c[None, :] - origins[flat]) @ normal) / denom
        rel = origins[flat] + _FORWARD[None, :] * t[:, None] - c[None, :]
        a = (rel @ u) / float(u @ u)
        b = (rel @ v) / float(v @ v)
        hit = (t > 0.001) & (np.abs(a) <= 1.0) & (np.abs(b) <= 1.0)
        better = hit & (t < best[flat])
        best[flat[better]] = t[better]
    return best


@ti.func
def shadow_factor(depth: ti.template(), p, n):
    """3x3 percentage-closer sample of the baked moon shadow map.

    Returns 1 in full moonlight, 0 behind an occluder. The bias grows on
    surfaces grazing to the light, which keeps the low moon's long
    projection acne-free on flat ground.
    """
    rel = ti.Vector([p.x - OX, p.y - OY, p.z - OZ])
    d = rel.dot(ti.Vector([FX, FY, FZ]))
    nl = n.dot(ti.Vector([MX, MY, MZ]))
    bias = BIAS_BASE + BIAS_SLOPE * ti.max(0.0, 1.0 - nl)
    lu = rel.dot(ti.Vector([RX, RY, RZ]))
    lv = rel.dot(ti.Vector([UX, UY, UZ]))
    iu = ti.cast((lu + _SPAN_U * 0.5) / _SPAN_U * RES_U, ti.i32)
    iv = ti.cast((lv + _SPAN_V * 0.5) / _SPAN_V * RES_V, ti.i32)
    lit = 0.0
    for a, b in ti.static(ti.ndrange((-1, 2), (-1, 2))):
        uu, vv = iu + a, iv + b
        if 0 <= uu < RES_U and 0 <= vv < RES_V:
            if d - bias <= depth[uu, vv]:
                lit += 1.0
    return lit * (1.0 / 9.0)
