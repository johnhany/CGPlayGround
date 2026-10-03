"""Wind-driven ocean: inverse-FFT swell from a Phillips spectrum.

The sea surface is a periodic 512 m tile synthesized from a Phillips
spectrum. A 512-point master and coherent 128/32-point low-pass levels share
Fourier modes and phases. Continuous cubic sampling supplies the height and
its exact derivative; shading LOD follows the projected pixel footprint.
Deep water hides the seabed and avoids height-based color bands. Ray marching,
PBR shading, fog, and the lighthouse skerry reuse demo 03's wave hooks.

Run: uv run python water/demo_05.py
Export: uv run python water/demo_05.py --headless --output output/windsea.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .demo_03 import (FOG_DISTANCE, GRAVITY, OCEAN_PRESETS, SEABED,
                          SunsetOceanRenderer, apply_preset)
    from .scene import OrbitCamera
else:
    from demo_03 import (FOG_DISTANCE, GRAVITY, OCEAN_PRESETS, SEABED,
                         SunsetOceanRenderer, apply_preset)
    from scene import OrbitCamera

__all__ = ["FFTLevel", "FFTOcean", "WindOceanRenderer", "main", "parse_args",
           "TILE_LENGTH", "LEVEL_SIZES", "SEA_PRESETS"]

TILE_LENGTH = 512.0
# Coarse levels crop and filter the master spectrum without renormalizing it.
LEVEL_SIZES = (512, 128, 32)
FOAM_DET_LOW = 0.90
FOAM_DET_HIGH = 0.70
# Sea presets: (wind speed m/s, wind direction deg, height, chop, swell mode)
SEA_PRESETS = {
    "breeze": (3.0, 35.0, 0.35, 0.25, False),
    "swell": (6.0, 35.0, 0.9, 0.15, True),
    "gale": (18.0, 35.0, 2.6, 0.9, False),
}


@ti.func
def band_smooth(a, b, x):
    u = ti.max(0.0, ti.min(1.0, (x - a) / (b - a)))
    return u * u * (3.0 - 2.0 * u)


@ti.data_oriented
class FFTLevel:
    """One resolution level of the ocean tile: spectrum plus a 2D inverse FFT.

    The transform runs entirely in Taichi: a bit-reversal permutation
    followed by radix-2 butterfly stages along rows, then the same along
    columns, ping-ponging between two complex buffers. Odd stage counts copy
    the finished axis into the buffer expected by the next stage. The finish
    kernel extracts nodal derivatives, conservative bounds and crest foam.
    """

    def __init__(self, n, length, seed):
        self.n = n
        self.length = length
        self.cell = length / n
        self.stages = int(round(math.log2(n)))
        assert n >= 2 and n & (n - 1) == 0, "FFT requires a power-of-two size"
        self.h0_re = ti.field(ti.f32, shape=(n, n))
        self.h0_im = ti.field(ti.f32, shape=(n, n))
        self.ha_re = ti.field(ti.f32, shape=(n, n))
        self.ha_im = ti.field(ti.f32, shape=(n, n))
        self.hb_re = ti.field(ti.f32, shape=(n, n))
        self.hb_im = ti.field(ti.f32, shape=(n, n))
        self.dxa_re = ti.field(ti.f32, shape=(n, n))
        self.dxa_im = ti.field(ti.f32, shape=(n, n))
        self.dxb_re = ti.field(ti.f32, shape=(n, n))
        self.dxb_im = ti.field(ti.f32, shape=(n, n))
        self.dza_re = ti.field(ti.f32, shape=(n, n))
        self.dza_im = ti.field(ti.f32, shape=(n, n))
        self.dzb_re = ti.field(ti.f32, shape=(n, n))
        self.dzb_im = ti.field(ti.f32, shape=(n, n))
        self.height = ti.field(ti.f32, shape=(n, n))
        self.grad = ti.Vector.field(2, ti.f32, shape=(n, n))
        self.foam = ti.field(ti.f32, shape=(n, n))
        self.h_sigma = ti.field(ti.f32, shape=())
        self.h_sigma[None] = 0.1
        self.bounds = ti.Vector.field(2, ti.f32, shape=())
        self.table = None

    def build_spectrum(self, wind_speed, wind_dir, wave_height, chop, swell, seed):
        """Three-component Phillips sea, normalized to the height.

        A swell component follows the wind (L = V^2/g, so its peak wavelength
        grows with wind speed), a mid component scales with it at an offset
        direction, and a wind-sea component sits at a fixed short scale.
        Superposing several incommensurate wave systems breaks up the regular
        single-train interference pattern a narrowband spectrum produces on a
        periodic tile.
        """
        n, length = self.n, self.length
        rng = np.random.default_rng(seed + n)
        freq = np.fft.fftfreq(n, d=length / n)
        kx = 2.0 * math.pi * freq
        KX, KZ = np.meshgrid(kx, kx, indexing="ij")
        k = np.hypot(KX, KZ)
        cap = np.exp(-(k / (2.0 * math.pi / 8.0)) ** 8)

        def component(length_scale, direction_deg, power):
            alignment = np.zeros_like(k)
            d = math.radians(direction_deg)
            np.divide(KX * math.cos(d) + KZ * math.sin(d), k,
                      out=alignment, where=k > 1e-9)
            shape = np.exp(-1.0 / np.maximum(k * length_scale, 1e-6) ** 2)
            phil = shape / np.maximum(k, 1e-6) ** 4 * np.maximum(alignment, 0.0) ** power
            phil[k < 1e-9] = 0.0
            return phil * cap

        speed = max(wind_speed, 0.1)
        swell_scale = speed ** 2 / GRAVITY
        power = 4.0 if swell else 2.0
        parts = [(component(swell_scale, wind_dir, power)
                  * np.exp(-(k / (2.0 * math.pi / 18.0)) ** 4), 1.0)]
        # Mid sea: a third of the swell scale, with reduced variance and broader direction.
        parts.append((component(min(max(0.3 * swell_scale, 2.5), 20.0),
                                wind_dir + 18.0, 2.0), 0.12))
        # Short wind sea: fixed 1.8 m scale peaks near 16 m wavelength, turned
        # 35 degrees off the swell, with a small share of total variance.
        parts.append((component(1.8, wind_dir + 35.0, 2.0),
                      0.06 * min(speed / 10.0, 1.0) ** 2))
        if swell:
            # A swell sea keeps only long components and a narrow spread.
            parts[0] = (parts[0][0] * np.exp(-(k / (2.0 * math.pi / 25.0)) ** 6), 1.0)
            parts[1] = (parts[1][0], 0.0)
            parts[2] = (parts[2][0], 0.0)
        total = sum(weight for _, weight in parts)
        norm = sum(weight * phil / max(float(phil.sum()), 1e-12)
                   for phil, weight in parts) / total
        target_var = (max(wave_height, 0.05) / 4.0) ** 2
        sigma2 = target_var * norm / 2.0
        self.h_sigma[None] = max(wave_height, 0.05) / 4.0
        h0_re = rng.standard_normal((n, n)) * np.sqrt(np.maximum(sigma2, 0.0) / 2.0)
        h0_im = rng.standard_normal((n, n)) * np.sqrt(np.maximum(sigma2, 0.0) / 2.0)
        self.h0_re.from_numpy(h0_re.astype(np.float32))
        self.h0_im.from_numpy(h0_im.astype(np.float32))
        self.table = {"k": k.astype(np.float32),
                      "omega": np.sqrt(GRAVITY * np.maximum(k, 0.0)).astype(np.float32),
                      "sigma2": sigma2.astype(np.float32),
                      "h0_re": h0_re.astype(np.float32),
                      "h0_im": h0_im.astype(np.float32)}

    def load_filtered(self, master):
        """Preserve the master modes and phases; only attenuate high k."""
        n = self.n
        index = (np.fft.fftfreq(n) * n).astype(int) % master.n
        table = {name: values[np.ix_(index, index)].copy()
                 for name, values in master.table.items()}
        cutoff = 0.7 * math.pi / self.cell
        taper = np.exp(-(table["k"] / cutoff) ** 8)
        taper[n // 2, :] = 0
        taper[:, n // 2] = 0
        table["h0_re"] *= taper
        table["h0_im"] *= taper
        table["sigma2"] *= taper ** 2
        self.h0_re.from_numpy(table["h0_re"])
        self.h0_im.from_numpy(table["h0_im"])
        self.h_sigma[None] = master.h_sigma[None]
        self.table = table

    @ti.kernel
    def copy_ab(self):
        for i, j in self.ha_re:
            self.hb_re[i, j], self.hb_im[i, j] = self.ha_re[i, j], self.ha_im[i, j]
            self.dxb_re[i, j], self.dxb_im[i, j] = self.dxa_re[i, j], self.dxa_im[i, j]
            self.dzb_re[i, j], self.dzb_im[i, j] = self.dza_re[i, j], self.dza_im[i, j]

    @ti.kernel
    def copy_ba(self):
        for i, j in self.ha_re:
            self.ha_re[i, j], self.ha_im[i, j] = self.hb_re[i, j], self.hb_im[i, j]
            self.dxa_re[i, j], self.dxa_im[i, j] = self.dxb_re[i, j], self.dxb_im[i, j]
            self.dza_re[i, j], self.dza_im[i, j] = self.dzb_re[i, j], self.dzb_im[i, j]

    @ti.func
    def _bit_reverse(self, i):
        r, t = 0, i
        for _ in ti.static(range(self.stages)):
            r = r * 2 + t % 2
            t = t // 2
        return r

    @ti.kernel
    def set_spectra(self, t: ti.f32, chop: ti.f32):
        """Time-evolve the spectrum: h(k,t) with deep-water dispersion."""
        n = self.n
        for i, j in self.h0_re:
            ni = ti.cast(i if i < n // 2 else i - n, ti.f32)
            nj = ti.cast(j if j < n // 2 else j - n, ti.f32)
            kx = 2.0 * math.pi * ni / self.length
            kz = 2.0 * math.pi * nj / self.length
            k = ti.sqrt(kx * kx + kz * kz)
            h_re, h_im = 0.0, 0.0
            dx_re, dx_im, dz_re, dz_im = 0.0, 0.0, 0.0, 0.0
            if k > 1e-6:
                phase = ti.sqrt(GRAVITY * k) * t
                c, s = ti.cos(phase), ti.sin(phase)
                mi, mj = (n - i) % n, (n - j) % n
                a_re, a_im = self.h0_re[i, j], self.h0_im[i, j]
                g_re, g_im = self.h0_re[mi, mj], self.h0_im[mi, mj]
                # H = h0(k) e^{i wt} + conj(h0(-k)) e^{-i wt}: real surface.
                h_re = a_re * c - a_im * s + g_re * c - g_im * s
                h_im = a_re * s + a_im * c - g_re * s - g_im * c
                # -i * k-hat * H gives the horizontal displacement spectra.
                # The gain grows super-linearly with chop so that storm seas
                # (long Phillips swells are otherwise too gentle to fold)
                # actually produce compressed crests for the foam test.
                gain = chop * (1.0 + 3.0 * chop)
                kxk, kzk = kx / k * gain, kz / k * gain
                dx_re, dx_im = kxk * h_im, -kxk * h_re
                dz_re, dz_im = kzk * h_im, -kzk * h_re
            self.ha_re[i, j] = h_re
            self.ha_im[i, j] = h_im
            self.dxa_re[i, j] = dx_re
            self.dxa_im[i, j] = dx_im
            self.dza_re[i, j] = dz_re
            self.dza_im[i, j] = dz_im

    @ti.kernel
    def rev0(self):
        for i, j in self.ha_re:
            r = self._bit_reverse(i)
            self.hb_re[i, j] = self.ha_re[r, j]
            self.hb_im[i, j] = self.ha_im[r, j]
            self.dxb_re[i, j] = self.dxa_re[r, j]
            self.dxb_im[i, j] = self.dxa_im[r, j]
            self.dzb_re[i, j] = self.dza_re[r, j]
            self.dzb_im[i, j] = self.dza_im[r, j]

    @ti.kernel
    def rev1(self):
        for i, j in self.hb_re:
            r = self._bit_reverse(j)
            self.ha_re[i, j] = self.hb_re[i, r]
            self.ha_im[i, j] = self.hb_im[i, r]
            self.dxa_re[i, j] = self.dxb_re[i, r]
            self.dxa_im[i, j] = self.dxb_im[i, r]
            self.dza_re[i, j] = self.dzb_re[i, r]
            self.dza_im[i, j] = self.dzb_im[i, r]

    @ti.kernel
    def st0_froma(self, half: ti.i32, size: ti.i32):
        for i, j in self.ha_re:
            if i % size < half:
                jx = i % half
                angle = 2.0 * math.pi * jx / size
                wr, wi = ti.cos(angle), ti.sin(angle)
                p = i + half
                ar, ai = self.ha_re[i, j], self.ha_im[i, j]
                br, bi = self.ha_re[p, j], self.ha_im[p, j]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.hb_re[i, j], self.hb_im[i, j] = ar + tr, ai + ti_
                self.hb_re[p, j], self.hb_im[p, j] = ar - tr, ai - ti_
                ar, ai = self.dxa_re[i, j], self.dxa_im[i, j]
                br, bi = self.dxa_re[p, j], self.dxa_im[p, j]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dxb_re[i, j], self.dxb_im[i, j] = ar + tr, ai + ti_
                self.dxb_re[p, j], self.dxb_im[p, j] = ar - tr, ai - ti_
                ar, ai = self.dza_re[i, j], self.dza_im[i, j]
                br, bi = self.dza_re[p, j], self.dza_im[p, j]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dzb_re[i, j], self.dzb_im[i, j] = ar + tr, ai + ti_
                self.dzb_re[p, j], self.dzb_im[p, j] = ar - tr, ai - ti_

    @ti.kernel
    def st0_fromb(self, half: ti.i32, size: ti.i32):
        for i, j in self.hb_re:
            if i % size < half:
                jx = i % half
                angle = 2.0 * math.pi * jx / size
                wr, wi = ti.cos(angle), ti.sin(angle)
                p = i + half
                ar, ai = self.hb_re[i, j], self.hb_im[i, j]
                br, bi = self.hb_re[p, j], self.hb_im[p, j]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.ha_re[i, j], self.ha_im[i, j] = ar + tr, ai + ti_
                self.ha_re[p, j], self.ha_im[p, j] = ar - tr, ai - ti_
                ar, ai = self.dxb_re[i, j], self.dxb_im[i, j]
                br, bi = self.dxb_re[p, j], self.dxb_im[p, j]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dxa_re[i, j], self.dxa_im[i, j] = ar + tr, ai + ti_
                self.dxa_re[p, j], self.dxa_im[p, j] = ar - tr, ai - ti_
                ar, ai = self.dzb_re[i, j], self.dzb_im[i, j]
                br, bi = self.dzb_re[p, j], self.dzb_im[p, j]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dza_re[i, j], self.dza_im[i, j] = ar + tr, ai + ti_
                self.dza_re[p, j], self.dza_im[p, j] = ar - tr, ai - ti_

    @ti.kernel
    def st1_froma(self, half: ti.i32, size: ti.i32):
        for i, j in self.ha_re:
            if j % size < half:
                jx = j % half
                angle = 2.0 * math.pi * jx / size
                wr, wi = ti.cos(angle), ti.sin(angle)
                p = j + half
                ar, ai = self.ha_re[i, j], self.ha_im[i, j]
                br, bi = self.ha_re[i, p], self.ha_im[i, p]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.hb_re[i, j], self.hb_im[i, j] = ar + tr, ai + ti_
                self.hb_re[i, p], self.hb_im[i, p] = ar - tr, ai - ti_
                ar, ai = self.dxa_re[i, j], self.dxa_im[i, j]
                br, bi = self.dxa_re[i, p], self.dxa_im[i, p]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dxb_re[i, j], self.dxb_im[i, j] = ar + tr, ai + ti_
                self.dxb_re[i, p], self.dxb_im[i, p] = ar - tr, ai - ti_
                ar, ai = self.dza_re[i, j], self.dza_im[i, j]
                br, bi = self.dza_re[i, p], self.dza_im[i, p]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dzb_re[i, j], self.dzb_im[i, j] = ar + tr, ai + ti_
                self.dzb_re[i, p], self.dzb_im[i, p] = ar - tr, ai - ti_

    @ti.kernel
    def st1_fromb(self, half: ti.i32, size: ti.i32):
        for i, j in self.hb_re:
            if j % size < half:
                jx = j % half
                angle = 2.0 * math.pi * jx / size
                wr, wi = ti.cos(angle), ti.sin(angle)
                p = j + half
                ar, ai = self.hb_re[i, j], self.hb_im[i, j]
                br, bi = self.hb_re[i, p], self.hb_im[i, p]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.ha_re[i, j], self.ha_im[i, j] = ar + tr, ai + ti_
                self.ha_re[i, p], self.ha_im[i, p] = ar - tr, ai - ti_
                ar, ai = self.dxb_re[i, j], self.dxb_im[i, j]
                br, bi = self.dxb_re[i, p], self.dxb_im[i, p]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dxa_re[i, j], self.dxa_im[i, j] = ar + tr, ai + ti_
                self.dxa_re[i, p], self.dxa_im[i, p] = ar - tr, ai - ti_
                ar, ai = self.dzb_re[i, j], self.dzb_im[i, j]
                br, bi = self.dzb_re[i, p], self.dzb_im[i, p]
                tr, ti_ = wr * br - wi * bi, wr * bi + wi * br
                self.dza_re[i, j], self.dza_im[i, j] = ar + tr, ai + ti_
                self.dza_re[i, p], self.dza_im[i, p] = ar - tr, ai - ti_

    @ti.kernel
    def finish(self):
        """Extract height, slope by central differences, and crest foam."""
        n = self.n
        self.bounds[None] = ti.Vector([1e6, -1e6])
        for i, j in self.ha_re:
            self.height[i, j] = self.ha_re[i, j]
            ti.atomic_min(self.bounds[None][0], self.ha_re[i, j])
            ti.atomic_max(self.bounds[None][1], self.ha_re[i, j])
            ip, im = (i + 1) % n, (i + n - 1) % n
            jp, jm = (j + 1) % n, (j + n - 1) % n
            # Differentiate ha_re (read-only here); reading self.height would
            # race with other threads writing neighbouring cells above.
            self.grad[i, j] = ti.Vector([
                (self.ha_re[ip, j] - self.ha_re[im, j]) / (2.0 * self.cell),
                (self.ha_re[i, jp] - self.ha_re[i, jm]) / (2.0 * self.cell)])
            # Folding crests: Jacobian determinant of the displacement map.
            dxx = (self.dxa_re[ip, j] - self.dxa_re[im, j]) / (2.0 * self.cell)
            dxz = (self.dxa_re[i, jp] - self.dxa_re[i, jm]) / (2.0 * self.cell)
            dzx = (self.dza_re[ip, j] - self.dza_re[im, j]) / (2.0 * self.cell)
            dzz = (self.dza_re[i, jp] - self.dza_re[i, jm]) / (2.0 * self.cell)
            det = (1.0 + dxx) * (1.0 + dzz) - dxz * dzx
            fold = ti.max(0.0, ti.min(1.0,
                (FOAM_DET_LOW - det) / (FOAM_DET_LOW - FOAM_DET_HIGH)))
            # Whitecaps live on wave crests: gate the fold signal by height so
            # troughs and moderate slopes stay dark and the patches ride on
            # actual waves instead of speckling the whole tile.
            crest = band_smooth(0.5 * self.h_sigma[None],
                                1.4 * self.h_sigma[None], self.height[i, j])
            self.foam[i, j] = fold * crest

    def update(self, t, chop):
        self.set_spectra(t, chop)
        self.rev0()
        for s in range(1, self.stages + 1):
            half, size = 1 << (s - 1), 1 << s
            if s % 2 == 1:
                self.st0_fromb(half, size)
            else:
                self.st0_froma(half, size)
        if self.stages % 2:
            self.copy_ab()
        self.rev1()
        for s in range(1, self.stages + 1):
            half, size = 1 << (s - 1), 1 << s
            if s % 2 == 1:
                self.st1_froma(half, size)
            else:
                self.st1_fromb(half, size)
        if self.stages % 2:
            self.copy_ba()
        self.finish()

    def debug_ifft(self):
        """Transform the current a-buffers in place (used by tests)."""
        self.rev0()
        for s in range(1, self.stages + 1):
            half, size = 1 << (s - 1), 1 << s
            if s % 2 == 1:
                self.st0_fromb(half, size)
            else:
                self.st0_froma(half, size)
        if self.stages % 2:
            self.copy_ab()
        self.rev1()
        for s in range(1, self.stages + 1):
            half, size = 1 << (s - 1), 1 << s
            if s % 2 == 1:
                self.st1_froma(half, size)
            else:
                self.st1_fromb(half, size)

        if self.stages % 2:
            self.copy_ba()

    @ti.func
    def _uv(self, x, z):
        n = self.n
        u = (x / self.length + 0.5) * n
        v = (z / self.length + 0.5) * n
        u = u - ti.floor(u / n) * n
        v = v - ti.floor(v / n) * n
        return u, v

    @ti.func
    def cubic_weights(self, t):
        t2, t3 = t * t, t * t * t
        weights = ti.Vector([-0.5 * t + t2 - 0.5 * t3,
                             1 - 2.5 * t2 + 1.5 * t3,
                             0.5 * t + 2 * t2 - 1.5 * t3,
                             -0.5 * t2 + 0.5 * t3])
        derivative = ti.Vector([-0.5 + 2 * t - 1.5 * t2,
                                -5 * t + 4.5 * t2,
                                0.5 + 4 * t - 4.5 * t2,
                                -t + 1.5 * t2])
        return weights, derivative

    @ti.func
    def sample_surface(self, x, z):
        u, v = self._uv(x, z)
        i, j = ti.cast(u, ti.i32), ti.cast(v, ti.i32)
        wx, dx = self.cubic_weights(u - i)
        wz, dz = self.cubic_weights(v - j)
        result = ti.Vector([0.0, 0.0, 0.0])
        for a, b in ti.static(ti.ndrange(4, 4)):
            h = self.height[(i + a - 1 + self.n) % self.n, (j + b - 1 + self.n) % self.n]
            result += h * ti.Vector([wx[a] * wz[b], dx[a] * wz[b] / self.cell,
                                     wx[a] * dz[b] / self.cell])
        return result

    @ti.func
    def sample_height(self, x, z):
        return self.sample_surface(x, z).x

    @ti.func
    def sample_grad(self, x, z):
        surface = self.sample_surface(x, z)
        return ti.Vector([surface.y, surface.z])

    @ti.func
    def sample_foam(self, x, z):
        n = self.n
        u, v = self._uv(x, z)
        i0, j0 = ti.cast(u, ti.i32), ti.cast(v, ti.i32)
        fu, fv = u - i0, v - j0
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        return ((self.foam[i0, j0] * (1.0 - fu) + self.foam[i1, j0] * fu) * (1.0 - fv)
                + (self.foam[i0, j1] * (1.0 - fu) + self.foam[i1, j1] * fu) * fv)


@ti.data_oriented
class FFTOcean:
    """Phillips-spectrum ocean tile: three prefiltering resolution levels."""

    def __init__(self, seed=23):
        self.levels = [FFTLevel(n, TILE_LENGTH, seed) for n in LEVEL_SIZES]
        self.amp_scale = ti.field(ti.f32, shape=())
        self.chop = 0.6
        self.seed = seed
        self.rebuild(8.0, 35.0, 1.3, 0.6, False)
        self.update(0.0)

    def rebuild(self, wind_speed, wind_dir, wave_height, chop, swell):
        master = self.levels[0]
        master.build_spectrum(wind_speed, wind_dir, wave_height, chop, swell, self.seed)
        for level in self.levels[1:]:
            level.load_filtered(master)
        self.chop = float(chop)
        self.update(0.0)
        height = self.levels[0].height.to_numpy()
        self.amp_scale[None] = float(max(0.05, 4.0 * height.std()))

    def update(self, t):
        for level in self.levels:
            level.update(t, self.chop)

    @ti.func
    def sample_height(self, x, z):
        return self.levels[0].sample_height(x, z)

    @ti.func
    def sample_grad_lod(self, x, z, footprint=0.0):
        g = self.levels[0].sample_grad(x, z)
        w1 = band_smooth(self.levels[0].cell, self.levels[1].cell, footprint)
        w2 = band_smooth(self.levels[1].cell, self.levels[2].cell, footprint)
        g = g * (1.0 - w1) + self.levels[1].sample_grad(x, z) * w1
        g = g * (1.0 - w2) + self.levels[2].sample_grad(x, z) * w2
        return g

    @ti.func
    def sample_foam_lod(self, x, z, footprint=0.0):
        f = self.levels[0].sample_foam(x, z)
        w1 = band_smooth(self.levels[0].cell, self.levels[1].cell, footprint)
        w2 = band_smooth(self.levels[1].cell, self.levels[2].cell, footprint)
        f = f * (1.0 - w1) + self.levels[1].sample_foam(x, z) * w1
        f = f * (1.0 - w2) + self.levels[2].sample_foam(x, z) * w2
        return f


@ti.data_oriented
class WindOceanRenderer(SunsetOceanRenderer):
    """Demo 03 renderer whose sea surface comes from the FFT tile."""

    def __init__(self, width, height, samples=4):
        super().__init__(width, height, samples)
        self.ocean = FFTOcean(seed=23)
        self.water_controls[None] = [0.05, 1.0, 1.0, 4.0]
        self._fft_clock = None

    def refresh_ocean(self, wind_speed, wind_dir, wave_height, chop, swell):
        self.ocean.rebuild(wind_speed, wind_dir, wave_height, chop, swell)
        self._fft_clock = None

    @ti.func
    def wave_amplitude(self):
        return self.ocean.amp_scale[None]

    @ti.func
    def wave_height(self, x, z, clock):
        return self.ocean.sample_height(x, z)

    @ti.func
    def wave_bounds(self):
        bounds = self.ocean.levels[0].bounds[None]
        # Tensor Catmull-Rom has sum(abs(weights)) <= 1.5625: include
        # possible interpolation overshoot, not just sampled node extrema.
        padding = (bounds.y - bounds.x) * 0.28125 + 0.002
        return ti.Vector([bounds.x - padding, bounds.y + padding])

    @ti.func
    def wave_surface(self, x, z, clock):
        surface = self.ocean.levels[0].sample_surface(x, z)
        return surface.x, ti.Vector([-surface.y, 1.0, -surface.z]).normalized()

    @ti.func
    def shading_normal(self, p, clock, footprint, geometric_normal):
        grad = self.ocean.sample_grad_lod(p.x, p.z, footprint)
        coarse = self.ocean.sample_grad_lod(p.x, p.z, ti.max(footprint, self.ocean.levels[1].cell))
        grad = coarse + (grad - coarse) * self.water_controls[None].y
        return ti.Vector([-grad.x, 1.0, -grad.y]).normalized()

    @ti.func
    def seabed_depth(self):
        return -80.0

    @ti.func
    def floor_material(self, p, clock, amplitude):
        return ti.Vector([0.10, 0.11, 0.10]), ti.Vector([0.0, 1.0, 0.0])

    @ti.func
    def water_depth_limit(self):
        return 120.0

    @ti.func
    def water_body(self, color, thickness, clarity, wave_y, normal, lighting):
        transmission = ti.exp(-ti.Vector([0.34, 0.12, 0.065]) * thickness / clarity)
        # Smooth orientation-dependent illumination, not a height tint that
        # paints FFT cells onto the surface. Deep water hides the seabed.
        illumination = 0.75 + 0.25 * ti.max(0.0, normal.dot(self.sun_direction(lighting)))
        bulk = ti.Vector([0.014, 0.060, 0.085]) * illumination
        return color * transmission + bulk * (1 - transmission)

    @ti.func
    def foam_filtered(self, p, clock, lighting, footprint):
        amount = self.ocean.sample_foam_lod(p.x, p.z, footprint) * self.water_controls[None].z
        return ti.Vector([0.93, 0.90, 0.84]) * (0.55 * amount)

    @ti.func
    def foam(self, p, clock, lighting):
        return self.foam_filtered(p, clock, lighting, 0.0)

    def draw(self, camera, clock, clarity=1.4, lighting=0.0, exposure=1.05):
        if clock != self._fft_clock:
            self.ocean.update(clock)
            self._fft_clock = clock
        super().draw(camera, clock, clarity, lighting, exposure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 05: FFT wind-driven ocean")
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
    parser.add_argument("--sea", choices=("auto", "breeze", "swell", "gale"), default="auto",
                        help="Sea preset; auto uses the wind/height/chop values below")
    parser.add_argument("--wind", type=float, default=8.0, help="Wind speed in m/s")
    parser.add_argument("--wind-dir", type=float, default=35.0, help="Wind direction in degrees")
    parser.add_argument("--wave-height", type=float, default=1.3, help="Significant wave height in metres")
    parser.add_argument("--chop", type=float, default=0.6, help="Horizontal crest displacement, 0..1")
    parser.add_argument("--swell", action="store_true", help="Narrow long-component swell spectrum")
    parser.add_argument("--preset", choices=("overview", "waterline", "top"), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_05.png"))
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
    if not math.isfinite(args.wave_height) or not 0.05 <= args.wave_height <= 3.5:
        parser.error("wave-height must be finite and between 0.05 and 3.5")
    if not math.isfinite(args.chop) or not 0.0 <= args.chop <= 1.0:
        parser.error("chop must be finite and between 0 and 1")
    return args


def sea_parameters(args):
    if args.sea == "auto":
        return args.wind, args.wind_dir, args.wave_height, args.chop, args.swell
    wind, direction, height, chop, _ = SEA_PRESETS[args.sea]
    return wind, direction, height, chop, args.swell or SEA_PRESETS[args.sea][4]


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = WindOceanRenderer(args.width, args.height, args.samples)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 1.0, 1.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = OrbitCamera()
    apply_preset(camera, args.preset)
    sea = list(sea_parameters(args))
    speed = 1.0
    lighting = 0.0 if args.lighting == "sunset" else 1.0
    clarity, exposure = 1.4, 1.05
    clock = max(0.0, args.time)
    renderer.refresh_ocean(*sea)

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
    window = ti.ui.Window("05 / Wind Sea", (args.width, args.height), vsync=True)
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
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 1.0, 1.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                sea = list(sea_parameters(args))
                renderer.refresh_ocean(*sea)
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
                controls[2] = gui.slider_float("Crest foam", controls[2], 0.0, 2.0)
                controls[3] = 4 if gui.checkbox("4x reflections", controls[3] == 4) else 1
                renderer.water_controls[None] = controls
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
            with gui.sub_window("WIND SEA / 05", 0.02, 0.025, 0.30, 0.78):
                gui.text("FFT spectrum / dispersion / chop")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                if gui.button("Single step [N]"):
                    step = True
                wind_speed = gui.slider_float("Wind speed", sea[0], 0.0, 24.0)
                wind_dir = gui.slider_float("Wind direction", sea[1], 0.0, 360.0)
                wave_height = gui.slider_float("Wave height", sea[2], 0.05, 3.5)
                chop = gui.slider_float("Chop", sea[3], 0.0, 1.0)
                swell = gui.checkbox("Swell spectrum", sea[4])
                sea = [wind_speed, wind_dir, wave_height, chop, swell]
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
                gui.text("Breeze / swell / gale: --sea preset")
                gui.text("Drag LMB orbit / RMB pan")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        if paused:
            if step:
                clock += 1.0 / 60.0
        else:
            clock += dt * speed
        # The spectrum is rebuilt on the host only when its inputs move.
        current = tuple(float(v) for v in sea)
        if current != getattr(renderer, "_sea_key", None):
            renderer.refresh_ocean(*current)
            renderer._sea_key = current
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
