"""Night-specific display finishing: film grain on top of the shared
PostProcess bloom and tonemap chain.

The shared PostProcess in water/finish.py stays untouched; this subclass
reuses its kernels verbatim and only adds a grain pass after compose.
"""

import taichi as ti

from water.finish import PostProcess

__all__ = ["GRAIN_SEED", "NightPostProcess"]

# Fixed seed for the grain hash; distinct from the simulation seeds.
GRAIN_SEED = 23


@ti.func
def hash01(x: ti.i32, y: ti.i32, frame: ti.i32, seed: ti.i32) -> ti.f32:
    """Deterministic unsigned-integer hash mapped to [0, 1)."""
    h = (ti.u32(x) * ti.u32(374761393) + ti.u32(y) * ti.u32(668265263)
         + ti.u32(frame) * ti.u32(1274126177) + ti.u32(seed) * ti.u32(2246822519))
    h = (h ^ (h >> ti.u32(13))) * ti.u32(1274126177)
    h = h ^ (h >> ti.u32(16))
    return ti.cast(h & ti.u32(0xFFFFFF), ti.f32) * (1.0 / 16777216.0)


@ti.data_oriented
class NightPostProcess(PostProcess):
    """PostProcess plus low-intensity film grain keyed on a frame counter.

    The counter advances once per distinct 1/60 s clock quantum, so drawing
    the same clock twice (tests, paused frames) reproduces the same grain
    bit for bit, while an animating clock jitters the noise every frame.
    """

    def __init__(self, width, height, seed=GRAIN_SEED):
        super().__init__(width, height)
        self.seed = seed
        self.frame = ti.field(ti.i32, shape=())
        self.frame[None] = 0
        self._last_quantized = None
        self._grain_prepared = False

    def advance_frame(self, clock):
        """Advance the grain frame counter when the clock reaches a new frame."""
        quantized = int(round(clock * 60.0))
        if quantized != self._last_quantized:
            self.frame[None] = self.frame[None] + 1
            self._last_quantized = quantized

    @ti.kernel
    def add_grain(self, strength: ti.f32):
        # One shared value per pixel keeps the grain neutral in hue; the
        # zero mean keeps frame brightness stable while masking banding.
        for x, y in self.image:
            noise = hash01(x, y, self.frame[None], self.seed) - 0.5
            self.image[x, y] = ti.max(0.0, self.image[x, y] + noise * strength)

    def apply(self, exposure=1.0, strength=0.1, threshold=1.0, vignette=0.15,
              grain=0.0):
        super().apply(exposure, strength, threshold, vignette)
        # Compile the grain pass on the first call even when it is off, so
        # toggling it later does not trigger a mid-session JIT pause.
        if grain > 0.0 or not self._grain_prepared:
            self.add_grain(grain)
            self._grain_prepared = True
