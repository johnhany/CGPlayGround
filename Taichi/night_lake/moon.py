"""Production moon direction for the night lake scene.

The shared moon in water/pbr.py sits 26 degrees above the horizon, which
lands above the frame top once the lake view pitches down enough to hold
the shore and the far bank. The production frame therefore keeps its own
lower moon, a few degrees above the treeline, so the disk, its glitter
path, and the water reflections all stay inside one composition.

The baked night environment from water/pbr.py is built around the shared
direction, so scene.py re-bakes the night variant of the irradiance,
prefiltered, and sky maps against this local direction at start-up. The
day and sunset variants are used unchanged.
"""

import math

import numpy as np

__all__ = ["MOON_AZIMUTH", "MOON_ELEVATION", "MOON_DIRECTION",
           "MOON_TINT", "MOON_DIRECT_INTENSITY", "MOON_IBL_BOOST",
           "MOON_GLINT_INTENSITY", "MOON_GLOW_WIDE_POWER", "MOON_GLOW_WIDE",
           "MOON_GLOW_TIGHT_POWER", "MOON_GLOW_TIGHT",
           "MOON_DISK_BASE", "MOON_DISK_PHASE", "MOON_DISK_LIMB",
           "MOON_ENV_GLOW_POWER", "MOON_ENV_GLOW", "night_environment"]

# Ahead and to the left, matching the shared azimuth, but only a few
# degrees above the horizon so the disk clears the far treeline yet stays
# inside the pitched-down frame.
MOON_AZIMUTH = -0.55
MOON_ELEVATION = 0.13
MOON_DIRECTION = np.array([
    math.sin(MOON_AZIMUTH) * math.cos(MOON_ELEVATION),
    math.sin(MOON_ELEVATION),
    math.cos(MOON_AZIMUTH) * math.cos(MOON_ELEVATION),
], dtype=np.float32)

# One derivation chain for every moon-lit term in the frame, so the disk,
# the sky glow, the directional light, and the water glitter stay in fixed
# ratio to each other.
MOON_TINT = (0.95, 0.97, 1.0)
# Directional diffuse scale (terrain runs at full strength).
MOON_DIRECT_INTENSITY = 1.4
# Foliage and bark take the directional light at these fractions of the
# terrain strength.
MOON_FOLIAGE_DIRECT = 0.7
MOON_BARK_DIRECT = 0.65
# Irradiance-map multiplier applied to every ambient diffuse term.
MOON_IBL_BOOST = 3.0
# GGX moon glint on the water.
MOON_GLINT_INTENSITY = 3.0
# Atmospheric glow cosine lobes: the wide lobe shapes the glitter column,
# the tight lobe adds the bright core.
MOON_GLOW_WIDE_POWER = 10.0
MOON_GLOW_WIDE = 0.30
MOON_GLOW_TIGHT_POWER = 96.0
MOON_GLOW_TIGHT = 1.5
# Visible disk radiance in sky.py: base plus a phase term, with limb
# darkening.
MOON_DISK_BASE = 2.0
MOON_DISK_PHASE = 5.0
MOON_DISK_LIMB = 0.25
# The same glow as baked into the night environment map (tight power, low
# absolute scale: the bake is an ambient term, not a light).
MOON_ENV_GLOW_POWER = 24.0
MOON_ENV_GLOW = (0.030, 0.033, 0.040)


def night_environment(directions):
    """Deep night sky bake, identical in structure to the one in
    water/pbr.py but centred on the production moon direction."""
    y = directions[..., 1]
    azimuth = np.arctan2(directions[..., 0], directions[..., 2])
    t = np.clip(y, 0, 1)[..., None] ** 0.55
    horizon = np.array([0.030, 0.042, 0.066])
    zenith = np.array([0.006, 0.010, 0.026])
    color = horizon * (1 - t) + zenith * t
    cloud = np.sin(azimuth * 3.0 + y * 8.0) * np.sin(azimuth * 5.0 - y * 4.0)
    cloud = np.clip((cloud - 0.12) * 1.7, 0, 1) * np.clip(y * 6, 0, 1)
    cloud *= np.clip((0.85 - y) * 3, 0, 1)
    cloud_color = np.array([0.028, 0.036, 0.058])
    color = color * (1 - cloud[..., None] * 0.6) + cloud_color * cloud[..., None] * 0.6
    proximity = np.clip(directions @ MOON_DIRECTION, 0, 1) ** MOON_ENV_GLOW_POWER
    color += proximity[..., None] * cloud[..., None] * np.array([0.06, 0.065, 0.07])
    glow = np.clip(directions @ MOON_DIRECTION, 0, 1) ** MOON_ENV_GLOW_POWER
    color += glow[..., None] * np.array(MOON_ENV_GLOW)
    ridge = 0.028 + 0.018 * np.sin(azimuth * 5) + 0.01 * np.sin(azimuth * 13)
    hill = (y >= 0) & (y < ridge)
    color = np.where(hill[..., None], np.array([0.008, 0.011, 0.016]), color)
    ground = np.broadcast_to([0.005, 0.006, 0.009], color.shape)
    return np.where((y < 0)[..., None], ground, color).astype(np.float32)
