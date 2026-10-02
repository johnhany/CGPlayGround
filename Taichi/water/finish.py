"""Visible GGX reflection sampling and linear HDR bloom / display finishing."""
import math

import taichi as ti


@ti.func
def water_reflection(view, normal, roughness, sample, count):
    """GGX visible-normal sample; return outgoing ray and bounded F*Smith weight."""
    reference = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(normal.y) > 0.95:
        reference = ti.Vector([1.0, 0.0, 0.0])
    tangent = reference.cross(normal).normalized()
    bitangent = normal.cross(tangent)
    half_vector = normal
    alpha = ti.max(0.001, roughness * roughness)
    if roughness > 0.001:
        v = ti.Vector([alpha * view.dot(tangent), alpha * view.dot(bitangent), view.dot(normal)]).normalized()
        t1 = ti.Vector([1.0, 0.0, 0.0])
        length2 = v.x * v.x + v.y * v.y
        if length2 > 1e-8:
            t1 = ti.Vector([-v.y, v.x, 0.0]) / ti.sqrt(length2)
        t2 = v.cross(t1)
        u = (sample + 0.5) / count
        # Four-point radical-inverse sequence, fixed across frames.
        phi = 2.0 * math.pi * ((sample % 2) * 0.5 + (sample // 2) * 0.25) + 0.41
        disk_x, disk_y = ti.sqrt(u) * ti.cos(phi), ti.sqrt(u) * ti.sin(phi)
        blend = 0.5 * (1.0 + v.z)
        disk_y = (1.0 - blend) * ti.sqrt(ti.max(0.0, 1.0 - disk_x * disk_x)) + blend * disk_y
        nh = disk_x * t1 + disk_y * t2 + ti.sqrt(ti.max(0.0, 1.0 - disk_x * disk_x - disk_y * disk_y)) * v
        local = ti.Vector([alpha * nh.x, alpha * nh.y, ti.max(0.0, nh.z)]).normalized()
        half_vector = (tangent * local.x + bitangent * local.y + normal * local.z).normalized()
    outgoing = (2.0 * view.dot(half_vector) * half_vector - view).normalized()
    nl = ti.max(0.0, outgoing.dot(normal))
    f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
    fresnel = f0 + (1.0 - f0) * (1.0 - ti.max(0.0, view.dot(half_vector))) ** 5
    weight = 0.0
    if nl > 0.0:
        smith = 2.0 * nl / (nl + ti.sqrt(alpha * alpha + (1.0 - alpha * alpha) * nl * nl))
        weight = fresnel * smith
    return outgoing, weight


@ti.data_oriented
class PostProcess:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.hdr = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.image = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.bloom_width, self.bloom_height = (width + 3) // 4, (height + 3) // 4
        shape = (self.bloom_width, self.bloom_height)
        self.bright = ti.Vector.field(3, ti.f32, shape=shape)
        self.horizontal = ti.Vector.field(3, ti.f32, shape=shape)
        self.bloom = ti.Vector.field(3, ti.f32, shape=shape)
        self._prepared = False

    @ti.kernel
    def extract(self, exposure: ti.f32, threshold: ti.f32):
        for x, y in self.bright:
            color = ti.Vector([0.0, 0.0, 0.0])
            # Threshold before averaging so small glints survive downsampling.
            for a, b in ti.static(ti.ndrange(4, 4)):
                pixel = self.hdr[ti.min(4 * x + a, self.width - 1), ti.min(4 * y + b, self.height - 1)] * exposure
                brightness = pixel.max()
                knee = threshold * 0.5
                soft = ti.min(2.0 * knee, ti.max(0.0, brightness - threshold + knee))
                soft = soft * soft / ti.max(4.0 * knee, 1e-5)
                contribution = ti.max(brightness - threshold, soft) / ti.max(brightness, 1e-5)
                color += pixel * contribution / 16.0
            self.bright[x, y] = color

    @ti.kernel
    def blur(self, source: ti.template(), target: ti.template(), axis: ti.i32):
        for x, y in target:
            color, total = ti.Vector([0.0, 0.0, 0.0]), 0.0
            for offset in ti.static(range(-4, 5)):
                weight = ti.exp(-0.5 * offset * offset / 4.0)
                px = ti.min(self.bloom_width - 1, ti.max(0, x + offset * (1 - axis)))
                py = ti.min(self.bloom_height - 1, ti.max(0, y + offset * axis))
                color += source[px, py] * weight
                total += weight
            target[x, y] = color / total

    @ti.kernel
    def compose(self, exposure: ti.f32, bloom_strength: ti.f32, vignette: ti.f32):
        for x, y in self.image:
            color = ti.max(0.0, self.hdr[x, y]) * exposure
            if bloom_strength > 0.0:
                u = ti.max(0.0, ti.min(self.bloom_width - 1.0, (x + 0.5) / 4.0 - 0.5))
                v = ti.max(0.0, ti.min(self.bloom_height - 1.0, (y + 0.5) / 4.0 - 0.5))
                ix, iy = ti.cast(u, ti.i32), ti.cast(v, ti.i32)
                fx, fy = u - ix, v - iy
                for a, b in ti.static(ti.ndrange(2, 2)):
                    weight = (fx if a else 1.0 - fx) * (fy if b else 1.0 - fy)
                    color += self.bloom[ti.min(ix + a, self.bloom_width - 1), ti.min(iy + b, self.bloom_height - 1)] * weight * bloom_strength
            color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14)
            color = ti.min(1.0, ti.max(0.0, color)) ** (1.0 / 2.2)
            px = (2.0 * (x + 0.5) / self.width - 1.0) * self.width / self.height
            py = 2.0 * (y + 0.5) / self.height - 1.0
            self.image[x, y] = color * ti.max(0.0, 1.0 - vignette * (px * px + py * py) / 3.0)

    def apply(self, exposure=1.05, strength=0.08, threshold=1.0, vignette=0.12):
        # Compile every optional pass before the first window appears.
        if strength > 0.0 or not self._prepared:
            self.extract(exposure, threshold)
            self.blur(self.bright, self.horizontal, 0)
            self.blur(self.horizontal, self.bloom, 1)
        self.compose(exposure, strength, vignette)
        self._prepared = True
