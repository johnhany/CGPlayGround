"""Courtyard pool: analytical ripples and a Taichi water ray renderer.

Run: uv run python water/demo_01.py
Export: uv run python water/demo_01.py --headless --output output/pool.png
"""

import argparse
import math
from pathlib import Path
import time

import numpy as np
import taichi as ti

if __package__:
    from .pbr import fresnel_schlick, ggx_brdf
    from .finish import water_reflection
    from .scene import CourtyardScene, OrbitCamera
else:
    from pbr import fresnel_schlick, ggx_brdf
    from finish import water_reflection
    from scene import CourtyardScene, OrbitCamera

__all__ = ["OrbitCamera", "PoolRenderer", "main", "parse_args"]


@ti.data_oriented
class PoolRenderer(CourtyardScene):
    """Single-bounce reflection/refraction over an analytical height field.

    Spatial samples are averaged in linear radiance before tone mapping;
    no temporal history. The static scene and PBR shading come from
    CourtyardScene; this class adds the analytical wave field and the
    Newton water intersection.
    """

    def __init__(self, width, height, samples=4):
        if samples not in (1, 4):
            raise ValueError("samples must be 1 or 4")
        super().__init__(width, height)
        self.samples = samples
        self._caustic_key = None

    @ti.func
    def water(self, x, z, clock, amplitude, frequency):
        # Return (height, dh/dx, dh/dz), including the edge fade derivative.
        wave = ti.Vector([0.0, 0.0, 0.0])
        for source in ti.static(range(2)):
            center = ti.Vector([-0.85, 0.25])
            strength, speed = 1.0, 1.30
            if ti.static(source == 1):
                center = ti.Vector([1.30, -0.65])
                strength, speed = 0.56, 1.07
            delta = ti.Vector([x, z]) - center
            r = ti.sqrt(delta.dot(delta) + 0.06)
            k = frequency * (1.0 + 0.21 * source)
            phase = k * r - speed * clock * k
            envelope = amplitude * strength * ti.exp(-0.24 * r)
            height = envelope * ti.sin(phase)
            derivative = envelope * (k * ti.cos(phase) - 0.24 * ti.sin(phase))
            wave += ti.Vector([height, derivative * delta.x / r, derivative * delta.y / r])
        # Small directional motion makes the quiet regions less static.
        phase = 2.1 * x + 1.7 * z - 0.8 * clock
        wave += amplitude * ti.Vector([0.12 * ti.sin(phase), 0.252 * ti.cos(phase), 0.204 * ti.cos(phase)])
        # Resolved medium waves join the height field and analytical derivatives.
        for band in ti.static(range(2)):
            kx, kz, strength, speed = 7.5, 5.2, 0.055, 1.15
            if ti.static(band == 1):
                kx, kz, strength, speed = -15.3, 9.1, 0.025, 1.7
            phase = kx * x + kz * z - speed * clock
            scale = amplitude * strength * self.water_controls[None].y
            wave += scale * ti.Vector([ti.sin(phase), kx * ti.cos(phase), kz * ti.cos(phase)])
        ux = ti.max(0.0, ti.min(1.0, (3.2 - ti.abs(x)) / 0.45))
        uz = ti.max(0.0, ti.min(1.0, (2.15 - ti.abs(z)) / 0.45))
        fx, fz = ux * ux * (3.0 - 2.0 * ux), uz * uz * (3.0 - 2.0 * uz)
        dfx = -6.0 * ux * (1.0 - ux) * ti.math.sign(x) / 0.45
        dfz = -6.0 * uz * (1.0 - uz) * ti.math.sign(z) / 0.45
        return ti.Vector([wave.x * fx * fz,
                          wave.y * fx * fz + wave.x * dfx * fz,
                          wave.z * fx * fz + wave.x * fx * dfz])

    @ti.func
    def detail_gradient(self, x, z, clock, amplitude, footprint):
        gradient = ti.Vector([0.0, 0.0])
        edge = self.smooth(0.0, 0.45, 3.2 - ti.abs(x)) * self.smooth(0.0, 0.45, 2.15 - ti.abs(z))
        for band in ti.static(range(2)):
            kx, kz, strength, speed = 32.0, -24.0, 0.018, 2.2
            if ti.static(band == 1):
                kx, kz, strength, speed = 59.0, 43.0, 0.009, 3.1
            wavelength = 2.0 * math.pi / ti.sqrt(kx * kx + kz * kz)
            fade = 1.0 - self.smooth(wavelength * 0.12, wavelength * 0.5, footprint)
            phase = kx * x + kz * z - speed * clock
            gradient += amplitude * strength * ti.cos(phase) * ti.Vector([kx, kz]) * fade
        return gradient * edge * self.water_controls[None].y

    @ti.kernel
    def trace_caustics(self, clock: ti.f32, amplitude: ti.f32, frequency: ti.f32, lighting: ti.f32, shadows: ti.i32):
        for x, y in self.caustic_raw:
            self.caustic_raw[x, y] = 0.0
        for x, y in ti.ndrange(256, 192):
            px, pz = (x + 0.5) / 256 * 6.4 - 3.2, (y + 0.5) / 192 * 4.3 - 2.15
            wave = self.water(px, pz, clock, amplitude, frequency)
            fine = self.detail_gradient(px, pz, clock, amplitude, 6.4 / 128)
            normal = ti.Vector([-wave.y - fine.x, 1.0, -wave.z - fine.y]).normalized()
            p = ti.Vector([px, wave.x, pz])
            sun = self.sun_direction(lighting)
            cosine = ti.max(0.0, sun.dot(normal))
            eta = 1.0 / 1.333
            ray = (-eta * sun + (eta * cosine - ti.sqrt(ti.max(0.0, 1.0 - eta * eta * (1.0 - cosine * cosine)))) * normal).normalized()
            distance = (-0.68 - p.y) / ray.y
            target = p + distance * ray
            visible = 1
            if shadows == 1:
                visible = ti.cast(self.occlusion_distance(p + normal * 0.006, sun, 18.0) >= 18.0, ti.i32)
            if visible and cosine > 0.0 and distance > 0.0 and ti.abs(target.x) < 3.2 and ti.abs(target.z) < 2.15:
                f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
                transmitted = 1.0 - fresnel_schlick(cosine, f0)
                # Source area / receiver cell area = 1/4. Correct tilted flux.
                energy = transmitted * cosine / ti.max(0.001, normal.y * sun.y) * 0.25
                u, v = (target.x + 3.2) / 6.4 * 128 - 0.5, (target.z + 2.15) / 4.3 * 96 - 0.5
                ix, iy = ti.cast(ti.floor(u), ti.i32), ti.cast(ti.floor(v), ti.i32)
                fx, fy = u - ti.floor(u), v - ti.floor(v)
                for a, b in ti.static(ti.ndrange(2, 2)):
                    if 0 <= ix + a < 128 and 0 <= iy + b < 96:
                        weight = (fx if a else 1.0 - fx) * (fy if b else 1.0 - fy)
                        ti.atomic_add(self.caustic_raw[ix + a, iy + b], energy * weight)

    @ti.func
    def water_hit(self, origin, direction, clock, amplitude, frequency):
        t = 1e6
        normal = ti.Vector([0.0, 1.0, 0.0])
        if direction.y < -0.0001:
            candidate = -origin.y / direction.y
            for _ in ti.static(range(6)):
                p = origin + direction * candidate
                wave = self.water(p.x, p.z, clock, amplitude, frequency)
                derivative = direction.y - wave.y * direction.x - wave.z * direction.z
                if ti.abs(derivative) > 0.005:
                    step = (p.y - wave.x) / derivative
                    candidate -= ti.max(-0.35, ti.min(0.35, step))
            p = origin + direction * candidate
            wave = self.water(p.x, p.z, clock, amplitude, frequency)
            if candidate > 0.001 and ti.abs(p.x) < 3.2 and ti.abs(p.z) < 2.15 and ti.abs(p.y - wave.x) < 0.004:
                t = candidate
                normal = ti.Vector([-wave.y, 1.0, -wave.z]).normalized()
        return t, normal

    @ti.func
    def radiance(self, origin, direction, clock, amplitude, frequency, clarity, lighting, pixel_angle):
        opaque_t, opaque_normal, opaque_base, opaque_material = self.geometry(origin, direction)
        water_t, normal = self.water_hit(origin, direction, clock, amplitude, frequency)
        has_water = water_t < opaque_t
        p = origin + direction * water_t
        if has_water:
            footprint = water_t * pixel_angle / ti.max(0.15, ti.abs(direction.y))
            fine = self.detail_gradient(p.x, p.z, clock, amplitude, footprint)
            normal = (normal + ti.Vector([-fine.x, 0.0, -fine.y])).normalized()
        cosine = ti.max(0.0, ti.min(1.0, -direction.dot(normal)))
        eta = 1.0 / 1.333
        root = ti.sqrt(ti.max(0.0, 1.0 - eta * eta * (1.0 - cosine * cosine)))
        refracted = (eta * direction + (eta * cosine - root) * normal).normalized()
        color, reflection = ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0])
        thickness = 0.0
        reflection_count = 0
        if has_water:
            reflection_count = ti.cast(self.water_controls[None].w, ti.i32)
            if self.water_controls[None].x <= 0.001:
                reflection_count = 1
        # Share one shading body across primary and every water secondary ray.
        for bounce in range(1 + reflection_count):
            ray_origin, ray_direction = origin, direction
            t, n, base, material = opaque_t, opaque_normal, opaque_base, opaque_material
            weight = 1.0
            if has_water:
                if bounce < reflection_count:
                    ray_direction, weight = water_reflection(-direction, normal, self.water_controls[None].x, bounce, reflection_count)
                    ray_origin = p + normal * 0.004
                else:
                    ray_origin, ray_direction = p - normal * 0.004, refracted
                t, n, base, material = self.geometry(ray_origin, ray_direction)
            shaded = ti.Vector([0.0, 0.0, 0.0])
            if weight > 0.0 or bounce == reflection_count:
                shaded, _ = self.shade_surface(ray_origin, ray_direction, t, n, base, material, clock, lighting, amplitude, ti.cast(not has_water, ti.i32))
            color = shaded
            if has_water and bounce < reflection_count:
                reflection += shaded * weight / reflection_count
            elif has_water:
                thickness = ti.min(t, 8.0)
        if has_water:
            transmission = ti.exp(-ti.Vector([0.48, 0.18, 0.095]) * thickness / clarity)
            through_water = color * transmission + ti.Vector([0.035, 0.22, 0.20]) * (1.0 - transmission)
            f0 = ((1.333 - 1.0) / (1.333 + 1.0)) ** 2
            fresnel = f0 + (1.0 - f0) * (1.0 - cosine) ** 5
            color = through_water * (1.0 - fresnel) + reflection
            glint = ggx_brdf(ti.Vector([0.0, 0.0, 0.0]), ti.max(0.045, self.water_controls[None].x), 0.0, normal, -direction, self.sun_direction(lighting), f0)
            if glint.max() > 0.005:
                shade = self.visibility(p, normal, self.sun_direction(lighting), 0)
                color += glint * ti.Vector([1.0, 0.80, 0.52]) * 3.2 * shade
        return color

    @ti.kernel
    def render(self, eye: ti.types.vector(3, ti.f32), forward: ti.types.vector(3, ti.f32),
               right: ti.types.vector(3, ti.f32), up: ti.types.vector(3, ti.f32),
               clock: ti.f32, amplitude: ti.f32, frequency: ti.f32,
               clarity: ti.f32, lighting: ti.f32, exposure: ti.f32, scale: ti.f32, samples: ti.i32):
        for x, y in self.image:
            px = (2.0 * (x + 0.5) / self.width - 1.0) * self.width / self.height * scale
            py = (2.0 * (y + 0.5) / self.height - 1.0) * scale
            color = ti.Vector([0.0, 0.0, 0.0])
            for sample in range(samples):
                offset = ti.Vector([0.5, 0.5])
                if samples == 4:
                    offset = ti.Vector([0.25 + 0.5 * (sample % 2), 0.25 + 0.5 * (sample // 2)])
                sx = (2.0 * (x + offset.x) / self.width - 1.0) * self.width / self.height * scale
                sy = (2.0 * (y + offset.y) / self.height - 1.0) * scale
                direction = (forward + right * sx + up * sy).normalized()
                color += self.radiance(eye, direction, clock, amplitude, frequency, clarity, lighting, 2.0 * scale / self.height)
            color /= samples
            self.post.hdr[x, y] = color

    def draw(self, camera, clock, amplitude=0.035, frequency=7.0, clarity=1.4, lighting=1.0, exposure=1.05):
        water_controls = tuple(self.water_controls[None])
        key = (clock if amplitude != 0.0 else 0.0, amplitude, frequency, lighting, water_controls[1])
        # Reuse the same map while paused; no floating-atomic resampling noise.
        if self._caustic_key is None or (water_controls[2] > 0.0 and key != self._caustic_key):
            self.trace_caustics(clock, amplitude, frequency, lighting, 1)
            self.filter_caustics()
            self._caustic_key = key
        self.render(*camera.basis(), clock, amplitude, frequency, clarity, lighting,
                    exposure, math.tan(math.radians(camera.fov / 2)), self.samples)
        bloom, threshold, vignette = self.post_controls[None]
        self.post.apply(exposure, bloom, threshold, vignette)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Demo 01: courtyard pool and analytical ripples")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--samples", type=int, choices=(1, 4), default=4, help="Spatial samples per pixel (default: 4)")
    parser.add_argument("--lighting", choices=("day", "sunset"), default="day")
    parser.add_argument("--shadow-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-ao", action="store_true")
    parser.add_argument("--water-roughness", type=float, default=0.16)
    parser.add_argument("--reflection-samples", type=int, choices=(1, 4), default=4)
    parser.add_argument("--no-water-detail", action="store_true")
    parser.add_argument("--no-caustics", action="store_true")
    parser.add_argument("--no-bloom", action="store_true")
    parser.add_argument("--preset", choices=("overview", "waterline", "top"), default="overview")
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/demo_01.png"))
    parser.add_argument("--time", type=float, default=2.0, help="Initial wave time in seconds")
    parser.add_argument("--frames", type=int, default=1, help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0, help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache for troubleshooting")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0 or not math.isfinite(args.time):
        parser.error("width/height must be >= 32, frames >= 1, and time finite")
    if not math.isfinite(args.water_roughness) or not 0.0 <= args.water_roughness <= 0.45:
        parser.error("water-roughness must be finite and between 0 and 0.45")
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    renderer = PoolRenderer(args.width, args.height, args.samples)
    renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
    renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 1.0, 0.0 if args.no_caustics else 1.0, args.reflection_samples]
    renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
    camera = OrbitCamera()
    camera.preset(args.preset)
    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            renderer.draw(camera, args.time + frame / 60.0, lighting=1.0 if args.lighting == "day" else 0.0)
            ti.sync()
            timings.append(time.perf_counter() - frame_start)
        print(f"Saved {renderer.save(args.output)} ({args.width}x{args.height})")
        print(f"{args.frames} frame(s), including compilation: {time.perf_counter() - start:.2f}s")
        if len(timings) > 1:
            print(f"Post-compilation render mean: {np.mean(timings[1:]) * 1000:.1f} ms/frame")
        return

    amplitude, frequency, clarity, lighting, exposure, speed = 0.035, 7.0, 1.4, 1.0, 1.05, 1.0
    lighting = 1.0 if args.lighting == "day" else 0.0
    # Cold JIT compilation must finish before creating a native window: GGUI
    # polls OS events in show(), which cannot run while draw() is compiling.
    print("[Startup] Preparing first frame; cold compilation can take a minute or more. Window opens when ready.", flush=True)
    warmup_start = time.perf_counter()
    renderer.draw(camera, args.time, amplitude, frequency, clarity, lighting, exposure)
    ti.sync()
    print(f"[Startup] First frame ready in {time.perf_counter() - warmup_start:.2f}s. Opening window...", flush=True)
    window = ti.ui.Window("01 / Courtyard Ripples", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(renderer.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    paused, auto_orbit, show_panel = False, False, True
    clock, previous_time, previous_mouse = args.time, time.perf_counter(), None
    frame_ms, window_frames = 0.0, 1
    drag_in_scene = False
    print("LMB orbit | RMB pan | W/S zoom | 1/2/3 views | R reset | Space pause | N step | A orbit | H panel | P screenshot | Esc quit")
    while window.running and (not args.window_frames or window_frames < args.window_frames):
        now = time.perf_counter()
        dt = min(now - previous_time, 0.05)
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
                camera.preset({"1": "overview", "2": "waterline", "3": "top"}[key])
                auto_orbit = False
            elif key == "r":
                camera.preset("overview")
                clock, amplitude, frequency, clarity, lighting, exposure, speed = args.time, 0.035, 7.0, 1.4, 1.0, 1.05, 1.0
                renderer.samples = args.samples
                renderer.light_controls[None] = [math.radians(1.2), args.shadow_samples, 0.0 if args.no_ao else 0.65, 1.0]
                renderer.water_controls[None] = [args.water_roughness, 0.0 if args.no_water_detail else 1.0, 0.0 if args.no_caustics else 1.0, args.reflection_samples]
                renderer.post_controls[None] = [0.0 if args.no_bloom else 0.08, 1.0, 0.12]
                lighting = 1.0 if args.lighting == "day" else 0.0
                paused, auto_orbit = False, False
            elif key == "p":
                save_frame = True
            elif key in (ti.ui.LMB, ti.ui.RMB):
                drag_in_scene = not in_panel
        if not window.running:
            break
        # Reserve the panel rectangle so slider drags never move the camera.
        if previous_mouse is not None and drag_in_scene:
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
            with gui.sub_window("WATER / FINISH", 0.72, 0.335, 0.26, 0.61):
                controls = renderer.water_controls[None]
                controls[0] = gui.slider_float("Water roughness", controls[0], 0.0, 0.45)
                controls[1] = gui.slider_float("Fine waves", controls[1], 0.0, 2.0)
                controls[3] = 4 if gui.checkbox("4x reflections", controls[3] == 4) else 1
                controls[2] = gui.slider_float("Caustics", controls[2], 0.0, 1.0)
                renderer.water_controls[None] = controls
                finish = renderer.post_controls[None]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                renderer.post_controls[None] = finish
                gui.text("Zero strength disables an effect")
                gui.text("No temporal accumulation")
            with gui.sub_window("COURTYARD / 01", 0.02, 0.025, 0.30, 0.82):
                gui.text("Analytical ripples / Fresnel / Refraction")
                gui.text(f"Frame {frame_ms:.1f} ms (includes UI)")
                paused = gui.checkbox("Pause [Space]", paused)
                auto_orbit = gui.checkbox("Auto orbit [A]", auto_orbit)
                if gui.button("Single step [N]"):
                    step = True
                amplitude = gui.slider_float("Wave height", amplitude, 0.0, 0.075)
                frequency = gui.slider_float("Wave frequency", frequency, 3.0, 12.0)
                speed = gui.slider_float("Time speed", speed, 0.0, 2.0)
                clarity = gui.slider_float("Water clarity", clarity, 0.35, 3.0)
                lighting = gui.slider_float("Day / sunset", lighting, 0.0, 1.0)
                exposure = gui.slider_float("Exposure", exposure, 0.5, 1.8)
                renderer.samples = 4 if gui.checkbox("4x spatial AA", renderer.samples == 4) else 1
                if gui.button("Overview [1]"):
                    camera.preset("overview")
                    auto_orbit = False
                if gui.button("Waterline [2]"):
                    camera.preset("waterline")
                    auto_orbit = False
                if gui.button("Top view [3]"):
                    camera.preset("top")
                    auto_orbit = False
                gui.text("LMB orbit / RMB pan")
                gui.text("W/S or arrows: zoom / R: reset")
                gui.text("H: hide panel / P: save / Esc: exit")
        if not paused:
            clock += dt * speed
        elif step:
            clock += 1.0 / 60.0
        renderer.draw(camera, clock, amplitude, frequency, clarity, lighting, exposure)
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
