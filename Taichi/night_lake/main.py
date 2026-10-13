"""Night lake stage 6: photographic finish — thin-lens depth of field,
film grain, and the final exposure calibration.

Run: uv run python main.py   (from night_lake/)
     uv run python -m night_lake.main   (from Taichi/)
Export: uv run python main.py --headless --output output/night_lake.png
"""

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

if __package__:
    from night_lake.camera import FixedCamera
    from night_lake.scene import NightLakeScene
else:
    # Allow `python main.py` from inside night_lake/ as well as `python -m night_lake.main`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from night_lake.camera import FixedCamera
    from night_lake.scene import NightLakeScene

__all__ = ["main", "parse_args"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Night lake: stage 1 sky skeleton")
    parser.add_argument("--backend", choices=("gpu", "vulkan", "metal", "cpu"), default="gpu")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--headless", action="store_true", help="Render without a window and save PNG")
    parser.add_argument("--output", type=Path, default=Path("output/night_lake.png"))
    parser.add_argument("--time", type=float, default=2.0, help="Initial scene time in seconds")
    parser.add_argument("--frames", type=int, default=1, help="Headless frames at a fixed 1/60 s step; save the last")
    parser.add_argument("--window-frames", type=int, default=0, help="Close interactive window after N frames (0: unlimited)")
    parser.add_argument("--rain", type=float, default=40.0, help="Rain rate in drops per second")
    parser.add_argument("--rain-warmup", type=float, default=2.0,
                        help="Seconds of rain simulation to pre-roll before the first frame")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk compilation cache for troubleshooting")
    parser.add_argument("--dof", action="store_true",
                        help="Enable thin-lens depth of field (off by default: it misses the frame budget)")
    args = parser.parse_args(argv)
    if args.width < 32 or args.height < 32 or args.frames < 1 or args.window_frames < 0 or not math.isfinite(args.time):
        parser.error("width/height must be >= 32, frames >= 1, and time finite")
    if not math.isfinite(args.rain) or args.rain < 0.0:
        parser.error("rain must be finite and non-negative")
    if not math.isfinite(args.rain_warmup) or args.rain_warmup < 0.0:
        parser.error("rain-warmup must be finite and non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    cache_path = Path(__file__).resolve().parent.parent / ".taichi-cache"
    ti.init(arch=getattr(ti, args.backend), offline_cache=not args.no_cache,
            offline_cache_file_path=str(cache_path), random_seed=17)
    scene = NightLakeScene(args.width, args.height)
    if args.dof:
        lens = list(scene.camera_controls[None])
        lens[0] = 1.0
        scene.camera_controls[None] = lens
    scene.post_controls[None] = [0.12, 0.9, 0.22, 0.025]
    scene.water_controls[None] = [1.0, 1.0, args.rain, 0.06]
    # A zero rain rate turns the whole rain layer off, curtain included.
    # The last slot is the streak tail exposure (persistence) control.
    scene.rain_controls[None] = [1.0, 1.0, 0.8 if args.rain > 0.0 else 0.0, 1.0]
    camera = FixedCamera()
    scene.warm_rain(args.rain_warmup)
    if args.headless:
        start = time.perf_counter()
        timings = []
        for frame in range(args.frames):
            frame_start = time.perf_counter()
            scene.draw(camera, args.time + frame / 60.0)
            ti.sync()
            timings.append(time.perf_counter() - frame_start)
        print(f"Saved {scene.save(args.output)} ({args.width}x{args.height})")
        print(f"{args.frames} frame(s), including compilation: {time.perf_counter() - start:.2f}s")
        if len(timings) > 1:
            steady = np.array(timings[1:]) * 1000
            print(f"Post-compilation render mean: {steady.mean():.1f} ms/frame")
            print(f"Post-compilation render median: {np.median(steady):.1f} ms/frame")
        return

    exposure, fog_strength, show_panel = 0.92, 1.0, True
    clock = args.time
    # Cold JIT compilation must finish before creating a native window: GGUI
    # polls OS events in show(), which cannot run while draw() is compiling.
    print("[Startup] Preparing first frame; cold compilation can take a minute or more. Window opens when ready.", flush=True)
    warmup_start = time.perf_counter()
    scene.draw(camera, clock, exposure, fog_strength)
    ti.sync()
    print(f"[Startup] First frame ready in {time.perf_counter() - warmup_start:.2f}s. Opening window...", flush=True)
    window = ti.ui.Window("Night Lake / 06 Camera", (args.width, args.height), vsync=True)
    canvas, gui = window.get_canvas(), window.get_gui()
    canvas.set_image(scene.image)
    window.show()
    print("[Startup] Scene displayed; controls are ready.", flush=True)
    frame_ms, window_frames = 0.0, 1
    previous_time = time.perf_counter()
    print("H panel | P screenshot | Esc quit")
    while window.running and (not args.window_frames or window_frames < args.window_frames):
        now = time.perf_counter()
        dt = min(now - previous_time, 0.05)
        previous_time = now
        save_frame = False
        for event in window.get_events(ti.ui.PRESS):
            key = event.key
            if key == ti.ui.ESCAPE:
                window.running = False
            elif key == "h":
                show_panel = not show_panel
            elif key == "p":
                save_frame = True
        if not window.running:
            break
        if show_panel:
            # One snapshot/commit pair for every panel: six control blocks
            # travel to the UI and back as a single 24-float vector.
            state = scene.controls_snapshot().copy()
            with gui.sub_window("RAIN", 0.02, 0.02, 0.26, 0.20):
                rain = state[12:16]
                rain[0] = gui.slider_float("Streak strength", rain[0], 0.0, 2.0)
                rain[1] = gui.slider_float("Splash amount", rain[1], 0.0, 1.5)
                rain[2] = gui.slider_float("Curtain strength", rain[2], 0.0, 2.0)
                rain[3] = gui.slider_float("Streak persistence", rain[3], 0.0, 3.0)
                state[12:16] = rain
            with gui.sub_window("CAMERA/FILM", 0.02, 0.20, 0.26, 0.24):
                lens = state[4:8]
                lens[0] = 1.0 if gui.checkbox("Depth of field", lens[0] > 0.5) else 0.0
                lens[1] = gui.slider_float("Aperture", lens[1], 0.0, 0.3)
                lens[2] = gui.slider_float("Focus distance", lens[2], 1.0, 60.0)
                lens[3] = gui.slider_float("DOF samples", lens[3], 1.0, 8.0)
                state[4:8] = lens
                finish = state[8:12]
                finish[3] = gui.slider_float("Film grain", finish[3], 0.0, 0.12)
                state[8:12] = finish
            with gui.sub_window("WATER", 0.72, 0.02, 0.26, 0.24):
                controls = state[0:4]
                controls[0] = gui.slider_float("Wave height", controls[0], 0.0, 2.0)
                controls[1] = gui.slider_float("Ripple strength", controls[1], 0.0, 2.0)
                controls[2] = gui.slider_float("Rain rate (/s)", controls[2], 0.0, 200.0)
                controls[3] = gui.slider_float("Roughness", controls[3], 0.02, 0.3)
                state[0:4] = controls
            with gui.sub_window("TERRAIN", 0.72, 0.28, 0.26, 0.20):
                terrain = state[16:20]
                terrain[0] = gui.slider_float("Shore distance", terrain[0], 18.0, 40.0)
                terrain[1] = gui.slider_float("Detail amount", terrain[1], 0.0, 2.0)
                terrain[2] = gui.slider_float("Wetness decay", terrain[2], 0.4, 4.0)
                terrain[3] = gui.slider_float("Grass line", terrain[3], 0.2, 1.2)
                state[16:20] = terrain
            with gui.sub_window("VEGETATION", 0.72, 0.50, 0.26, 0.20):
                vegetation = state[20:24]
                vegetation[0] = gui.slider_float("Grass density", vegetation[0], 0.0, 1.0)
                vegetation[1] = gui.slider_float("Tree density", vegetation[1], 0.0, 1.0)
                vegetation[2] = gui.slider_float("Wind", vegetation[2], 0.0, 1.5)
                vegetation[3] = gui.slider_float("Leaf brightness", vegetation[3], 0.5, 1.5)
                state[20:24] = vegetation
            with gui.sub_window("FINISH", 0.72, 0.72, 0.26, 0.26):
                finish = state[8:12]
                finish[0] = gui.slider_float("Bloom", finish[0], 0.0, 0.25)
                finish[1] = gui.slider_float("Bloom threshold", finish[1], 0.3, 2.0)
                finish[2] = gui.slider_float("Vignette", finish[2], 0.0, 0.3)
                state[8:12] = finish
                exposure = gui.slider_float("Exposure", exposure, 0.5, 1.8)
                fog_strength = gui.slider_float("Fog strength", fog_strength, 0.0, 2.0)
                gui.text("P: save / H: hide panel / Esc: exit")
            scene.commit_controls(state)
        clock += dt
        scene.draw(camera, clock, exposure, fog_strength, dt)
        if save_frame:
            print(f"Saved {scene.save(args.output)}")
        canvas.set_image(scene.image)
        window.show()
        window_frames += 1
        if args.window_frames and window_frames >= args.window_frames:
            window.running = False
        frame_ms = 0.9 * frame_ms + 0.1 * (time.perf_counter() - now) * 1000


if __name__ == "__main__":
    main()
