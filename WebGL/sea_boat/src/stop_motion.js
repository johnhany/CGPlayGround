// 曝光时钟独立于波速。场景在两次曝光之间保持同一个时间和姿态。
export class StopMotionClock {
  constructor(fps = 12) {
    if (!Number.isFinite(fps) || fps <= 0) throw new RangeError('fps must be positive');
    this.fps = fps;
    this.reset(0);
  }

  reset(time) {
    this.elapsed = 0;
    this.frame = 0;
    this.time = time;
  }

  advance(delta, continuousTime, speed) {
    this.elapsed += delta;
    const frame = Math.floor((this.elapsed + 1e-9) * this.fps);
    if (frame === this.frame) return false;
    this.time = continuousTime - (this.elapsed - frame / this.fps) * speed;
    this.frame = frame;
    return true;
  }
}
