import assert from 'node:assert/strict';
import { StopMotionClock } from '../src/stop_motion.js';

for (const renderFps of [30, 60, 144]) {
  for (const speed of [0.2, 1, 2]) {
    const clock = new StopMotionClock(12);
    clock.reset(3.7);
    let raw = 3.7;
    const captures = [];
    for (let i = 1; i <= renderFps; i++) {
      const previous = clock.time;
      raw += speed / renderFps;
      if (clock.advance(1 / renderFps, raw, speed)) captures.push(clock.time);
      else assert.equal(clock.time, previous, '同一曝光帧内时间必须保持不变');
    }
    assert.equal(captures.length, 12, '波速和显示刷新率不应改变曝光次数');
    for (let i = 0; i < captures.length; i++) {
      assert.ok(Math.abs(captures[i] - (3.7 + (i + 1) * speed / 12)) < 1e-9);
    }
    clock.reset(20);
    assert.equal(clock.time, 20);
    assert.equal(clock.advance(1 / 144, 20 + speed / 144, speed), false);
  }
}
console.log('PASS: 30/60/144 Hz × 0.2/1/2 speed; 12 exposures per second, held poses, deterministic samples and reset');
