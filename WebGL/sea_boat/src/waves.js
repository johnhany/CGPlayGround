import * as THREE from 'three';

const GRAVITY = 9.8;

// Gerstner 波参数表：方向收敛于主方向正负三十度附近，使浪读作单向推进；
// 主导浪波长最长，承载波群包络；全部波陡之和小于 1 以防波峰打环；
// 波长需大于水面网格边长的两倍
const WAVE_TABLE = [
  { dir: [ 1.00,  0.10], steepness: 0.100, wavelength: 40.0 },
  { dir: [ 0.95, -0.30], steepness: 0.085, wavelength: 24.0 },
  { dir: [ 0.80,  0.45], steepness: 0.065, wavelength: 14.0 },
  { dir: [ 1.00, -0.10], steepness: 0.050, wavelength:  9.0 },
  { dir: [ 0.88,  0.35], steepness: 0.038, wavelength:  5.6 }
];

// 风格化 Gerstner 波场：在标准位移上叠加二次谐波锐化与波峰前倾，
// 使波形呈现前坡陡、后坡缓、波峰前探的非对称形态；
// 主导浪乘以慢变包络，让画面周期性涌起一道主角浪。
// 公式与 water_common.js 的 GLSL 片段严格一致，浮力采样与渲染贴合
export class WaveField {
  constructor() {
    this.windScale = 1.0;       // 风浪强度，统一缩放全部波的振幅
    this.amplitudeScale = 1.0;  // 风格级振幅比例
    this.speedScale = 1.0;      // 波形演化速度倍率
    this.time = 0.0;            // 波面相位时钟，单位秒
    this.sharp = 0.32;          // 谐波锐化系数，越大前坡越陡
    this.lean = 0.30;           // 波峰前倾系数，占振幅的比例
    this.groupDepth = 0.35;     // 波群包络深度，仅作用于主导浪

    this.waves = WAVE_TABLE.map((w) => {
      const dir = new THREE.Vector2(w.dir[0], w.dir[1]).normalize();
      const k = (2.0 * Math.PI) / w.wavelength;
      return {
        dx: dir.x,
        dz: dir.y,
        k,
        steepness: w.steepness,
        wavelength: w.wavelength,
        amp: w.steepness / k,                // 振幅 a = S / k
        phaseRate: Math.sqrt(GRAVITY * k)    // 相位项 k * c = sqrt(g k)
      };
    });

    // 包络周期取主导浪周期的五倍
    this.groupRate = this.waves[0].phaseRate / 5.0;
  }

  // 主导浪的波群包络，随时间在 0.65 至 1.35 之间缓变
  get envelope() {
    return 1.0 - this.groupDepth * Math.cos(this.groupRate * this.time);
  }

  // 当前包络与风浪强度下的最大瞬时波高估计，供浪花播撒阈值使用
  maxAmplitude() {
    let sum = 0.0;
    const scale = this.windScale * this.amplitudeScale;
    for (let i = 0; i < this.waves.length; i++) {
      const a = this.waves[i].amp * scale * (i === 0 ? this.envelope : 1.0);
      sum += a;
    }
    return sum * 1.06;   // 锐化使波峰略高于基波振幅
  }

  // 计算静止网格点 (x, z) 处的波面位移，结果写入 out 避免逐点分配对象
  sample(x, z, out) {
    let dx = 0.0;
    let dy = 0.0;
    let dz = 0.0;
    const t = this.time;
    const scale = this.windScale * this.amplitudeScale;
    const sharp = this.sharp;
    const lean = this.lean;

    for (let i = 0; i < this.waves.length; i++) {
      const w = this.waves[i];
      const a = w.amp * scale * (i === 0 ? this.envelope : 1.0);
      const f = w.k * (w.dx * x + w.dz * z) - w.phaseRate * t;
      const s = Math.sin(f);
      const c = Math.cos(f);
      const sh = s - sharp * Math.sin(2.0 * f) * 0.5;   // 锐化波形
      const hz = c + lean * sh;                          // 水平位移含波峰前倾
      dx += w.dx * a * hz;
      dz += w.dz * a * hz;
      dy += a * sh;
    }

    out.dx = dx;
    out.dy = dy;
    out.dz = dz;
    return out;
  }
}
