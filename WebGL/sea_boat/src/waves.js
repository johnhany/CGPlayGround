import * as THREE from 'three';

const GRAVITY = 9.8;

// Gerstner 波参数表：方向为归一化前的原始向量，全部波陡之和小于 1 以防波峰打环，
// 波长需大于水面网格边长的两倍，否则会出现锯齿或完全不可见
const WAVE_TABLE = [
  { dir: [ 1.00,  0.20], steepness: 0.100, wavelength: 34.0 },
  { dir: [ 0.68, -0.62], steepness: 0.085, wavelength: 21.0 },
  { dir: [-0.35,  0.88], steepness: 0.065, wavelength: 13.0 },
  { dir: [ 0.88, -0.18], steepness: 0.050, wavelength:  8.0 },
  { dir: [-0.22, -0.92], steepness: 0.038, wavelength:  5.2 }
];

// 多组 Gerstner 波叠加的水平面。渲染端按此函数位移网格顶点，
// 浮力端按同一函数采样船底波高，保证船体与水面严格贴合
export class WaveField {
  constructor() {
    this.windScale = 1.0;       // 风浪强度，统一缩放全部波的振幅
    this.amplitudeScale = 1.0;  // 风格级振幅比例，吉卜力风格下波形更平缓
    this.speedScale = 1.0;      // 波形演化速度倍率
    this.time = 0.0;            // 波面相位时钟，单位秒

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
        phaseRate: Math.sqrt(GRAVITY * k)    // 相位项 k * c = sqrt(g k)，深水波速 c = sqrt(g / k)
      };
    });
  }

  // 计算静止网格点 (x, z) 处的波面位移，结果写入 out 避免逐点分配对象
  sample(x, z, out) {
    let dx = 0.0;
    let dy = 0.0;
    let dz = 0.0;
    const t = this.time;
    const scale = this.windScale * this.amplitudeScale;

    for (let i = 0; i < this.waves.length; i++) {
      const w = this.waves[i];
      const a = w.amp * scale;
      const f = w.k * (w.dx * x + w.dz * z) - w.phaseRate * t;
      const s = Math.sin(f);
      const c = Math.cos(f);
      dx += w.dx * a * c;
      dz += w.dz * a * c;
      dy += a * s;
    }

    out.dx = dx;
    out.dy = dy;
    out.dz = dz;
    return out;
  }
}
