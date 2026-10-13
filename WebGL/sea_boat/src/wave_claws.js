import * as THREE from 'three';
import { STYLE_CONFIG } from './styles.js';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, buildWaveUniforms } from './water_common.js';
import { UKIYO_PAINT_GLSL, ukiyoPaintUniforms } from './water_ukiyoe.js';

const ACROSS = 96;
const PROFILE = 48;
const TAU = Math.PI * 2;

// 一条完整浪脊：横向坐标严格单调，包络只控制高度及卷曲，不折返网格。
function createGeometry() {
  const geo = new THREE.BufferGeometry();
  const count = (ACROSS + 1) * (PROFILE + 1);
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3).setUsage(THREE.DynamicDrawUsage));
  geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(count * 2), 2));
  geo.setAttribute('base', new THREE.BufferAttribute(new Float32Array(count * 2), 2).setUsage(THREE.DynamicDrawUsage));
  const indices = [];
  for (let i = 0; i <= ACROSS; i++) {
    for (let j = 0; j <= PROFILE; j++) {
      const n = i * (PROFILE + 1) + j;
      geo.attributes.uv.setXY(n, i / ACROSS, j / PROFILE);
      if (i < ACROSS && j < PROFILE) {
        const b = n + PROFILE + 1;
        indices.push(n, n + 1, b, n + 1, b + 1, b);
      }
    }
  }
  geo.setIndex(indices);
  return geo;
}

function createPaint(waveField) {
  return new THREE.ShaderMaterial({
    uniforms: { ...ukiyoPaintUniforms(STYLE_CONFIG.E.water), ...buildWaveUniforms(waveField), uStrength: { value: 1 } },
    side: THREE.DoubleSide, toneMapped: false,
    polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
    vertexShader: /* glsl */`
      attribute vec2 base;
      varying vec2 vBase;
      varying vec2 vUv;
      varying vec3 vWorld;
      void main() {
        vBase = base;
        vUv = uv;
        vWorld = position;
        gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      uniform float uStrength;
      ${WAVE_UNIFORMS_GLSL}
      ${GERSTNER_GLSL}
      varying vec2 vBase;
      varying vec2 vUv;
      varying vec3 vWorld;
      ${UKIYO_PAINT_GLSL}
      void main() {
        float v = vUv.y;
        float edge = pow(max(sin(vUv.x * 3.14159265), 0.0), 0.8) * uStrength;
        // 根部与水面共用颜料；卷曲内侧用深靛、浪背用中蓝。
        vec3 n;
        vec3 disp = gerstnerDisp(vBase, n);
        vec3 col = seaPaint(disp.y, dot(n.xz, uWaves[0].xy));
        float body = smoothstep(0.02, 0.20, v) * smoothstep(0.02, 0.25, edge);
        vec3 paint = mix(uMid, uLight, smoothstep(0.12, 0.62, v));
        paint = mix(paint, uDeep, smoothstep(0.70, 0.83, v));
        col = mix(col, paint, body);
        // 顺着卷曲母线的细流线，线宽按片元导数抗锯齿。
        float line = sin(vUv.x * 3.14159265 * 22.0 + v * 1.5);
        float mask = smoothstep(0.12, 0.25, v) * (1.0 - smoothstep(0.72, 0.88, v));
        col = mix(col, uInk, inkStroke(line, 0.035) * mask * body * 0.5);
        float lip = 0.90 + 0.038 * sin(vUv.x * 3.14159265 * 18.0) + 0.010 * sin(vUv.x * 3.14159265 * 31.0);
        float aa = max(fwidth(v), 0.002);
        float foam = smoothstep(lip - aa, lip + aa, v) * smoothstep(0.03, 0.20, edge);
        col = mix(col, uFoam, foam);
        col = mix(col, uInk, inkStroke(v - lip, 0.003) * body * 0.65);
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }
    `
  });
}

export class WaveClaws {
  constructor(scene, waveField) {
    this.group = new THREE.Group();
    this.group.visible = false;
    scene.add(this.group);
    // 主浪靠侧后方，次浪向远处递减；固定布局不再依相机传送。
    this.claws = [
      { lateral: 5, width: 32, height: 12.0 },
      { lateral: -8, width: 38, height: 3.2 },
      { lateral: 8, width: 48, height: 2.0 }
    ].map(def => {
      const mesh = new THREE.Mesh(createGeometry(), createPaint(waveField));
      mesh.frustumCulled = false; // 顶点逐帧移动，避免旧包围球误裁剪。
      this.group.add(mesh);
      return { ...def, mesh };
    });
    this.out = { dx: 0, dy: 0, dz: 0 };
  }

  update(dt, waveField) {
    const w = waveField.waves[0];
    // 整个浪体沿主浪相位前进；循环边界在高度为零时发生。
    const cycle = waveField.time * w.phaseRate / TAU;
    for (let c = 0; c < this.claws.length; c++) {
      const claw = this.claws[c];
      const offset = 0.35 + c * 0.21;
      const q = ((cycle + offset) % 1 + 1) % 1;
      const strength = Math.pow(Math.sin(Math.PI * q), 0.65)
        * THREE.MathUtils.smoothstep(q, 0, 0.08)
        * (1 - THREE.MathUtils.smoothstep(q, 0.92, 1));
      const alongCrest = (q - offset + 1.8 / TAU - c - 1) * w.wavelength;
      const amplitude = Math.min(1.35, waveField.windScale) * waveField.amplitudeScale;
      const uniforms = claw.mesh.material.uniforms;
      uniforms.uStrength.value = strength * amplitude;
      uniforms.uTime.value = waveField.time;
      uniforms.uWindScale.value = waveField.windScale * waveField.amplitudeScale;
      uniforms.uEnv.value = waveField.envelope;
      const attr = claw.mesh.geometry.attributes.position;
      for (let i = 0; i <= ACROSS; i++) {
        const u = i / ACROSS;
        const across = (u - 0.5) * claw.width;
        const envelope = Math.pow(Math.sin(Math.PI * u), 1.35) * (1.20 - 0.40 * u) * strength * amplitude;
        const height = claw.height * envelope;
        for (let j = 0; j <= PROFILE; j++) {
          const v = j / PROFILE;
          let forward, rise;
          if (v <= 0.60) {
            const s = v / 0.60;
            forward = -6.5 * (1 - s);
            rise = height * s * s;
          } else {
            const a = (v - 0.60) / 0.40 * Math.PI * 1.35;
            const radius = height * 0.38;
            forward = radius * (1 - Math.cos(a));
            rise = height + radius * Math.sin(a);
            // 指尖长短交错，浪唇自然分叉，替代圆弧排贴纸。
            const tip = Math.pow((v - 0.60) / 0.40, 3);
            rise += (Math.sin(u * Math.PI * 18) + 0.25 * Math.sin(u * Math.PI * 31)) * height * 0.12 * tip;
            forward += (Math.sin(u * Math.PI * 18) + 0.25 * Math.sin(u * Math.PI * 31)) * height * 0.11 * tip;
          }
          const along = alongCrest + forward;
          const lateral = claw.lateral + across;
          const x = w.dx * along - w.dz * lateral;
          const z = w.dz * along + w.dx * lateral;
          claw.mesh.geometry.attributes.base.setXY(i * (PROFILE + 1) + j, x, z);
          waveField.sample(x, z, this.out);
          // 每个点都采样并使用水平位移，浪根整段接入实际 GPU 波面。
          attr.setXYZ(i * (PROFILE + 1) + j,
            x + this.out.dx, this.out.dy + rise - 0.035, z + this.out.dz);
        }
      }
      attr.needsUpdate = true;
      claw.mesh.geometry.attributes.base.needsUpdate = true;
    }
  }
}
