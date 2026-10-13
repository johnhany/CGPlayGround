import * as THREE from 'three';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, NOISE_GLSL, buildWaveUniforms } from './water_common.js';

const SIZE = 280;       // 水面边长，单位米
const SEGMENTS = 200;   // 网格划分数，顶点位移在 GPU 完成

// 浮世绘海浪水面：无光照平绘，片元按波高平涂五档靛蓝色阶；
// 爪状浪花以噪声扰动的等高线为中心生成米色条带，条带断出指状空隙，
// 外缘勾一圈深靛墨线，贴近木版画的套印轮廓；整体叠极轻的纸面斑驳
export function createUkiyoeWater(waveField, waterConfig) {
  const geometry = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
  geometry.rotateX(-Math.PI / 2);

  const uniforms = THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog,
    {
      uColor0: { value: new THREE.Color(waterConfig.c0) },
      uColor1: { value: new THREE.Color(waterConfig.c1) },
      uColor2: { value: new THREE.Color(waterConfig.c2) },
      uColor3: { value: new THREE.Color(waterConfig.c3) },
      uColor4: { value: new THREE.Color(waterConfig.c4) },
      uFoamColor: { value: new THREE.Color(waterConfig.foam) },
      uInkColor: { value: new THREE.Color(waterConfig.ink) }
    }
  ]);
  // merge 会克隆数值，波形数组在合并后单独挂载
  Object.assign(uniforms, buildWaveUniforms(waveField));

  const material = new THREE.ShaderMaterial({
    uniforms,
    fog: true,
    vertexShader: /* glsl */`
      #include <fog_pars_vertex>

      ${WAVE_UNIFORMS_GLSL}
      ${GERSTNER_GLSL}

      varying vec3 vWorldPos;
      varying float vHeight;

      void main() {
        vec3 base = position;
        vec3 nrm;
        vec3 pos = base + gerstnerDisp(base.xz, nrm);

        vHeight = pos.y;
        vWorldPos = pos;

        vec4 mvPosition = viewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */`
      #include <fog_pars_fragment>

      uniform vec3 uColor0;
      uniform vec3 uColor1;
      uniform vec3 uColor2;
      uniform vec3 uColor3;
      uniform vec3 uColor4;
      uniform vec3 uFoamColor;
      uniform vec3 uInkColor;
      uniform float uTime;

      varying vec3 vWorldPos;
      varying float vHeight;

      ${NOISE_GLSL}

      void main() {
        vec2 p = vWorldPos.xz;
        float t = uTime;

        // 波高加轻微噪声扰动后再分档，使色带边界呈有机的等高线形态
        float wob = (fbm(p * 0.25) - 0.5) * 0.14;
        float hb = vHeight + wob;

        // 五档靛蓝平涂，边界以极窄过渡抗锯齿
        float aa = 0.02;
        vec3 col = uColor0;
        col = mix(col, uColor1, smoothstep(-0.28 - aa, -0.28 + aa, hb));
        col = mix(col, uColor2, smoothstep(0.04 - aa, 0.04 + aa, hb));
        col = mix(col, uColor3, smoothstep(0.26 - aa, 0.26 + aa, hb));
        col = mix(col, uColor4, smoothstep(0.52 - aa, 0.52 + aa, hb));

        // 爪状浪花：噪声扰动的等高线条带，断出指状空隙，外缘勾墨线
        float flow = t * 0.10;
        float level = 0.60 + (fbm(p * 0.35 + vec2(flow, -flow)) - 0.5) * 0.40;
        float d = abs(hb - level);
        float seg = fbm(p * 0.9 + vec2(-flow, flow * 0.6) + 13.1);
        float band = (1.0 - smoothstep(0.045, 0.085, d)) * smoothstep(0.40, 0.52, seg);
        float line = smoothstep(0.085, 0.10, d) - smoothstep(0.115, 0.13, d);

        // 次级细浪花带
        float level2 = 0.34 + (fbm(p * 0.5 + vec2(7.7, -flow)) - 0.5) * 0.26;
        float d2 = abs(hb - level2);
        float band2 = (1.0 - smoothstep(0.03, 0.05, d2)) * smoothstep(0.44, 0.55, seg);

        col = mix(col, uFoamColor, clamp(band + band2 * 0.8, 0.0, 1.0));
        col = mix(col, uInkColor, line * 0.9);

        // 纸面斑驳
        col *= 0.965 + 0.07 * vnoise(p * 2.6);

        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        #include <fog_fragment>
      }
    `
  });

  const mesh = new THREE.Mesh(geometry, material);

  function update() {
    uniforms.uTime.value = waveField.time;
    uniforms.uWindScale.value = waveField.windScale * waveField.amplitudeScale;
  }

  return { mesh, uniforms, update };
}
