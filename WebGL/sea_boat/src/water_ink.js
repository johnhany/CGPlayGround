import * as THREE from 'three';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, NOISE_GLSL, buildWaveUniforms } from './water_common.js';

const SIZE = 280;       // 水面边长，单位米
const SEGMENTS = 200;   // 网格划分数，顶点位移在 GPU 完成

// 水墨留白水面：无光照平绘，墨量随波高变化——波谷淡出为纸色留白，
// 波峰聚成浓墨，叠低频晕染斑块使浓淡不匀；波峰等高线上以噪声扰动的
// 一道淡墨线作笔触。无反射无高光，天空即纸色，雾为纸色渐隐
export function createInkWater(waveField, waterConfig) {
  const geometry = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
  geometry.rotateX(-Math.PI / 2);

  const uniforms = THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog,
    {
      uPaperColor: { value: new THREE.Color(waterConfig.paper) },
      uInkDeep: { value: new THREE.Color(waterConfig.inkDeep) },
      uInkMid: { value: new THREE.Color(waterConfig.inkMid) }
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

      uniform vec3 uPaperColor;
      uniform vec3 uInkDeep;
      uniform vec3 uInkMid;

      varying vec3 vWorldPos;
      varying float vHeight;

      ${NOISE_GLSL}

      void main() {
        vec2 p = vWorldPos.xz;

        // 墨量随波高增加：波谷近纸留白，中景为可读的淡灰水洗，波峰聚墨
        float wob = (fbm(p * 0.22) - 0.5) * 0.22;
        float hb = vHeight + wob;
        float inkT = smoothstep(-0.55, 0.60, hb);
        float blotch = 0.55 + 0.45 * fbm(p * 0.09 + 3.0);
        float density = (0.20 + 0.74 * pow(inkT, 1.45)) * blotch;

        // 纸色为底，浓淡墨按密度覆盖，浓墨核心在密度最高处
        vec3 col = uPaperColor * (1.0 - clamp(density, 0.0, 0.88));
        col = mix(col, uInkDeep, smoothstep(0.60, 0.90, density));

        // 波峰等高线上的淡墨笔触，位置与浓度随噪声摇曳
        float level = 0.40 + (fbm(p * 0.4) - 0.5) * 0.24;
        float stroke = (1.0 - smoothstep(0.015, 0.05, abs(hb - level)))
          * (0.35 + 0.65 * fbm(p * 1.4 + 5.0));
        col = mix(col, mix(uInkMid, uInkDeep, 0.6), stroke * 0.6);

        // 纸纹
        col *= 0.97 + 0.05 * vnoise(p * 2.8);

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
    uniforms.uEnv.value = waveField.envelope;
  }

  return { mesh, uniforms, update };
}
