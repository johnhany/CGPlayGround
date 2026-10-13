import * as THREE from 'three';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, NOISE_GLSL, buildWaveUniforms } from './water_common.js';

// 水面和卷浪共用颜料与分区函数，根部不会出现两套蓝色。
export const UKIYO_PAINT_GLSL = /* glsl */`
  uniform vec3 uDeep;
  uniform vec3 uMid;
  uniform vec3 uLight;
  uniform vec3 uFoam;
  uniform vec3 uInk;
  vec3 seaPaint(float h, float slope) {
    float shade = h * 0.42 + slope * 0.55;
    float aa = max(fwidth(shade), 0.012);
    vec3 col = mix(uDeep, uMid, smoothstep(-0.035 - aa, -0.005 + aa, shade));
    return mix(col, uLight, smoothstep(0.48 - aa, 0.52 + aa, shade));
  }
  float inkStroke(float d, float width) {
    float aa = max(fwidth(d), 0.0005);
    return 1.0 - smoothstep(width, width + aa, abs(d));
  }
`;

export function ukiyoPaintUniforms(config) {
  return {
    uDeep: { value: new THREE.Color(config.c1) },
    uMid: { value: new THREE.Color(config.c2) },
    uLight: { value: new THREE.Color(config.c3) },
    uFoam: { value: new THREE.Color(config.foam) },
    uInk: { value: new THREE.Color(config.ink) }
  };
}

export function createUkiyoeWater(waveField, config) {
  // 近景网格需要解析最短的 5.6 米波形；不再以粗三角形承载细墨线。
  const geometry = new THREE.PlaneGeometry(280, 280, 360, 360);
  geometry.rotateX(-Math.PI / 2);
  const uniforms = THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog, ukiyoPaintUniforms(config)
  ]);
  Object.assign(uniforms, buildWaveUniforms(waveField));
  const material = new THREE.ShaderMaterial({
    uniforms, fog: true, toneMapped: false,
    vertexShader: /* glsl */`
      #include <fog_pars_vertex>
      ${WAVE_UNIFORMS_GLSL}
      ${GERSTNER_GLSL}
      varying vec3 vWorld;
      varying vec2 vBase;
      void main() {
        vec3 n;
        vec3 p = position + gerstnerDisp(position.xz, n);
        vWorld = p;
        vBase = position.xz;
        vec4 mvPosition = viewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */`
      #include <fog_pars_fragment>
      ${WAVE_UNIFORMS_GLSL}
      ${GERSTNER_GLSL}
      varying vec3 vWorld;
      varying vec2 vBase;
      ${NOISE_GLSL}
      ${UKIYO_PAINT_GLSL}
      void main() {
        vec4 w = uWaves[0];
        float k = 6.2831853 / w.w;
        float phase = k * dot(w.xy, vBase) - sqrt(9.8 * k) * uTime;
        vec3 normal;
        vec3 displacement = gerstnerDisp(vBase, normal);
        float slope = dot(normal.xz, w.xy);
        vec3 col = seaPaint(displacement.y, slope);
        // 流线只出现在浪坡，按主浪相位推进；不逐档勾勒等高线。
        float curve = phase + 0.12 * sin(dot(vBase, vec2(-w.y, w.x)) * 0.34);
        float d = sin(curve * 3.0);
        float slopeMask = smoothstep(0.12, 0.48, slope);
        float distanceFade = 1.0 - smoothstep(30.0, 110.0, length(cameraPosition.xz - vWorld.xz));
        float lines = inkStroke(d, 0.025) * slopeMask * distanceFade;
        col = mix(col, uInk, lines * 0.24);
        // 白沫集中在高且陡的峰顶，远处逐渐省略细节。
        float crest = smoothstep(0.65, 0.95, displacement.y) * smoothstep(0.0, 0.2, slope);
        float streak = inkStroke(sin(curve + 0.10 * sin(vBase.y * 1.3)), 0.035);
        col = mix(col, uFoam, crest * streak * distanceFade * 0.55);
        // 稳定且很轻的纸纹；不使用随时间漂移的噪声。
        col *= 0.985 + 0.025 * vnoise(gl_FragCoord.xy * 0.7);
        gl_FragColor = vec4(col, 1.0);
        #include <fog_fragment>
        #include <colorspace_fragment>
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
