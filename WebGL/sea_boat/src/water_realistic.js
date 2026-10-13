import * as THREE from 'three';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, NOISE_GLSL, buildWaveUniforms } from './water_common.js';

const SIZE = 280;       // 水面边长，单位米
const SEGMENTS = 200;   // 网格划分数，顶点位移在 GPU 完成

// 半写实明亮海水面：在共享 Gerstner 位移之上叠加完整的水面光学。
// 片元着色器包含：视线角度与波高联合决定的水体吸收渐变、施密特菲涅耳
// 加权的程序化天空反射、微法线扰动驱动的太阳高光闪烁、逆光波峰的
// 伪次表面散射透光，以及两张不同流向噪声混合的波峰与拖尾双通道泡沫
export function createRealisticWater(waveField, waterConfig) {
  const geometry = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
  geometry.rotateX(-Math.PI / 2);

  const uniforms = THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog,
    {
      uDeepColor: { value: new THREE.Color(waterConfig.deep) },
      uShallowColor: { value: new THREE.Color(waterConfig.shallow) },
      uSssColor: { value: new THREE.Color(waterConfig.sss) },
      uFoamColor: { value: new THREE.Color(waterConfig.foam) },
      uSkyColor: { value: new THREE.Color(waterConfig.sky) },
      uHorizonColor: { value: new THREE.Color(waterConfig.horizon) },
      uSunColor: { value: new THREE.Color(waterConfig.sun) },
      uSunDir: { value: new THREE.Vector3(0.5, 0.8, 0.2).normalize() }
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
      varying vec3 vNormalW;
      varying float vHeight;

      void main() {
        vec3 base = position;
        vec3 nrm;
        vec3 pos = base + gerstnerDisp(base.xz, nrm);

        vHeight = pos.y;
        vWorldPos = pos;
        vNormalW = nrm;

        vec4 mvPosition = viewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */`
      #include <fog_pars_fragment>

      uniform vec3 uDeepColor;
      uniform vec3 uShallowColor;
      uniform vec3 uSssColor;
      uniform vec3 uFoamColor;
      uniform vec3 uSkyColor;
      uniform vec3 uHorizonColor;
      uniform vec3 uSunColor;
      uniform vec3 uSunDir;
      uniform float uTime;

      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying float vHeight;

      ${NOISE_GLSL}

      void main() {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);
        vec3 L = normalize(uSunDir);
        vec2 p = vWorldPos.xz;
        float t = uTime;

        // 三组交叉行波扰动微法线，驱动太阳高光闪烁
        float w1 = sin(dot(p, vec2(0.62, 0.78)) * 2.3 - t * 2.1);
        float w2 = sin(dot(p, vec2(-0.52, 0.85)) * 3.1 - t * 2.7);
        float w3 = sin(dot(p, vec2(0.9, -0.44)) * 2.6 + t * 1.8);
        N = normalize(N + vec3(w1 + w3 * 0.6, 0.0, w2 - w1 * 0.4) * 0.055);

        // 视线角度主导水体吸收，波高为次要项：俯视深处浓郁，
        // 波峰仅在逆光时经伪次表面散射透亮，不整体变浅
        float upFacing = clamp(dot(V, vec3(0.0, 1.0, 0.0)), 0.0, 1.0);
        float thin = clamp(0.45 + vHeight * 0.9, 0.0, 1.0);
        float absorption = clamp(upFacing * 1.4 - thin * 0.25, 0.0, 1.0);
        vec3 body = mix(uShallowColor, uDeepColor, absorption);

        // 光照：直射阳光加天空环境
        float ndl = max(dot(N, L), 0.0);
        vec3 ambient = mix(uHorizonColor, uSkyColor, N.y * 0.5 + 0.5) * 0.55;
        vec3 col = body * (ambient + uSunColor * (0.25 + 0.85 * ndl));

        // 施密特菲涅耳加权的程序化天空反射，反射色随仰角快速转入天穹蓝
        float fres = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
        vec3 R = reflect(-V, N);
        vec3 skyRef = mix(uHorizonColor, uSkyColor, pow(clamp(R.y, 0.0, 1.0), 0.30));
        skyRef += uSunColor * pow(max(dot(R, L), 0.0), 60.0) * 0.25;
        col = mix(col, skyRef, fres * 0.62);

        // 太阳高光闪烁
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 90.0);
        col += uSunColor * spec * 1.6 * (0.2 + 0.8 * fres);

        // 伪次表面散射：逆光方向的薄波峰透出青绿亮色
        float sss = smoothstep(0.12, 0.5, vHeight) * pow(max(dot(V, -L), 0.0), 3.0);
        col += uSssColor * sss * 0.45;

        // 双通道泡沫：两张不同流向的噪声混合，波峰阈值触发，
        // 拖尾通道以更低高度与更松阈值模拟浪峰过后残留的白沫
        float f1 = fbm(p * 0.32 + vec2(t * 0.14, t * 0.09));
        float f2 = fbm(p * 0.5 + vec2(-t * 0.11, t * 0.15) + 7.3);
        float field = mix(f1, f2, 0.5);
        float crestF = smoothstep(0.38, 0.60, vHeight);
        float foam = smoothstep(0.74 - crestF * 0.30, 0.84 - crestF * 0.30, field);
        float trail = smoothstep(0.63, 0.72, field) * smoothstep(0.18, 0.34, vHeight);
        foam = clamp(foam + trail * 0.45, 0.0, 1.0);
        col = mix(col, uFoamColor, foam * 0.92);

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
