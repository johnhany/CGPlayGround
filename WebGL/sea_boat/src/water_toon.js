import * as THREE from 'three';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, NOISE_GLSL, buildWaveUniforms } from './water_common.js';

const SIZE = 280;       // 水面边长，单位米
const SEGMENTS = 200;   // 网格划分数，顶点位移在 GPU 完成

// 风之杖式卡通海面：顶点位移与吉卜力方案共用同一套 Gerstner 代码，
// 片元着色器按波高分硬边三档色带，背光面压暗形成两档受光；
// 浪花为域扭曲值噪声生成的旋涡图案，由波高阈值触发并随时间流动消散，
// 船周以环状遮罩叠加同源噪声，得到绕船的旋涡白沫
export function createToonWater(waveField, waterConfig) {
  const geometry = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
  geometry.rotateX(-Math.PI / 2);

  const uniforms = THREE.UniformsUtils.merge([
    THREE.UniformsLib.fog,
    {
      uDeepColor: { value: new THREE.Color(waterConfig.deep) },
      uMidColor: { value: new THREE.Color(waterConfig.mid) },
      uCrestColor: { value: new THREE.Color(waterConfig.crest) },
      uFoamColor: { value: new THREE.Color(waterConfig.foam) },
      uSkyColor: { value: new THREE.Color(waterConfig.sky) },
      uSunColor: { value: new THREE.Color(waterConfig.sun) },
      uSunDir: { value: new THREE.Vector3(0.5, 0.8, 0.2).normalize() },
      uBoatPos: { value: new THREE.Vector2(0.0, 0.0) }
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
      uniform vec3 uMidColor;
      uniform vec3 uCrestColor;
      uniform vec3 uFoamColor;
      uniform vec3 uSkyColor;
      uniform vec3 uSunColor;
      uniform vec3 uSunDir;
      uniform vec2 uBoatPos;
      uniform float uTime;

      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying float vHeight;

      ${NOISE_GLSL}

      void main() {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);

        // 硬边三档色带，边界以极窄过渡做抗锯齿
        float aa = 0.025;
        vec3 col = uDeepColor;
        col = mix(col, uMidColor, smoothstep(-aa, aa, vHeight));
        col = mix(col, uCrestColor, smoothstep(0.28 - aa, 0.28 + aa, vHeight));

        // 两档受光：背光坡面整体压暗
        float ndl = dot(N, normalize(uSunDir));
        col *= mix(0.82, 1.0, smoothstep(-0.05, 0.06, ndl));

        // 域扭曲 fbm 生成旋涡图案，随时间缓慢流动
        vec2 np = vWorldPos.xz * 0.16;
        float flow = uTime * 0.22;
        vec2 q = vec2(
          fbm(np + vec2(0.0, flow)),
          fbm(np + vec2(5.2, 1.3) - flow)
        );
        float swirl = fbm(np + 2.8 * q + vec2(flow * 0.5, -flow * 0.3));

        // 波峰触发旋涡浪花
        float crestMask = smoothstep(0.30, 0.38, vHeight);
        float foam = crestMask * smoothstep(0.46, 0.55, swirl);

        // 船周旋涡浪花：贴边环边界受噪声扰动，外侧为断续旋涡环
        float d = distance(vWorldPos.xz, uBoatPos);
        float ringPat = fbm(np * 3.1 + vec2(-flow, flow) + vec2(3.7, 8.1));
        float rim = 1.0 - smoothstep(1.3, 1.85, d + (ringPat - 0.5) * 0.7);
        float ring = smoothstep(3.4, 2.6, d) * (1.0 - smoothstep(1.9, 2.5, d));
        foam = clamp(foam + rim + ring * smoothstep(0.38, 0.52, ringPat), 0.0, 1.0);

        col = mix(col, uFoamColor, foam * 0.95);

        // 分档太阳亮斑
        vec3 L = normalize(uSunDir);
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 48.0);
        col += uSunColor * smoothstep(0.35, 0.5, spec) * 0.15;

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
