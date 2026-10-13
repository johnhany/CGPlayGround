import * as THREE from 'three';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, buildWaveUniforms } from './water_common.js';

const SIZE = 280;       // 水面边长，单位米
const SEGMENTS = 200;   // 网格划分数，顶点位移在 GPU 完成，可用较密网格

// 吉卜力式晴海水面：顶点着色器做 Gerstner 位移并求解析法线，
// 片元着色器按波高做三段色渐变，波峰处以等高线式细白描边表现浪花，
// 掠射角叠加微弱天空反射
export function createGhibliWater(waveField, waterConfig) {
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
      uniform vec3 uMidColor;
      uniform vec3 uCrestColor;
      uniform vec3 uFoamColor;
      uniform vec3 uSkyColor;
      uniform vec3 uSunColor;
      uniform vec3 uSunDir;

      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying float vHeight;

      void main() {
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(cameraPosition - vWorldPos);

        // 高度驱动的三段色渐变，阈值按方案 A 的平缓波形标定
        vec3 col = mix(uDeepColor, uMidColor, smoothstep(-0.26, 0.0, vHeight));
        col = mix(col, uCrestColor, smoothstep(0.18, 0.40, vHeight));

        // 波峰等高线式细白描边，两道宽度不同
        float band = 1.0 - smoothstep(0.012, 0.04, abs(vHeight - 0.42));
        float band2 = (1.0 - smoothstep(0.006, 0.02, abs(vHeight - 0.26))) * 0.35;
        float foam = clamp(band + band2, 0.0, 1.0);
        col = mix(col, uFoamColor, foam * 0.85);

        // 掠射角处微弱的天空反射
        float fres = pow(1.0 - max(dot(V, N), 0.0), 3.0);
        col = mix(col, uSkyColor, fres * 0.16);

        // 柔和的太阳亮斑
        vec3 L = normalize(uSunDir);
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 60.0);
        col += uSunColor * spec * 0.22;

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
