import * as THREE from 'three';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// 水彩纸本后处理：对渲染结果做手绘抖动、湿画法柔化、颜料沉积边缘加深、
// 亮度分区量化（量化边界用纸纹扰动生成有机色块）、颜料沉降颗粒、
// 高光留白与纸底罩染。全部效果叠加后仍经 OutputPass 做色调映射与输出编码
export function createWatercolorPass() {
  const shader = {
    name: 'WatercolorPass',
    uniforms: {
      tDiffuse: { value: null },
      uTime: { value: 0.0 },
      uResolution: { value: new THREE.Vector2(1.0, 1.0) },
      uPaperColor: { value: new THREE.Color(0xf4efe2) }
    },
    vertexShader: /* glsl */`
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      uniform sampler2D tDiffuse;
      uniform float uTime;
      uniform vec2 uResolution;
      uniform vec3 uPaperColor;
      varying vec2 vUv;

      float hash12(vec2 p) {
        vec3 p3 = fract(vec3(p.xyx) * 0.1031);
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.x + p3.y) * p3.z);
      }

      float vnoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(
          mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
          mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x),
          u.y
        );
      }

      float fbm(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        for (int i = 0; i < 3; i++) {
          v += a * vnoise(p);
          p = p * 2.13 + vec2(11.7, 5.3);
          a *= 0.5;
        }
        return v;
      }

      float lum(vec3 c) {
        return dot(c, vec3(0.299, 0.587, 0.114));
      }

      void main() {
        // 纸面：低频组织与高频纤维
        float scale = uResolution.y / 900.0;
        float paperLo = fbm(vUv * 42.0 * scale + 7.3);
        float paperHi = vnoise(vUv * uResolution * 0.22);

        // 手绘抖动：采样坐标随低频噪声偏移
        vec2 wob = (vec2(fbm(vUv * 7.0 + 3.1), fbm(vUv * 7.0 + 9.7)) - 0.5) * 0.0045;
        vec2 suv = vUv + wob;

        vec3 col = texture2D(tDiffuse, suv).rgb;
        vec2 px = 2.0 / uResolution;
        vec3 blur = (
          texture2D(tDiffuse, suv + vec2(px.x, 0.0)).rgb +
          texture2D(tDiffuse, suv - vec2(px.x, 0.0)).rgb +
          texture2D(tDiffuse, suv + vec2(0.0, px.y)).rgb +
          texture2D(tDiffuse, suv - vec2(0.0, px.y)).rgb
        ) * 0.25;
        col = mix(col, blur, 0.22);   // 湿画法柔化

        // 颜料沉积：中心比周边暗的位置加深
        float dl = lum(blur) - lum(col);
        col *= 1.0 - clamp(dl * 7.0, 0.0, 1.0) * 0.40;

        // 亮度分区量化，量化边界用纸纹扰动生成有机色块
        float l0 = lum(col);
        float q = floor(l0 * 7.0 + paperLo * 0.45) / 7.0;
        col *= mix(l0, q, 0.85) / max(l0, 0.0001);

        // 颜料沉降与纸纹颗粒
        float gran = (paperLo - 0.5) * 0.18 + (paperHi - 0.5) * 0.12;
        col *= 1.0 + gran;

        // 高光留白：亮部推向纸色
        float l1 = lum(col);
        col = mix(col, uPaperColor, smoothstep(0.82, 1.04, l1 + paperLo * 0.08) * 0.55);

        // 纸底罩染
        col = mix(col, uPaperColor, 0.05);

        gl_FragColor = vec4(col, 1.0);
      }
    `
  };

  const pass = new ShaderPass(shader);
  pass.uniforms.uResolution.value.set(
    window.innerWidth * 2.0,
    window.innerHeight * 2.0
  );
  return pass;
}
