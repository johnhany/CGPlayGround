import * as THREE from 'three';

// 各套着色器水面共用的 Gerstner 位移代码与波形 uniform 构造。
// GLSL 片段约定调用方已声明 uTime、uWindScale、uWaves 三个 uniform，
// 波形公式与 waves.js 完全一致，保证浮力采样贴合

export const WAVE_UNIFORMS_GLSL = /* glsl */`
  uniform float uTime;
  uniform float uWindScale;
  uniform float uSharp;     // 谐波锐化系数
  uniform float uLean;      // 波峰前倾系数
  uniform float uEnv;       // 主导浪波群包络
  uniform vec4 uWaves[5];   // xy 为波向，z 为波陡，w 为波长
`;

export const GERSTNER_GLSL = /* glsl */`
  vec3 gerstnerDisp(vec2 p, out vec3 outNormal) {
    vec3 disp = vec3(0.0);
    vec3 tangent = vec3(1.0, 0.0, 0.0);
    vec3 binormal = vec3(0.0, 0.0, 1.0);
    for (int i = 0; i < 5; i++) {
      vec4 w = uWaves[i];
      vec2 dir = w.xy;
      float k = 6.2831853 / w.w;
      float steep = w.z * uWindScale;
      float a = steep / k;
      if (i == 0) { a *= uEnv; }
      float c = sqrt(9.8 / k);
      float f = k * (dot(dir, p) - c * uTime);
      float s = sin(f);
      float sh = s - uSharp * sin(2.0 * f) * 0.5;   // 锐化波形
      float hz = cos(f) + uLean * sh;               // 水平位移含波峰前倾
      disp += vec3(dir.x * a * hz, a * sh, dir.y * a * hz);
      // 位移对 x 与 z 的偏导累加，兼容任意波形改造
      float dsh = cos(f) - uSharp * cos(2.0 * f);
      float dhz = -s + uLean * dsh;
      tangent += dir.x * a * k * vec3(dir.x * dhz, dsh, dir.y * dhz);
      binormal += dir.y * a * k * vec3(dir.x * dhz, dsh, dir.y * dhz);
    }
    outNormal = normalize(cross(binormal, tangent));
    return disp;
  }
`;

// 由波场构造波形 uniform，uWaves 与随时间变化的 uEnv 需在每帧更新
export function buildWaveUniforms(waveField) {
  return {
    uTime: { value: 0.0 },
    uWindScale: { value: 1.0 },
    uSharp: { value: waveField.sharp },
    uLean: { value: waveField.lean },
    uEnv: { value: 1.0 },
    uWaves: {
      value: waveField.waves.map(
        (w) => new THREE.Vector4(w.dx, w.dz, w.steepness, w.wavelength)
      )
    }
  };
}

// 片元着色器共用的值噪声与三阶 fbm
export const NOISE_GLSL = /* glsl */`
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
`;
