import * as THREE from 'three';

// 两套着色器水面共用的 Gerstner 位移代码与波形 uniform 构造。
// GLSL 片段约定调用方已声明 uTime、uWindScale、uWaves 三个 uniform，
// 波形公式与 waves.js 完全一致，保证浮力采样贴合

export const WAVE_UNIFORMS_GLSL = /* glsl */`
  uniform float uTime;
  uniform float uWindScale;
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
      float c = sqrt(9.8 / k);
      float f = k * (dot(dir, p) - c * uTime);
      float s = sin(f);
      float co = cos(f);
      disp += vec3(dir.x * a * co, a * s, dir.y * a * co);
      tangent += vec3(
        -dir.x * dir.x * steep * s,
        dir.x * steep * co,
        -dir.x * dir.y * steep * s
      );
      binormal += vec3(
        -dir.x * dir.y * steep * s,
        dir.y * steep * co,
        -dir.y * dir.y * steep * s
      );
    }
    outNormal = normalize(cross(binormal, tangent));
    return disp;
  }
`;

// 由波场构造波形 uniform，uWaves 需在 UniformsUtils.merge 之后单独挂载
export function buildWaveUniforms(waveField) {
  return {
    uTime: { value: 0.0 },
    uWindScale: { value: 1.0 },
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
