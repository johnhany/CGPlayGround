import * as THREE from 'three';

// 固定在物体坐标上的稀疏指腹压痕，三向投影避免 UV 拉伸和接缝。
// 高光由真实 PBR 光照产生，凹凸由高度梯度产生，不把随机数直接加到法线上。
const CLAY_GLSL = /* glsl */`
  varying vec3 vClayPosition;
  varying vec3 vClayNormal;
  uniform float uClayScale;
  uniform float uClayDepth;
  float clayHash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float thumbPrint(vec2 p) {
    vec2 cell = floor(p);
    float seed = clayHash(cell);
    vec2 center = vec2(clayHash(cell + 4.7), clayHash(cell + 9.1));
    vec2 q = fract(p) - vec2(0.5) - (center - 0.5) * 0.14;
    float a = clayHash(cell + 13.1) * 6.2831853;
    q = mat2(cos(a), -sin(a), sin(a), cos(a)) * q;
    float r = length(q * vec2(1.0, 1.28));
    float mask = (1.0 - smoothstep(0.20, 0.34, r)) * smoothstep(0.68, 0.86, seed);
    // 指纹圆心位于压痕外侧，只留下弧形纹路，避免靶心式同心圆。
    float arc = length((q + vec2(0.38, 0.24)) * vec2(0.8, 1.1));
    float ridgeAA = 1.0 - smoothstep(2.0, 3.14, fwidth(arc) * 100.0);
    float ridges = sin(arc * 100.0 + q.x * 5.0) * 0.18 * ridgeAA;
    float dish = -0.7 * exp(-r * r * 25.0);
    return (dish + ridges) * mask;
  }
  float clayHeight(vec3 p, vec3 n) {
    vec3 w = pow(abs(n), vec3(6.0));
    w /= max(dot(w, vec3(1.0)), 0.0001);
    p *= uClayScale;
    float h = thumbPrint(p.yz) * w.x + thumbPrint(p.xz + 7.3) * w.y + thumbPrint(p.xy + 15.7) * w.z;
    return h * uClayDepth;
  }
`;

export function createClayMaterial(color, options = {}) {
  const { scale = 3.0, depth = 0.025, ...physical } = options;
  const material = new THREE.MeshPhysicalMaterial({
    color, roughness: 0.72, metalness: 0, specularIntensity: 0.65,
    envMapIntensity: 0.55, ...physical
  });
  const uniforms = {
    uClayScale: { value: scale }, uClayDepth: { value: depth }
  };
  material.userData.clayUniforms = uniforms;
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = 'varying vec3 vClayPosition;\nvarying vec3 vClayNormal;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>', `
      #include <begin_vertex>
      vClayPosition = transformed;
      vClayNormal = normalize(objectNormal);
    `);
    shader.fragmentShader = CLAY_GLSL + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      float pigment = sin(vClayPosition.x * 3.1 + sin(vClayPosition.z * 2.7))
        * sin(vClayPosition.y * 3.7 + vClayPosition.z * 1.9);
      diffuseColor.rgb *= 1.0 + 0.025 * pigment;
    `);
    shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>', `
      #include <roughnessmap_fragment>
      roughnessFactor = clamp(roughnessFactor + pigment * 0.035, 0.48, 0.95);
    `);
    shader.fragmentShader = shader.fragmentShader.replace('#include <normal_fragment_maps>', `
      #include <normal_fragment_maps>
      float clayH = clayHeight(vClayPosition, normalize(vClayNormal));
      vec3 clayDx = dFdx(-vViewPosition);
      vec3 clayDy = dFdy(-vViewPosition);
      vec3 clayR1 = cross(clayDy, normal);
      vec3 clayR2 = cross(normal, clayDx);
      float clayDet = dot(clayDx, clayR1);
      if (abs(clayDet) > 1e-10) {
        vec3 grad = sign(clayDet) * (dFdx(clayH) * clayR1 + dFdy(clayH) * clayR2);
        normal = normalize(abs(clayDet) * normal - grad);
      }
    `);
  };
  material.customProgramCacheKey = () => 'clay-thumbprint-pbr-v1';
  return material;
}
