import * as THREE from 'three';
import { createClayMaterial } from './clay_material.js';
import { WAVE_UNIFORMS_GLSL, GERSTNER_GLSL, buildWaveUniforms } from './water_common.js';

export const CLAY_RADIUS = 9.4;

// 边缘被压平接入粘土底座。船底采样点都在半径 6.8 内，浮力无需另加高度。
export function sampleClaySea(field, x, z, out) {
  field.sample(x, z, out);
  const fade = 1 - THREE.MathUtils.smoothstep(Math.hypot(x, z), 6.8, CLAY_RADIUS);
  out.dx *= fade; out.dy *= fade; out.dz *= fade;
  return out;
}

export function createClayWater(waveField, config) {
  const group = new THREE.Group();
  const geometry = new THREE.PlaneGeometry(20, 20, 180, 180);
  geometry.rotateX(-Math.PI / 2);
  const uniforms = {
    ...buildWaveUniforms(waveField),
    uSunDir: { value: new THREE.Vector3() }, // 与主场景太阳刷新接口一致
    uBoatHeading: { value: 0 }
  };
  const material = createClayMaterial(config.deep, { scale: 1.05, depth: 0.030, roughness: 0.68 });
  const clayCompile = material.onBeforeCompile;
  material.onBeforeCompile = shader => {
    clayCompile(shader);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = WAVE_UNIFORMS_GLSL + GERSTNER_GLSL + /* glsl */`
      vec3 clayDisp(vec2 p) {
        vec3 n;
        return gerstnerDisp(p, n) * (1.0 - smoothstep(6.8, 9.4, length(p)));
      }
    ` + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <beginnormal_vertex>', `
      #include <beginnormal_vertex>
      vec3 clayD = clayDisp(position.xz);
      vec3 tx = vec3(0.04, 0.0, 0.0) + clayDisp(position.xz + vec2(0.02, 0.0)) - clayDisp(position.xz - vec2(0.02, 0.0));
      vec3 tz = vec3(0.0, 0.0, 0.04) + clayDisp(position.xz + vec2(0.0, 0.02)) - clayDisp(position.xz - vec2(0.0, 0.02));
      objectNormal = normalize(cross(tz, tx));
    `).replace('#include <begin_vertex>', '#include <begin_vertex>\ntransformed += clayD;')
      .replace('vClayPosition = transformed;', 'vClayPosition = position;');
    shader.fragmentShader = 'uniform float uBoatHeading;\n' + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
      if (length(vClayPosition.xz) > 9.4) discard;
      #include <color_fragment>
      // 船底接触暗部只占水线附近，不再用一张悬空的阴影贴片。
      vec2 contact = mat2(cos(uBoatHeading), -sin(uBoatHeading), sin(uBoatHeading), cos(uBoatHeading)) * vClayPosition.xz;
      float ao = exp(-dot(contact / vec2(0.82, 2.0), contact / vec2(0.82, 2.0)) * 2.0);
      diffuseColor.rgb *= 1.0 - 0.28 * ao;
    `);
  };
  material.customProgramCacheKey = () => 'clay-sea-pbr-v1';
  const surface = new THREE.Mesh(geometry, material);
  surface.receiveShadow = true;
  surface.frustumCulled = false;
  group.add(surface);

  // 可见的圆润厚边和桌面投影，让海洋读作真实捏制的微缩布景。
  const edgeMaterial = createClayMaterial(config.deep, { scale: 2, depth: 0.025 });
  const profile = [[0, -0.78], [9.1, -0.78], [9.45, -0.72], [9.65, -0.50],
    [9.65, -0.30], [9.52, -0.08], [9.4, 0]].map(([x,y]) => new THREE.Vector2(x,y));
  const edge = new THREE.Mesh(new THREE.LatheGeometry(profile, 128), edgeMaterial);
  edge.castShadow = true;
  edge.receiveShadow = true;
  group.add(edge);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ color: 0xd8c5ae, roughness: 0.94 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.81;
  floor.receiveShadow = true;
  group.add(floor);

  // 立体白色粘土条，模拟手工搓出的短浪脊，拥有真实厚度与接触阴影。
  const foamMaterial = createClayMaterial(0xe9e6d1, { scale: 4, depth: 0.012, roughness: 0.77 });
  const ropes = [
    { x: -5.3, z: -2.4, length: 3.0, radius: 0.10 },
    { x: -2.8, z: -5.3, length: 2.5, radius: 0.085 },
    { x: 2.5, z: -4.8, length: 3.8, radius: 0.095 },
    { x: 5.2, z: -0.8, length: 2.1, radius: 0.09 },
    { x: -4.3, z: 3.4, length: 2.6, radius: 0.10 },
    { x: 3.2, z: 4.4, length: 3.0, radius: 0.09 }
  ].map(def => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(33 * 9 * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    const indices = [];
    for (let i = 0; i < 32; i++) for (let j = 0; j < 8; j++) {
      const a = i * 9 + j, b = a + 9;
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
    geo.setIndex(indices);
    const mesh = new THREE.Mesh(geo, foamMaterial);
    mesh.castShadow = true; mesh.receiveShadow = true; mesh.frustumCulled = false;
    group.add(mesh);
    return { ...def, mesh };
  });
  const out = {};
  function update() {
    uniforms.uTime.value = waveField.time;
    uniforms.uWindScale.value = waveField.windScale * waveField.amplitudeScale;
    uniforms.uEnv.value = waveField.envelope;
    for (const rope of ropes) {
      const attr = rope.mesh.geometry.attributes.position;
      for (let i = 0; i <= 32; i++) {
        const u = i / 32;
        const z = rope.z + (u - 0.5) * rope.length;
        const x = rope.x + Math.sin(u * Math.PI) * 0.38 + 0.035 * Math.sin(u * Math.PI * 5 + rope.z);
        sampleClaySea(waveField, x, z, out);
        const radius = rope.radius * Math.pow(Math.sin(Math.PI * u), 0.5)
          * (0.94 + 0.06 * Math.sin(u * Math.PI * 7 + rope.x));
        for (let j = 0; j <= 8; j++) {
          const a = j / 8 * Math.PI * 2;
          attr.setXYZ(i * 9 + j, x + out.dx + Math.cos(a) * radius,
            out.dy + radius * 0.55 + Math.sin(a) * radius, z + out.dz);
        }
      }
      attr.needsUpdate = true;
      rope.mesh.geometry.computeVertexNormals();
      // 收尖处的圆环退化为一个点，为两端补明确的切向法线，避免零法线。
      const normals = rope.mesh.geometry.attributes.normal;
      const center = index => new THREE.Vector3().fromBufferAttribute(attr, index)
        .add(new THREE.Vector3().fromBufferAttribute(attr, index + 4)).multiplyScalar(0.5);
      const startNormal = center(0).sub(center(9)).normalize();
      const endNormal = center(32 * 9).sub(center(31 * 9)).normalize();
      for (let j = 0; j <= 8; j++) {
        normals.setXYZ(j, startNormal.x, startNormal.y, startNormal.z);
        normals.setXYZ(32 * 9 + j, endNormal.x, endNormal.y, endNormal.z);
      }
      normals.needsUpdate = true;
    }
  }
  function setEnvironmentMap(texture) {
    for (const mat of [material, edgeMaterial, foamMaterial]) mat.envMap = texture;
  }
  update();
  return { mesh: group, surface, material, uniforms, ropes, update, setEnvironmentMap };
}
