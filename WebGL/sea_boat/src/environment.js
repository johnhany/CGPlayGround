import * as THREE from 'three';
import { STYLE_CONFIG, getGradientMap } from './styles.js';
import { createClayMaterial } from './clay_material.js';

// 大气散射风格的渐变天穹。着色器末尾接入 three.js 的色调映射与输出编码，
// 使天空与水面经过同一套 ACES 管线，地平线处与雾色无缝衔接
export function createSky(scene) {
  const uniforms = {
    uTopColor: { value: new THREE.Color(STYLE_CONFIG.A.skyTop) },
    uHorizonColor: { value: new THREE.Color(STYLE_CONFIG.A.skyHorizon) },
    uSunColor: { value: new THREE.Color(0xfff3d6) },
    uSunDir: { value: new THREE.Vector3(0.5, 0.8, 0.2).normalize() },
    uSunGlow: { value: STYLE_CONFIG.A.sunGlow }
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    side: THREE.BackSide,
    depthWrite: false,
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() {
        vDir = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      uniform vec3 uTopColor;
      uniform vec3 uHorizonColor;
      uniform vec3 uSunColor;
      uniform vec3 uSunDir;
      uniform float uSunGlow;
      varying vec3 vDir;

      void main() {
        vec3 dir = normalize(vDir);
        float h = dir.y;

        vec3 col = mix(uHorizonColor, uTopColor, pow(clamp(h, 0.0, 1.0), 0.36));
        col = mix(col, uHorizonColor * 0.985, clamp(-h * 8.0, 0.0, 1.0));

        // 太阳圆盘、近距光晕与远距辉光
        float d = max(dot(dir, uSunDir), 0.0);
        col += uSunColor * uSunGlow * (pow(d, 1500.0) * 1.5 + pow(d, 50.0) * 0.15 + pow(d, 6.0) * 0.05);

        gl_FragColor = vec4(col, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `
  });

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1500, 40, 20), material);
  scene.add(mesh);
  return { uniforms, material };
}

// 用当前天穹材质生成 PMREM 环境贴图，供方案 B 与 D 的 PBR 船体做环境光照。
// 环境贴图需要线性辐射度数据，克隆材质时关闭色调映射
export function createEnvironmentMap(renderer, skyMaterial) {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tempScene = new THREE.Scene();
  const clone = skyMaterial.clone();
  clone.toneMapped = false;
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(200, 32, 16), clone);
  tempScene.add(sphere);
  const renderTarget = pmrem.fromScene(tempScene, 0.04, 0.1, 500);
  pmrem.dispose();
  sphere.geometry.dispose();
  clone.dispose();
  return renderTarget;
}

// 平行光阳光与半球环境光
export function createLights() {
  const sun = new THREE.DirectionalLight(0xfff1da, 1.7);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const cam = sun.shadow.camera;
  cam.left = -26;
  cam.right = 26;
  cam.top = 26;
  cam.bottom = -26;
  cam.near = 60;
  cam.far = 280;
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.15;

  const hemi = new THREE.HemisphereLight(0xbfe4ff, 0x2a6a8f, 0.85);
  return { sun, hemi };
}

// 按高度角与方位角更新太阳方向，同步平行光位置与天穹 uniform，
// 返回方向向量供需要太阳方向的水面着色器复用
export function updateSun(lights, skyUniforms, elevationDeg, azimuthDeg) {
  const phi = THREE.MathUtils.degToRad(90 - elevationDeg);
  const theta = THREE.MathUtils.degToRad(azimuthDeg);
  const sunDir = new THREE.Vector3().setFromSphericalCoords(1.0, phi, theta);
  lights.sun.position.copy(sunDir).multiplyScalar(150);
  skyUniforms.uSunDir.value.copy(sunDir);
  return sunDir;
}

// 低多边形积云：每朵云由若干压扁的二十面体拼成，沿风向缓慢漂移，超出边界后回绕。
// 云体保留三套材质，方案 A 与 C 下换用卡通渐变着色，方案 D 用平滑着色；
// 方案 E 下隐藏立体积云，改为轻淡的水平纸色云带
export function createClouds(scene) {
  const group = new THREE.Group();

  const gradientMap = getGradientMap();
  const matB = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    flatShading: true,
    roughness: 1.0,
    metalness: 0.0,
    emissive: 0xe8f2fb,
    emissiveIntensity: 0.32
  });
  const matA = new THREE.MeshToonMaterial({
    color: 0xffffff,
    gradientMap,
    emissive: 0xdcf0fa,
    emissiveIntensity: 0.3
  });
  const matD = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    flatShading: false,
    roughness: 0.95,
    metalness: 0.0,
    emissive: 0xe4f1fb,
    emissiveIntensity: 0.28
  });

  // H 的近景粘土云独立于远景多边形积云：几个捏圆的小团块拼接。
  const clayClouds = new THREE.Group();
  const cloudClay = createClayMaterial(0xeee6d5, { scale: 3, depth: 0.010, roughness: 0.8 });
  for (const [cx, cy, cz, size] of [[-5.2, 3.7, -5.7, 0.9], [4.2, 4.5, -7.0, 0.8]]) {
    const cluster = new THREE.Group();
    for (const [x, y, r] of [[-1.0, 0, 0.64], [-0.4, 0.25, 0.84], [0.35, 0.38, 0.92], [1.0, 0, 0.60]]) {
      const geo = new THREE.SphereGeometry(r * size, 24, 16);
      const p = geo.attributes.position;
      for (let i = 0; i < p.count; i++) {
        const dx = p.getX(i), dy = p.getY(i), dz = p.getZ(i);
        const wobble = 1 + 0.035 * Math.sin(dx * 6 + dy * 3) * Math.cos(dz * 5);
        p.setXYZ(i, dx * wobble, dy * wobble, dz * wobble);
      }
      geo.computeVertexNormals();
      const blob = new THREE.Mesh(geo, cloudClay);
      blob.scale.set(1, 0.78, 0.7);
      blob.position.set(x * size, y * size, 0);
      cluster.add(blob);
    }
    cluster.position.set(cx, cy, cz);
    clayClouds.add(cluster);
  }
  clayClouds.visible = false;
  group.add(clayClouds);

  const clusters = [];
  const blobs = [];
  for (let i = 0; i < 7; i++) {
    const cluster = new THREE.Group();
    const blobCount = 4 + Math.floor(Math.random() * 4);
    for (let b = 0; b < blobCount; b++) {
      const r = 5 + Math.random() * 8;
      const blob = new THREE.Mesh(new THREE.IcosahedronGeometry(r, 0), matB);
      blob.position.set(
        (Math.random() - 0.5) * 30,
        (Math.random() - 0.5) * 4,
        (Math.random() - 0.5) * 14
      );
      blob.scale.y = 0.42 + Math.random() * 0.18;
      blobs.push(blob);
      cluster.add(blob);
    }

    const angle = Math.random() * Math.PI * 2;
    const radius = 100 + Math.random() * 230;
    cluster.position.set(
      Math.cos(angle) * radius,
      50 + Math.random() * 50,
      Math.sin(angle) * radius
    );
    clusters.push({ g: cluster, speed: 1.5 + Math.random() * 1.5 });
    group.add(cluster);
  }

  // 浮世绘云纹：canvas 绘制低对比度的水平云带，
  // 以精灵方式始终面向相机
  const cloudTexture = createUkiyoeCloudTexture();
  const spriteGroup = new THREE.Group();
  const spriteDefs = [
    { pos: [-150, 30, -320], scale: [150, 52], rot: 0.02 },
    { pos: [-300, 42, -240], scale: [120, 44], rot: -0.03 },
    { pos: [-70, 22, -350], scale: [96, 34], rot: 0.04 }
  ];
  const sprites = [];
  for (const def of spriteDefs) {
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: cloudTexture,
      transparent: true,
      depthWrite: false,
      fog: false,
      toneMapped: false,
      rotation: def.rot
    }));
    sprite.position.set(...def.pos);
    sprite.scale.set(def.scale[0] * 0.65, def.scale[1] * 0.45, 1);
    sprites.push({ s: sprite, speed: 0.8 + Math.random() * 0.8 });
    spriteGroup.add(sprite);
  }
  group.add(spriteGroup);

  scene.add(group);

  function update(dt) {
    for (const c of clusters) {
      c.g.position.x += c.speed * dt;
      if (c.g.position.x > 420) {
        c.g.position.x = -420;
      }
    }
    for (const item of sprites) {
      item.s.position.x += item.speed * dt;
      if (item.s.position.x > 260) {
        item.s.position.x = -260;
      }
    }
  }

  function setStyle(style) {
    const ukiyoe = style === 'E';
    const clay = style === 'H';
    clayClouds.visible = clay;
    const mat = style === 'B' ? matB : (style === 'D' ? matD : matA);
    for (const blob of blobs) {
      blob.material = mat;
      blob.visible = !ukiyoe && !clay;
    }
    const s = STYLE_CONFIG[style].cloudScale;
    for (const c of clusters) {
      c.g.scale.setScalar(s);
      c.g.visible = !ukiyoe && !clay;
    }
    spriteGroup.visible = ukiyoe;
  }

  return { group, update, setStyle };
}

// 纸色云带：少量平缓贝塞尔曲线配淡墨线，保留天空留白。
function createUkiyoeCloudTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 192;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#f7efd9';
  ctx.strokeStyle = '#b3aa8b';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(18, 124);
  ctx.bezierCurveTo(90, 114, 100, 87, 170, 95);
  ctx.bezierCurveTo(205, 50, 260, 65, 280, 91);
  ctx.bezierCurveTo(350, 80, 366, 113, 490, 122);
  ctx.bezierCurveTo(380, 137, 115, 142, 18, 124);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// 将风格配置应用到天空、雾、背景、光照与云
export function applyEnvironmentStyle(name, targets) {
  const config = STYLE_CONFIG[name];
  targets.skyMaterial.toneMapped = name !== 'E';
  targets.skyUniforms.uTopColor.value.setHex(config.skyTop);
  targets.skyUniforms.uHorizonColor.value.setHex(config.skyHorizon);
  targets.skyUniforms.uSunGlow.value = config.sunGlow;
  targets.scene.fog.color.setHex(config.fog);
  targets.scene.background.setHex(config.fog);
  targets.lights.hemi.color.setHex(config.hemiSky ?? 0xbfe4ff);
  targets.lights.hemi.groundColor.setHex(config.hemiGround ?? 0x2a6a8f);
  const studio = name === 'H';
  targets.lights.sun.shadow.normalBias = studio ? 0.025 : 0.15;
  targets.lights.sun.shadow.radius = studio ? 10 : 1;
  const shadowCam = targets.lights.sun.shadow.camera;
  const extent = studio ? 16 : 26;
  shadowCam.left = shadowCam.bottom = -extent;
  shadowCam.right = shadowCam.top = extent;
  shadowCam.updateProjectionMatrix();
  targets.lights.sun.intensity = config.sunIntensity;
  targets.lights.hemi.intensity = config.hemiIntensity;
  targets.clouds.setStyle(name);
}
