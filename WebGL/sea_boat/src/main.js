import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import Stats from 'three/addons/libs/stats.module.js';
import { WaveField } from './waves.js';
import { createWater } from './water.js';
import { createGhibliWater } from './water_ghibli.js';
import { createToonWater } from './water_toon.js';
import { createRealisticWater } from './water_realistic.js';
import { createUkiyoeWater } from './water_ukiyoe.js';
import { buildBoat, updateBoat } from './boat.js';
import { createSky, createEnvironmentMap, createClouds, createLights, updateSun, applyEnvironmentStyle } from './environment.js';
import { STYLE_CONFIG } from './styles.js';

const params = {
  画面风格: 'A',
  风浪强度: 1.0,
  波速: 1.0,
  暂停: false,
  太阳高度角: 55,
  太阳方位角: 125,
  曝光: 1.12,
  船体朝向: 35,
  自动环绕: true,
  水面线框: false
};

// ---------------------------------------------------------------------------
// 渲染器、场景与相机
// ---------------------------------------------------------------------------

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = params.曝光;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe8f5fa);
scene.fog = new THREE.Fog(0xe8f5fa, 55, 245);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 4000);
camera.position.set(11, 5.5, 13.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 1.2, 0);
controls.minDistance = 6;
controls.maxDistance = 60;
controls.maxPolarAngle = 1.45;
controls.autoRotate = params.自动环绕;
controls.autoRotateSpeed = 0.5;

const stats = new Stats();
document.body.appendChild(stats.dom);

// ---------------------------------------------------------------------------
// 场景内容
// ---------------------------------------------------------------------------

const waveField = new WaveField();
waveField.windScale = params.风浪强度;

const sky = createSky(scene);
const skyUniforms = sky.uniforms;
const lights = createLights();
scene.add(lights.sun, lights.hemi);

const waterGhibli = createGhibliWater(waveField, STYLE_CONFIG.A.water);
const waterFlat = createWater(waveField);
const waterToon = createToonWater(waveField, STYLE_CONFIG.C.water);
const waterRealistic = createRealisticWater(waveField, STYLE_CONFIG.D.water);
const waterUkiyoe = createUkiyoeWater(waveField, STYLE_CONFIG.E.water);
scene.add(waterGhibli.mesh, waterFlat.mesh, waterToon.mesh, waterRealistic.mesh, waterUkiyoe.mesh);

const boat = buildBoat();
boat.heading = THREE.MathUtils.degToRad(params.船体朝向);
scene.add(boat.group, boat.foamRing, boat.shadowBlob);

const clouds = createClouds(scene);

// 太阳方向同时驱动平行光、天穹与各着色器水面的高光
let sunDir = new THREE.Vector3();
function refreshSun() {
  sunDir = updateSun(lights, skyUniforms, params.太阳高度角, params.太阳方位角);
  waterGhibli.uniforms.uSunDir.value.copy(sunDir);
  waterToon.uniforms.uSunDir.value.copy(sunDir);
  waterRealistic.uniforms.uSunDir.value.copy(sunDir);
}
refreshSun();

// 以当前天穹生成 PMREM 环境贴图，供 PBR 船体做环境光照
let envRenderTarget = null;
function refreshEnvironmentMap() {
  if (envRenderTarget !== null) {
    envRenderTarget.dispose();
  }
  envRenderTarget = createEnvironmentMap(renderer, sky.material);
  boat.setEnvironmentMap(envRenderTarget.texture);
}
refreshEnvironmentMap();

// ---------------------------------------------------------------------------
// 画面风格切换
// ---------------------------------------------------------------------------

const styleButtons = document.querySelectorAll('#style-switch button');
let styleController = null;

function setStyle(name) {
  params.画面风格 = name;
  waveField.amplitudeScale = STYLE_CONFIG[name].waveScale;

  waterGhibli.mesh.visible = name === 'A';
  waterFlat.mesh.visible = name === 'B';
  waterToon.mesh.visible = name === 'C';
  waterRealistic.mesh.visible = name === 'D';
  waterUkiyoe.mesh.visible = name === 'E';
  waterFlat.material.wireframe = params.水面线框 && name === 'B';

  boat.applyStyle(name);
  applyEnvironmentStyle(name, { scene, lights, skyUniforms, clouds });
  refreshEnvironmentMap();

  for (const btn of styleButtons) {
    btn.classList.toggle('active', btn.dataset.style === name);
  }
  if (styleController !== null) {
    styleController.updateDisplay();
  }
}

for (const btn of styleButtons) {
  btn.addEventListener('click', () => setStyle(btn.dataset.style));
}
window.addEventListener('keydown', (event) => {
  if (event.code === 'Digit1') {
    setStyle('A');
  } else if (event.code === 'Digit2') {
    setStyle('B');
  } else if (event.code === 'Digit3') {
    setStyle('C');
  } else if (event.code === 'Digit4') {
    setStyle('D');
  } else if (event.code === 'Digit5') {
    setStyle('E');
  }
});

// ---------------------------------------------------------------------------
// 控制面板
// ---------------------------------------------------------------------------

const gui = new GUI({ title: '海面小船' });

const styleFolder = gui.addFolder('风格');
styleController = styleFolder.add(params, '画面风格', {
  '方案 A · 吉卜力晴海': 'A',
  '方案 B · 低多边形海面': 'B',
  '方案 C · 风之杖卡通海': 'C',
  '方案 D · 半写实明亮海': 'D',
  '方案 E · 浮世绘海浪': 'E'
}).onChange(setStyle);

const seaFolder = gui.addFolder('海面');
seaFolder.add(params, '风浪强度', 0.2, 2.0, 0.05).onChange((v) => { waveField.windScale = v; });
seaFolder.add(params, '波速', 0.2, 2.0, 0.05).onChange((v) => { waveField.speedScale = v; });
seaFolder.add(params, '暂停');

const lightFolder = gui.addFolder('光照');
lightFolder.add(params, '太阳高度角', 10, 85, 1).onChange(refreshSun);
lightFolder.add(params, '太阳方位角', 0, 360, 1).onChange(refreshSun);
lightFolder.add(params, '曝光', 0.6, 1.6, 0.01).onChange((v) => { renderer.toneMappingExposure = v; });

const sceneFolder = gui.addFolder('场景');
sceneFolder.add(params, '船体朝向', 0, 360, 1).onChange((v) => { boat.heading = THREE.MathUtils.degToRad(v); });
sceneFolder.add(params, '自动环绕').onChange((v) => { controls.autoRotate = v; });
sceneFolder.add(params, '水面线框').onChange((v) => { waterFlat.material.wireframe = v; });

// ---------------------------------------------------------------------------
// 视口尺寸与主循环
// ---------------------------------------------------------------------------

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

setStyle(params.画面风格);

let lastTime = performance.now();

renderer.setAnimationLoop(() => {
  const now = performance.now();
  let delta = (now - lastTime) / 1000;
  lastTime = now;
  delta = Math.min(delta, 1 / 30);

  if (!params.暂停) {
    waveField.time += delta * waveField.speedScale;
    if (params.画面风格 === 'A') {
      waterGhibli.update();
    } else if (params.画面风格 === 'C') {
      waterToon.update();
    } else if (params.画面风格 === 'D') {
      waterRealistic.update();
    } else if (params.画面风格 === 'E') {
      waterUkiyoe.update();
    } else {
      waterFlat.update();
    }
    updateBoat(boat, waveField, delta);
    clouds.update(delta);
    waterToon.uniforms.uBoatPos.value.set(boat.group.position.x, boat.group.position.z);
  }

  controls.update();
  renderer.render(scene, camera);
  stats.update();
});
