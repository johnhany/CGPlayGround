import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPixelatedPass } from 'three/addons/postprocessing/RenderPixelatedPass.js';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import Stats from 'three/addons/libs/stats.module.js';
import { WaveField } from './waves.js';
import { StopMotionClock } from './stop_motion.js';
import { createWater } from './water.js';
import { createGhibliWater } from './water_ghibli.js';
import { createToonWater } from './water_toon.js';
import { createRealisticWater } from './water_realistic.js';
import { createUkiyoeWater } from './water_ukiyoe.js';
import { createClayWater } from './water_clay.js';
import { createInkWater } from './water_ink.js';
import { FoamDecals } from './foam_decals.js';
import { WaveClaws } from './wave_claws.js';
import { createWatercolorPass } from './watercolor_pass.js';
import { buildBoat, updateBoat } from './boat.js';
import { createSky, createEnvironmentMap, createClouds, createLights, updateSun, applyEnvironmentStyle } from './environment.js';
import { STYLE_CONFIG } from './styles.js';

// 可复现的风格对比入口，例如 ?style=E&time=0&paused=1。
const previewQuery = new URLSearchParams(window.location.search);
const initialStyle = previewQuery.get('style');
const initialTime = Number(previewQuery.get('time') || 0);
const params = {
  画面风格: Object.hasOwn(STYLE_CONFIG, initialStyle) ? initialStyle : 'A',
  像素尺寸: 5,
  风浪强度: 1.0,
  波速: 1.0,
  暂停: previewQuery.get('paused') === '1',
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
// 模块版本错配时（浏览器缓存了旧配置文件）降级跳过，避免启动崩溃
const waterWash = STYLE_CONFIG.F ? createGhibliWater(waveField, STYLE_CONFIG.F.water) : null;
const waterClay = STYLE_CONFIG.H ? createClayWater(waveField, STYLE_CONFIG.H.water) : null;
const waterInk = STYLE_CONFIG.I ? createInkWater(waveField, STYLE_CONFIG.I.water) : null;
scene.add(
  waterGhibli.mesh, waterFlat.mesh, waterToon.mesh,
  waterRealistic.mesh, waterUkiyoe.mesh, ...(waterWash ? [waterWash.mesh] : []),
  ...(waterClay ? [waterClay.mesh] : []), ...(waterInk ? [waterInk.mesh] : [])
);

const boat = buildBoat();
boat.heading = THREE.MathUtils.degToRad(params.船体朝向);
scene.add(boat.group, boat.foamRing, boat.shadowBlob);

const clouds = createClouds(scene);

// 浮世绘风格的爪形浪花贴花与立体爪浪，仅在方案 E 下启用
const foamDecals = new FoamDecals(scene);
const waveClaws = new WaveClaws(scene, waveField);

// 太阳方向同时驱动平行光、天穹与各着色器水面的高光
let sunDir = new THREE.Vector3();
function refreshSun() {
  sunDir = updateSun(lights, skyUniforms, params.太阳高度角, params.太阳方位角);
  waterGhibli.uniforms.uSunDir.value.copy(sunDir);
  waterToon.uniforms.uSunDir.value.copy(sunDir);
  waterRealistic.uniforms.uSunDir.value.copy(sunDir);
  if (waterWash) waterWash.uniforms.uSunDir.value.copy(sunDir);
  if (waterClay) waterClay.uniforms.uSunDir.value.copy(sunDir);
}
refreshSun();

// 后处理链：常规风格走 RenderPass 直出，方案 G 走像素化 pass，
// 方案 F 叠加水彩 pass，最后经 OutputPass 做色调映射与编码
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);
const pixelPass = new RenderPixelatedPass(params.像素尺寸, scene, camera, {
  normalEdgeStrength: 0.4,
  depthEdgeStrength: 0.4,
  pixelAligned: true
});
pixelPass.enabled = false;
composer.addPass(pixelPass);
const washPass = createWatercolorPass();
washPass.enabled = false;
composer.addPass(washPass);
composer.addPass(new OutputPass());

// 以当前天穹生成 PMREM 环境贴图，供 PBR 船体做环境光照
let envRenderTarget = null;
function refreshEnvironmentMap() {
  if (envRenderTarget !== null) {
    envRenderTarget.dispose();
  }
  envRenderTarget = createEnvironmentMap(renderer, sky.material);
  boat.setEnvironmentMap(envRenderTarget.texture);
  if (waterClay) waterClay.setEnvironmentMap(envRenderTarget.texture);
}
refreshEnvironmentMap();

// ---------------------------------------------------------------------------
// 画面风格切换
// ---------------------------------------------------------------------------

const styleButtons = document.querySelectorAll('#style-switch button');
let styleController = null;
let savedView = null;
let activeStyle = null;

function setStyle(name) {
  // 版画与微缩布景各有默认机位；离开后恢复用户的原始视图。
  if (name !== activeStyle) {
    if (savedView) {
      camera.position.copy(savedView.position);
      controls.target.copy(savedView.target);
      camera.fov = savedView.fov;
      params.自动环绕 = savedView.autoRotate;
      params.船体朝向 = savedView.heading;
      if (savedView.studioLight) {
        [params.太阳高度角, params.太阳方位角, params.曝光] = savedView.studioLight;
      }
      if (savedView.guiClosed) gui.close(); else gui.open();
      savedView = null;
    }
    if (name === 'E' || name === 'H') {
      savedView = { position: camera.position.clone(), target: controls.target.clone(),
        fov: camera.fov, autoRotate: params.自动环绕, heading: params.船体朝向, guiClosed: gui._closed };
      gui.close();
      if (name === 'E') {
        camera.position.set(19, 10, 23);
        camera.fov = 42;
        controls.target.set(-9, 1.8, 0);
      } else {
        savedView.studioLight = [params.太阳高度角, params.太阳方位角, params.曝光];
        camera.position.set(12, 11, 15);
        camera.fov = 42;
        controls.target.set(0, 0.8, 0);
        params.太阳高度角 = 40;
        params.太阳方位角 = 315;
        params.曝光 = 0.95;
      }
      params.船体朝向 = -20;
      params.自动环绕 = false;
    }
    controls.autoRotate = params.自动环绕;
    boat.heading = THREE.MathUtils.degToRad(params.船体朝向);
    camera.updateProjectionMatrix();
  }
  renderer.toneMappingExposure = params.曝光;
  const shadowType = THREE.PCFShadowMap;
  if (renderer.shadowMap.type !== shadowType) {
    renderer.shadowMap.type = shadowType;
    renderer.shadowMap.needsUpdate = true;
  }
  if (name !== activeStyle && name === 'H') {
    clayClock.reset(waveField.time);
  }
  activeStyle = name;
  params.画面风格 = name;
  waveField.amplitudeScale = (STYLE_CONFIG[name] || STYLE_CONFIG.A).waveScale;

  waterGhibli.mesh.visible = name === 'A' || name === 'G';
  waterFlat.mesh.visible = name === 'B';
  waterToon.mesh.visible = name === 'C';
  waterRealistic.mesh.visible = name === 'D';
  waterUkiyoe.mesh.visible = name === 'E';
  if (waterWash !== null) {
    waterWash.mesh.visible = name === 'F';
  }
  if (waterClay !== null) {
    waterClay.mesh.visible = name === 'H';
  }
  if (waterInk !== null) {
    waterInk.mesh.visible = name === 'I';
  }
  waterFlat.material.wireframe = params.水面线框 && name === 'B';

  boat.applyStyle(name);
  applyEnvironmentStyle(name, { scene, lights, skyUniforms, skyMaterial: sky.material, clouds });
  refreshSun();
  refreshEnvironmentMap();

  foamDecals.group.visible = name === 'E';
  waveClaws.group.visible = name === 'E';
  // 切换发生在暂停状态时，也立即更新波形与浮力，避免保留上一风格的振幅。
  waterGhibli.update();
  waterFlat.update();
  waterToon.update();
  waterRealistic.update();
  waterUkiyoe.update();
  if (waterClay) {
    waterClay.uniforms.uBoatHeading.value = boat.heading;
    waterClay.update();
  }
  if (waterInk) waterInk.update();
  if (waterWash) waterWash.update();
  if (name === 'E') {
    foamDecals.reset(waveField, camera.position);
    waveClaws.update(0, waveField);
  }
  updateBoat(boat, waveField, 1);

  renderPass.enabled = name !== 'G';
  pixelPass.enabled = name === 'G';
  washPass.enabled = name === 'F';

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
  } else if (event.code === 'Digit6') {
    setStyle('F');
  } else if (event.code === 'Digit7') {
    setStyle('G');
  } else if (event.code === 'Digit8') {
    setStyle('H');
  } else if (event.code === 'Digit9') {
    setStyle('I');
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
  '方案 E · 浮世绘海浪': 'E',
  '方案 F · 水彩纸本': 'F',
  '方案 G · 像素艺术': 'G',
  '方案 H · 黏土定格动画': 'H',
  '方案 I · 水墨留白': 'I'
}).onChange(setStyle);

const seaFolder = gui.addFolder('海面');
seaFolder.add(params, '风浪强度', 0.2, 2.0, 0.05).onChange((v) => { waveField.windScale = v; setStyle(params.画面风格); });
seaFolder.add(params, '波速', 0.2, 2.0, 0.05).onChange((v) => { waveField.speedScale = v; });
seaFolder.add(params, '暂停');

const lightFolder = gui.addFolder('光照');
lightFolder.add(params, '太阳高度角', 10, 85, 1).listen().onChange(refreshSun);
lightFolder.add(params, '太阳方位角', 0, 360, 1).listen().onChange(refreshSun);
lightFolder.add(params, '曝光', 0.6, 1.6, 0.01).listen().onChange((v) => { renderer.toneMappingExposure = v; });

const sceneFolder = gui.addFolder('场景');
sceneFolder.add(params, '像素尺寸', 2, 12, 1).onChange((v) => { pixelPass.setPixelSize(v); });
sceneFolder.add(params, '船体朝向', -180, 360, 1).listen().onChange((v) => {
  boat.heading = THREE.MathUtils.degToRad(v);
  if (waterClay) waterClay.uniforms.uBoatHeading.value = boat.heading;
});
sceneFolder.add(params, '自动环绕').listen().onChange((v) => { controls.autoRotate = v; });
sceneFolder.add(params, '水面线框').onChange((v) => { waterFlat.material.wireframe = v; });

// ---------------------------------------------------------------------------
// 视口尺寸与主循环
// ---------------------------------------------------------------------------

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
  washPass.uniforms.uResolution.value.set(
    window.innerWidth * 2.0,
    window.innerHeight * 2.0
  );
});

let lastTime = performance.now();
let waveTimeRaw = Number.isFinite(initialTime) && initialTime >= 0 ? initialTime : 0.0;   // 波面连续时钟，方案 G 下量化后写入波场

waveField.time = waveTimeRaw;
let lastClayTime = waveField.time;
const clayClock = new StopMotionClock(12);
clayClock.reset(waveField.time);
setStyle(params.画面风格);

renderer.setAnimationLoop(() => {
  const now = performance.now();
  let delta = (now - lastTime) / 1000;
  lastTime = now;
  delta = Math.min(delta, 1 / 30);

  if (!params.暂停) {
    // H 按真实曝光时钟拍摄十二帧；改变波速不会把定格帧率变成六帧或二十四帧。
    waveTimeRaw += delta * waveField.speedScale * (params.画面风格 === 'E' ? 0.32 : 1.0);
    if (params.画面风格 === 'H') {
      clayClock.advance(delta, waveTimeRaw, waveField.speedScale);
      waveField.time = clayClock.time;
    } else {
      waveField.time = params.画面风格 === 'G' ? Math.floor(waveTimeRaw * 12) / 12 : waveTimeRaw;
    }
    if (params.画面风格 === 'A' || params.画面风格 === 'G') {
      waterGhibli.update();
    } else if (params.画面风格 === 'C') {
      waterToon.update();
    } else if (params.画面风格 === 'D') {
      waterRealistic.update();
    } else if (params.画面风格 === 'E') {
      waterUkiyoe.update();
      foamDecals.update(delta, waveField, camera.position);
      waveClaws.update(delta, waveField, camera.position);
    } else if (params.画面风格 === 'F' && waterWash !== null) {
      waterWash.update();
    } else if (params.画面风格 === 'H' && waterClay !== null) {
      if (waveField.time !== lastClayTime) waterClay.update();
    } else if (params.画面风格 === 'I' && waterInk !== null) {
      waterInk.update();
    } else {
      waterFlat.update();
    }
    if (params.画面风格 === 'H') {
      // 船的平滑器和云也只在曝光到下一定格帧时更新；同一帧内保持姿态。
      if (waveField.time !== lastClayTime) {
        const frameDelta = Math.min(0.25, Math.max(1 / 12, waveField.time - lastClayTime));
        updateBoat(boat, waveField, frameDelta);
        clouds.update(frameDelta);
        lastClayTime = waveField.time;
      }
    } else {
      updateBoat(boat, waveField, delta);
      clouds.update(delta);
      lastClayTime = waveField.time;
    }
    waterToon.uniforms.uBoatPos.value.set(boat.group.position.x, boat.group.position.z);
  }

  washPass.uniforms.uTime.value = waveField.time;
  controls.update();
  if (params.画面风格 === 'F' || params.画面风格 === 'G') {
    composer.render();
  } else {
    renderer.render(scene, camera);
  }
  stats.update();
});
