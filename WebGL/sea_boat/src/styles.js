import * as THREE from 'three';

// 各画面风格的场景级配置：天空、雾、光照、云与波形振幅比例。
// 水面配色由各自的水面实现读取 water 字段
export const STYLE_CONFIG = {
  A: {
    label: '吉卜力晴海',
    skyTop: 0x1f6fc9,
    skyHorizon: 0xe8f5fa,
    sunGlow: 1.6,
    fog: 0xe8f5fa,
    sunIntensity: 2.2,
    hemiIntensity: 0.9,
    cloudScale: 1.45,
    waveScale: 0.55,
    water: {
      deep: 0x1565a0,
      mid: 0x1e93d1,
      crest: 0x62d2e2,
      foam: 0xfcffff,
      sky: 0xdff4fb,
      sun: 0xfff3d6
    }
  },
  B: {
    label: '低多边形海面',
    skyTop: 0x2a86cc,
    skyHorizon: 0xd4eef8,
    sunGlow: 1.0,
    fog: 0xd4eef8,
    sunIntensity: 1.7,
    hemiIntensity: 0.85,
    cloudScale: 1.0,
    waveScale: 1.0
  },
  C: {
    label: '风之杖卡通海',
    skyTop: 0x1e63c8,
    skyHorizon: 0xbfeaf6,
    sunGlow: 1.25,
    fog: 0xbfeaf6,
    sunIntensity: 2.0,
    hemiIntensity: 1.0,
    cloudScale: 1.2,
    waveScale: 0.75,
    water: {
      deep: 0x0f55a4,
      mid: 0x2b90cc,
      crest: 0x63ccd9,
      foam: 0xf6ffff,
      sky: 0xcfeef8,
      sun: 0xfff3d6
    }
  },
  D: {
    label: '半写实明亮海',
    skyTop: 0x2470c8,
    skyHorizon: 0xd6eefa,
    sunGlow: 1.1,
    fog: 0xd6eefa,
    sunIntensity: 2.4,
    hemiIntensity: 1.0,
    cloudScale: 1.1,
    waveScale: 1.0,
    water: {
      deep: 0x083f78,
      shallow: 0x1795bd,
      sss: 0x8ff0e0,
      foam: 0xf4fcff,
      sky: 0x2f7fd0,
      horizon: 0xd6eefa,
      sun: 0xfff2d8
    }
  },
  E: {
    label: '浮世绘海浪',
    skyTop: 0xe3d3ae,
    skyHorizon: 0xf5ecd8,
    sunGlow: 0.0,
    fog: 0xf0e6cf,
    sunIntensity: 1.3,
    hemiIntensity: 0.9,
    cloudScale: 1.0,
    waveScale: 0.8,
    water: {
      c0: 0x0b2440,
      c1: 0x114567,
      c2: 0x2a6b8f,
      c3: 0x5495ad,
      c4: 0xa8c4c9,
      foam: 0xf5efdf,
      ink: 0x16334a
    }
  }
};

let gradientMapCache = null;

// 卡通着色的三档明暗渐变映射，船体与云共用
export function getGradientMap() {
  if (gradientMapCache === null) {
    const steps = [96, 200, 255];
    const data = new Uint8Array(steps);
    gradientMapCache = new THREE.DataTexture(data, steps.length, 1, THREE.RedFormat);
    gradientMapCache.minFilter = THREE.NearestFilter;
    gradientMapCache.magFilter = THREE.NearestFilter;
    gradientMapCache.generateMipmaps = false;
    gradientMapCache.needsUpdate = true;
  }
  return gradientMapCache;
}
