import * as THREE from 'three';

// 小块白沫沿主浪峰成组推进，整张细分贴片逐顶点贴合位移后的水面。
// 不面向相机、不补偿透视尺寸、不随机播撒；尺寸在 XZ 两轴共同生长。
function createFoamTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#f5efdf';
  ctx.strokeStyle = '#244c63';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(18, 92);
  ctx.bezierCurveTo(60, 84, 92, 80, 118, 76);
  // 不等长的弯曲尖指，统一向前，避免云朵式半圆排。
  for (let i = 0; i < 4; i++) {
    const x = 122 + i * 27;
    ctx.bezierCurveTo(x - 9, 52 - i * 6, x + 9, 25 + i * 3, x + 18, 35 + i * 4);
    ctx.bezierCurveTo(x + 8, 46 + i * 3, x + 8, 64, x + 24, 76);
  }
  ctx.bezierCurveTo(225, 99, 72, 112, 18, 92);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class FoamDecals {
  constructor(scene) {
    this.group = new THREE.Group();
    this.group.visible = false;
    scene.add(this.group);
    const texture = createFoamTexture();
    this.patches = Array.from({ length: 15 }, (_, i) => {
      const geometry = new THREE.PlaneGeometry(1, 1, 8, 4);
      geometry.rotateX(-Math.PI / 2);
      const base = geometry.attributes.position.array.slice();
      const material = new THREE.MeshBasicMaterial({
        map: texture, transparent: true, depthWrite: false,
        side: THREE.DoubleSide, toneMapped: false,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.frustumCulled = false;
      this.group.add(mesh);
      return { mesh, base, row: Math.floor(i / 5), lateral: (i % 5 - 2) * 6 + Math.floor(i / 5) * 2.3 };
    });
    this.out = { dx: 0, dy: 0, dz: 0 };
  }

  reset(waveField, cameraPos) {
    this.update(0, waveField, cameraPos);
  }

  update(dt, waveField, cameraPos) {
    const w = waveField.waves[0];
    const cycle = waveField.time * w.phaseRate / (Math.PI * 2);
    for (const patch of this.patches) {
      const offset = patch.row * 0.17 + 0.29;
      const q = ((cycle + offset) % 1 + 1) % 1;
      const life = Math.sin(Math.PI * q);
      const along = (q - offset + 1.8 / (Math.PI * 2) - patch.row - 1) * w.wavelength;
      const x = w.dx * along - w.dz * patch.lateral;
      const z = w.dz * along + w.dx * patch.lateral;
      const dist = cameraPos ? Math.hypot(x - cameraPos.x, z - cameraPos.z) : 20;
      patch.mesh.material.opacity = life * Math.min(1, waveField.windScale) * (1 - THREE.MathUtils.smoothstep(dist, 45, 95)) * 0.75;
      const size = 1.2 + life * 0.8;
      const attr = patch.mesh.geometry.attributes.position;
      for (let n = 0; n < attr.count; n++) {
        const lateral = patch.lateral + patch.base[n * 3] * size;
        const forward = along - patch.base[n * 3 + 2] * size * 0.48;
        const bx = w.dx * forward - w.dz * lateral;
        const bz = w.dz * forward + w.dx * lateral;
        waveField.sample(bx, bz, this.out);
        attr.setXYZ(n, bx + this.out.dx, this.out.dy + 0.025, bz + this.out.dz);
      }
      attr.needsUpdate = true;
    }
  }
}
