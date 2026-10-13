import * as THREE from 'three';

const SIZE = 280;      // 水面边长，单位米
const SEGMENTS = 96;   // 网格划分数，三角形边长约 2.9 米，保留低多边形的折面感

// 按面重心高度分档的逐面配色，从波谷到浪花共四档
const BANDS = [
  { limit: -0.46,      color: 0x14709f },   // 波谷深水
  { limit:  0.34,      color: 0x18a0cc },   // 主体海蓝
  { limit:  0.68,      color: 0x3cc2d4 },   // 波峰浅青
  { limit:  Infinity,  color: 0xe8fbff }    // 浪花白
];

// 低多边形海面：索引几何体负责 Gerstner 位移计算，散射到非索引几何体后
// 逐三角形按重心高度写入分档顶点色，配合 flatShading 得到折面平涂效果
export function createWater(waveField) {
  const baseGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
  baseGeo.rotateX(-Math.PI / 2);

  // 静止网格坐标备份，位移每帧从静止位置重新计算
  const rest = baseGeo.attributes.position.array.slice();
  const restCount = baseGeo.attributes.position.count;
  const index = baseGeo.index.array;

  const geo = baseGeo.toNonIndexed();
  geo.setAttribute(
    'color',
    new THREE.BufferAttribute(new Float32Array(geo.attributes.position.count * 3), 3)
  );

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.45,
    metalness: 0.0
  });

  const mesh = new THREE.Mesh(geo, material);
  mesh.receiveShadow = true;

  const bandColors = BANDS.map((b) => new THREE.Color(b.color));
  const out = { dx: 0.0, dy: 0.0, dz: 0.0 };

  function update() {
    const src = baseGeo.attributes.position.array;

    // 对每个独立顶点求一次 Gerstner 位移
    for (let i = 0; i < restCount; i++) {
      const o = i * 3;
      waveField.sample(rest[o], rest[o + 2], out);
      src[o] = rest[o] + out.dx;
      src[o + 1] = out.dy;
      src[o + 2] = rest[o + 2] + out.dz;
    }

    // 把独立顶点散射到非索引顶点，并按面重心高度写入分档颜色
    const dstPos = geo.attributes.position.array;
    const dstCol = geo.attributes.color.array;
    for (let v = 0; v < index.length; v++) {
      const s = index[v] * 3;
      const d = v * 3;
      dstPos[d] = src[s];
      dstPos[d + 1] = src[s + 1];
      dstPos[d + 2] = src[s + 2];
    }
    for (let v = 0; v < index.length; v += 3) {
      const h = (dstPos[v * 3 + 1] + dstPos[(v + 1) * 3 + 1] + dstPos[(v + 2) * 3 + 1]) / 3.0;
      let band = 0;
      while (h > BANDS[band].limit) {
        band++;
      }
      const col = bandColors[band];
      for (let k = 0; k < 3; k++) {
        const o = (v + k) * 3;
        dstCol[o] = col.r;
        dstCol[o + 1] = col.g;
        dstCol[o + 2] = col.b;
      }
    }

    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
  }

  return { mesh, material, update };
}
