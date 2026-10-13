import * as THREE from 'three';
import { getGradientMap } from './styles.js';

// 船体分站数据，沿纵向排布：z 为纵向位置，g 为舷边半宽，gy 为舷边高度，
// c 为舭部半宽，cy 为舭部高度，k 为龙骨高度。y 等于 0 为设计水线
const STATIONS = [
  { z:  2.30, g: 0.04, gy: 0.50, c: 0.03, cy:  0.26, k:  0.24 },
  { z:  1.45, g: 0.46, gy: 0.50, c: 0.33, cy: -0.14, k: -0.42 },
  { z:  0.45, g: 0.62, gy: 0.50, c: 0.49, cy: -0.22, k: -0.52 },
  { z: -0.60, g: 0.66, gy: 0.50, c: 0.53, cy: -0.22, k: -0.50 },
  { z: -1.55, g: 0.60, gy: 0.52, c: 0.47, cy: -0.18, k: -0.40 },
  { z: -2.10, g: 0.55, gy: 0.55, c: 0.41, cy: -0.14, k: -0.26 }
];

// 方案 B 低多边形配色
const B_COLORS = {
  bottom: 0xb0472f,   // 水线以下暗红防污漆
  hull: 0xf3f5f0,     // 水线以上暖白船壳
  deck: 0xd8b078,     // 甲板木色
  wood: 0x8a6a48,     // 桅杆与帆桁
  sail: 0xf7f6ef,     // 主帆
  jib: 0xe9ece4,      // 前帆
  flag: 0xd94f3d,     // 旗子
  roof: 0xcf5b3e,     // 舱顶
  glass: 0x27455c     // 舷窗
};

// 方案 A 吉卜力配色：明快提纯，配卡通渐变着色
const A_COLORS = {
  bottom: 0xe2543a,
  hull: 0xfefef6,
  deck: 0xe8c084,
  wood: 0x9a7350,
  sail: 0xfffef4,
  jib: 0xf6f8ee,
  flag: 0xe8402f,
  roof: 0xe8552f,
  glass: 0x2f5d7a
};

// 方案 C 风之杖配色：暖调木船体，配描边轮廓
const C_COLORS = {
  bottom: 0xd0402e,
  hull: 0xfff6e4,
  deck: 0xe0b070,
  wood: 0x7a5238,
  sail: 0xfffef0,
  jib: 0xf3f0e0,
  flag: 0xd83a24,
  roof: 0xc84a2e,
  glass: 0x1f4a66
};

// 方案 D 半写实配色：木纹质感，水线附近带湿润渐变
const D_COLORS = {
  bottom: 0x7a3a28,
  hull: 0xd9a86a,
  deck: 0xc59a63,
  wood: 0x6e4a2e,
  sail: 0xf6f2e6,
  jib: 0xefe9da,
  flag: 0xc73b2a,
  roof: 0x8f5636,
  glass: 0x223c4e
};

// 方案 E 浮世绘配色：木色剪影与米色纸帆，旗为套印的一点朱红
const E_COLORS = {
  bottom: 0x2e2013,
  hull: 0xb58a54,
  deck: 0xc6a16c,
  wood: 0x86603c,
  sail: 0xf0e6cc,
  jib: 0xe8dcc0,
  flag: 0xb03a28,
  roof: 0x936744,
  glass: 0x24303a
};

// 方案 F 水彩配色：柔和低饱和的水洗色调
const F_COLORS = {
  bottom: 0x6a4a34,
  hull: 0x8a6a4a,
  deck: 0xb59a6e,
  wood: 0x7a5a3e,
  sail: 0xf2ead6,
  jib: 0xeadfc6,
  flag: 0xc2726a,
  roof: 0x9a6a4e,
  glass: 0x5a7a8a
};

const WATERLINE = -0.12;         // 船壳分色判据
const FLOAT_FORE = 1.7;          // 浮心采样点：船前距离
const FLOAT_AFT = 1.6;           // 船后距离
const FLOAT_BEAM = 0.85;         // 左右采样半宽
const OUTLINE_COLOR = 0x22384e;  // 方案 C 描边颜色
const OUTLINE_WIDTH = 0.05;      // 描边宽度，单位米

function smoothstep(a, b, x) {
  const t = Math.min(Math.max((x - a) / (b - a), 0.0), 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// 程序化帆船。船体由分站轮廓放样而成，船首船尾以扇形封口，
// 每个部件保留三套材质：方案 B 平顶着色、方案 A 与方案 C 卡通渐变着色，
// 方案 C 额外在实体部件外侧叠加法线外扩的倒置外壳描边
export function buildBoat() {
  const group = new THREE.Group();
  const styleEntries = [];
  const envMats = [];   // 参与 PBR 环境反射的标准材质及其环境光强度

  function trackEnv(mat, intensity) {
    mat.envMapIntensity = intensity;
    envMats.push({ mat, intensity });
    return mat;
  }

  // ---- 船体放样 ----

  const positions = [];
  const colors = [];
  const bottomColor = new THREE.Color(B_COLORS.bottom);
  const hullColor = new THREE.Color(B_COLORS.hull);
  const deckColor = new THREE.Color(B_COLORS.deck);

  // 每个分站的半边轮廓，从右舷舷边到左舷舷边共五点
  function outline(s) {
    return [
      [ s.g,  s.gy],
      [ s.c,  s.cy],
      [ 0.0,  s.k  ],
      [-s.c,  s.cy],
      [-s.g,  s.gy]
    ];
  }

  function addTriangle(a, b, c, color) {
    positions.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    for (let i = 0; i < 3; i++) {
      colors.push(color.r, color.g, color.b);
    }
  }

  function addQuad(a, b, c, d, color) {
    addTriangle(a, b, c, color);
    addTriangle(a, c, d, color);
  }

  function pointAt(s, p) {
    return [p[0], p[1], s.z];
  }

  // 侧壁：相邻分站轮廓对应点连成四段板面
  for (let i = 0; i < STATIONS.length - 1; i++) {
    const A = outline(STATIONS[i]);
    const B = outline(STATIONS[i + 1]);
    for (let seg = 0; seg < 4; seg++) {
      const p0 = pointAt(STATIONS[i], A[seg]);
      const p1 = pointAt(STATIONS[i], A[seg + 1]);
      const p2 = pointAt(STATIONS[i + 1], B[seg + 1]);
      const p3 = pointAt(STATIONS[i + 1], B[seg]);
      addQuad(p0, p1, p2, p3, hullColor);
    }
  }

  // 甲板：左右舷边连成的顶面板，略低于舷边
  for (let i = 0; i < STATIONS.length - 1; i++) {
    const s0 = STATIONS[i];
    const s1 = STATIONS[i + 1];
    const y0 = s0.gy - 0.06;
    const y1 = s1.gy - 0.06;
    addQuad(
      [-s0.g, y0, s0.z],
      [ s0.g, y0, s0.z],
      [ s1.g, y1, s1.z],
      [-s1.g, y1, s1.z],
      deckColor
    );
  }

  // 船首与船尾封口：轮廓五点的扇形剖分
  function cap(s, color) {
    const O = outline(s).map((p) => pointAt(s, p));
    addTriangle(O[0], O[1], O[2], color);
    addTriangle(O[0], O[2], O[3], color);
    addTriangle(O[0], O[3], O[4], color);
  }
  cap(STATIONS[0], hullColor);
  cap(STATIONS[STATIONS.length - 1], hullColor);

  const hullGeo = new THREE.BufferGeometry();
  hullGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  hullGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  hullGeo.computeVertexNormals();

  // 六套配色下的船壳顶点色，切换风格时整体替换 color 属性
  const gradientMap = getGradientMap();
  const paletteOf = { A: A_COLORS, B: B_COLORS, C: C_COLORS, D: D_COLORS, E: E_COLORS, F: F_COLORS };
  const hullColorAttrs = {};
  for (const key of ['A', 'B', 'C', 'D', 'E', 'F']) {
    const data = new Float32Array(colors);
    if (key === 'D') {
      recolorHullWet(
        data, positions,
        new THREE.Color(paletteOf[key].bottom),
        new THREE.Color(paletteOf[key].hull)
      );
    } else {
      recolorHull(
        data, positions,
        new THREE.Color(paletteOf[key].bottom),
        new THREE.Color(paletteOf[key].hull)
      );
    }
    hullColorAttrs[key] = new THREE.BufferAttribute(data, 3);
  }

  const woodTexture = createWoodTexture();

  const hullMatB = trackEnv(new THREE.MeshStandardMaterial({
    vertexColors: true,
    flatShading: true,
    roughness: 0.6,
    metalness: 0.0,
    side: THREE.DoubleSide
  }), 0.15);
  const hullMatD = trackEnv(new THREE.MeshStandardMaterial({
    vertexColors: true,
    map: woodTexture,
    roughness: 0.5,
    metalness: 0.02,
    side: THREE.DoubleSide
  }), 0.85);
  const hullMatToon = new THREE.MeshToonMaterial({
    vertexColors: true,
    gradientMap,
    side: THREE.DoubleSide
  });

  // 固定绘画方向的两档颜料，不受太阳角度、曝光或环境反射改变。
  function printMaterial(hex, vertexColors = false) {
    const mat = new THREE.MeshBasicMaterial({
      color: hex, vertexColors: vertexColors === true, side: THREE.DoubleSide, toneMapped: false
    });
    mat.onBeforeCompile = shader => {
      shader.vertexShader = 'varying float vPrintShade;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvPrintShade = dot(normalize(normal), normalize(vec3(-0.4, 1.0, 0.6)));');
      shader.fragmentShader = 'varying float vPrintShade;\n' + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>',
        '#include <color_fragment>\ndiffuseColor.rgb *= mix(0.72, 1.0, step(0.15, vPrintShade));');
    };
    mat.customProgramCacheKey = () => 'ukiyoe-two-ink-v1';
    return mat;
  }
  const hullMatPrint = printMaterial(0xffffff, true);

  const hull = new THREE.Mesh(hullGeo, hullMatB);
  hull.castShadow = true;
  group.add(hull);
  styleEntries.push({
    mesh: hull,
    solid: true,
    apply(style) {
      hull.geometry.setAttribute('color', hullColorAttrs[style] || hullColorAttrs.A);
      hull.material = style === 'B' ? hullMatB
        : (style === 'D' ? hullMatD : (style === 'E' ? hullMatPrint : hullMatToon));
    }
  });

  // ---- 上层建筑 ----

  const plainB = (hex, extra = {}) => trackEnv(new THREE.MeshStandardMaterial({
    color: hex,
    flatShading: true,
    roughness: 0.75,
    metalness: 0.0,
    ...extra
  }), 0.15);
  const plainToon = (hex, extra = {}) => new THREE.MeshToonMaterial({
    color: hex,
    gradientMap,
    ...extra
  });
  const plainD = (hex, extra = {}) => {
    const { envIntensity, ...rest } = extra;
    return trackEnv(new THREE.MeshStandardMaterial({
      color: hex,
      roughness: 0.55,
      metalness: 0.02,
      ...rest
    }), envIntensity ?? 0.75);
  };

  // mats 形如 { B, A, C }，C 未单独给出时与 A 共用卡通材质
  function addStyledMesh(mesh, mats, solid = true) {
    mesh.castShadow = true;
    group.add(mesh);
    styleEntries.push({
      mesh,
      solid,
      apply(style) { mesh.material = mats[style] || mats.A; }
    });
    return mesh;
  }

  function addBox(w, h, d, x, y, z, mats) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mats.B);
    mesh.position.set(x, y, z);
    return addStyledMesh(mesh, mats);
  }

  const cabinMats = {
    B: plainB(B_COLORS.hull),
    A: plainToon(A_COLORS.hull),
    C: plainToon(C_COLORS.hull),
    D: plainD(D_COLORS.hull, { map: woodTexture }),
    E: printMaterial(E_COLORS.hull),
    F: plainToon(F_COLORS.hull)
  };
  const roofMats = {
    B: plainB(B_COLORS.roof),
    A: plainToon(A_COLORS.roof),
    C: plainToon(C_COLORS.roof),
    D: plainD(D_COLORS.roof, { roughness: 0.5 }),
    E: printMaterial(E_COLORS.roof),
    F: plainToon(F_COLORS.roof)
  };
  const glassMats = {
    B: plainB(B_COLORS.glass),
    A: plainToon(A_COLORS.glass),
    C: plainToon(C_COLORS.glass),
    D: plainD(D_COLORS.glass, { roughness: 0.12, metalness: 0.1, envIntensity: 1.3 }),
    E: printMaterial(E_COLORS.glass),
    F: plainToon(F_COLORS.glass)
  };
  const woodMats = {
    B: plainB(B_COLORS.wood),
    A: plainToon(A_COLORS.wood),
    C: plainToon(C_COLORS.wood),
    D: plainD(D_COLORS.wood, { map: woodTexture, roughness: 0.45 }),
    E: printMaterial(E_COLORS.wood),
    F: plainToon(F_COLORS.wood)
  };

  addBox(0.95, 0.5, 1.1, 0, 0.71, -0.55, cabinMats);              // 船舱
  addBox(1.08, 0.08, 1.24, 0, 1.0, -0.55, roofMats);              // 舱顶
  addBox(0.02, 0.18, 0.5, 0.485, 0.75, -0.55, glassMats);         // 左舷窗
  addBox(0.02, 0.18, 0.5, -0.485, 0.75, -0.55, glassMats);        // 右舷窗
  addBox(0.5, 0.18, 0.02, 0, 0.75, 0.005, glassMats);             // 前窗

  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.06, 3.8, 6), woodMats.B);
  mast.position.set(0, 2.36, 0.5);
  addStyledMesh(mast, woodMats);

  const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.6, 6), woodMats.B);
  boom.rotation.x = Math.PI / 2;
  boom.position.set(0, 1.2, -0.28);
  addStyledMesh(boom, woodMats);

  // ---- 帆与旗 ----

  function addTriangleMesh(a, b, c, mats) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(
      [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]], 3
    ));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mats.B);
    return addStyledMesh(mesh, mats);
  }

  const sailMats = {
    B: plainB(B_COLORS.sail, { side: THREE.DoubleSide, roughness: 0.9 }),
    A: plainToon(A_COLORS.sail, { side: THREE.DoubleSide }),
    C: plainToon(C_COLORS.sail, { side: THREE.DoubleSide }),
    D: plainD(D_COLORS.sail, { side: THREE.DoubleSide, roughness: 0.85, envIntensity: 0.3 }),
    E: printMaterial(E_COLORS.sail),
    F: plainToon(F_COLORS.sail, { side: THREE.DoubleSide })
  };
  const jibMats = {
    B: plainB(B_COLORS.jib, { side: THREE.DoubleSide, roughness: 0.9 }),
    A: plainToon(A_COLORS.jib, { side: THREE.DoubleSide }),
    C: plainToon(C_COLORS.jib, { side: THREE.DoubleSide }),
    D: plainD(D_COLORS.jib, { side: THREE.DoubleSide, roughness: 0.85, envIntensity: 0.3 }),
    E: printMaterial(E_COLORS.jib),
    F: plainToon(F_COLORS.jib, { side: THREE.DoubleSide })
  };
  const flagMats = {
    B: plainB(B_COLORS.flag, { side: THREE.DoubleSide }),
    A: plainToon(A_COLORS.flag, { side: THREE.DoubleSide }),
    C: plainToon(C_COLORS.flag, { side: THREE.DoubleSide }),
    D: plainD(D_COLORS.flag, { side: THREE.DoubleSide, roughness: 0.7 }),
    E: printMaterial(E_COLORS.flag),
    F: plainToon(F_COLORS.flag, { side: THREE.DoubleSide }),
    G: plainToon(A_COLORS.flag, { side: THREE.DoubleSide })
  };

  addTriangleMesh(
    [0.04, 4.08, 0.47],
    [0.04, 1.2, 0.47],
    [0.04, 1.2, -1.05],
    sailMats
  );
  addTriangleMesh(
    [-0.04, 3.95, 0.42],
    [-0.04, 0.6, 2.1],
    [-0.04, 0.9, 0.8],
    jibMats
  );

  // 旗子以桅顶为原点，主体伸向船尾，主循环里做飘动
  const flagGeo = new THREE.BufferGeometry();
  flagGeo.setAttribute('position', new THREE.Float32BufferAttribute(
    [0, 0.1, 0, 0, -0.02, 0, 0, 0.04, -0.38], 3
  ));
  flagGeo.computeVertexNormals();
  const flag = new THREE.Mesh(flagGeo, flagMats.B);
  flag.position.set(0, 4.12, 0.5);
  group.add(flag);
  styleEntries.push({
    mesh: flag,
    solid: false,
    apply(style) { flag.material = flagMats[style] || flagMats.A; }
  });

  // ---- 方案 C 倒置外壳描边 ----

  // 在实体部件内侧向外沿法线挤出薄壳，仅渲染背面形成轮廓线
  const outlineMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uWidth: { value: OUTLINE_WIDTH },
      uColor: { value: new THREE.Color(OUTLINE_COLOR) }
    },
    side: THREE.BackSide,
    vertexShader: /* glsl */`
      uniform float uWidth;
      void main() {
        vec3 pos = position + normal * uWidth;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      uniform vec3 uColor;
      void main() {
        gl_FragColor = vec4(uColor, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `
  });

  const outlineMeshes = [];
  for (const entry of styleEntries) {
    if (!entry.solid) {
      continue;
    }
    const shell = new THREE.Mesh(entry.mesh.geometry, outlineMaterial);
    shell.visible = false;
    entry.mesh.add(shell);
    outlineMeshes.push(shell);
  }

  // ---- 水线泡沫环（仅方案 A）与柔和船影（仅方案 D）----

  // 方案 A 的水面不接收阴影，改用一圈柔和的水线白沫把船压在水面上；
  // 方案 D 同样不接收实时阴影，用椭圆形软阴影贴图近似投影。
  // 两者独立于船体加入场景，只随波高与航向移动，不跟随纵摇横摇。
  // 方案 C 的船周浪花由水面着色器直接生成，无需泡沫环
  const foamRing = new THREE.Mesh(
    createFoamRingGeometry(),
    new THREE.MeshBasicMaterial({
      map: createFoamRingTexture(),
      transparent: true,
      depthWrite: false
    })
  );
  foamRing.renderOrder = 2;
  foamRing.visible = false;

  const shadowBlob = new THREE.Mesh(
    createShadowBlobGeometry(),
    new THREE.MeshBasicMaterial({
      map: createShadowBlobTexture(),
      transparent: true,
      depthWrite: false
    })
  );
  shadowBlob.renderOrder = 1;
  shadowBlob.visible = false;

  return {
    group,
    flag,
    foamRing,
    shadowBlob,
    heading: 0.0,
    applyStyle(style) {
      for (const entry of styleEntries) {
        entry.apply(style);
      }
      outlineMaterial.toneMapped = style !== 'E';
      outlineMaterial.uniforms.uWidth.value = style === 'E' ? 0.025 : OUTLINE_WIDTH;
      for (const shell of outlineMeshes) {
        shell.visible = style === 'C' || style === 'E';
      }
      foamRing.visible = style === 'A' || style === 'F' || style === 'G';
      shadowBlob.visible = style === 'D';
    },
    // 将天穹生成的 PMREM 环境贴图赋给全部 PBR 标准材质
    setEnvironmentMap(tex) {
      for (const { mat, intensity } of envMats) {
        mat.envMap = tex;
        mat.envMapIntensity = intensity;
      }
    }
  };
}

function recolorHull(colors, vertexPositions, bottomColor, hullColor) {
  for (let i = 0; i < colors.length; i += 9) {
    const y = (vertexPositions[i + 1] + vertexPositions[i + 4] + vertexPositions[i + 7]) / 3.0;
    const col = y < WATERLINE ? bottomColor : hullColor;
    for (let k = 0; k < 3; k++) {
      colors[i + k * 3] = col.r;
      colors[i + k * 3 + 1] = col.g;
      colors[i + k * 3 + 2] = col.b;
    }
  }
}

// 方案 D 的船壳配色：在两色分界之上叠加水线附近的湿润暗带
function recolorHullWet(colors, vertexPositions, bottomColor, hullColor) {
  const wet = new THREE.Color();
  for (let i = 0; i < colors.length; i += 9) {
    const y = (vertexPositions[i + 1] + vertexPositions[i + 4] + vertexPositions[i + 7]) / 3.0;
    const col = y < WATERLINE ? bottomColor : hullColor;
    const band = smoothstep(-0.04, 0.05, y) * (1.0 - smoothstep(0.05, 0.24, y));
    wet.copy(col).multiplyScalar(1.0 - 0.42 * band);
    for (let k = 0; k < 3; k++) {
      colors[i + k * 3] = wet.r;
      colors[i + k * 3 + 1] = wet.g;
      colors[i + k * 3 + 2] = wet.b;
    }
  }
}

function createWoodTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 256, 256);
  // 接近白的底色上叠加木色条痕，与顶点色相乘后保留原色调
  for (let i = 0; i < 110; i++) {
    const y = Math.random() * 256;
    const h = 1 + Math.random() * 3;
    ctx.fillStyle = `rgba(122, 86, 54, ${0.05 + Math.random() * 0.22})`;
    ctx.fillRect(0, y, 256, h);
  }
  for (let i = 0; i < 7; i++) {
    const x = Math.random() * 256;
    const y = Math.random() * 256;
    const r = 2 + Math.random() * 4;
    ctx.fillStyle = 'rgba(96, 62, 36, 0.28)';
    ctx.beginPath();
    ctx.ellipse(x, y, r * 1.8, r, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function createShadowBlobGeometry() {
  // 椭圆软阴影，长轴对应船长，预旋转至水平面
  const geo = new THREE.PlaneGeometry(4.8, 2.7);
  geo.rotateX(-Math.PI / 2);
  return geo;
}

function createShadowBlobTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(128, 128, 20, 128, 128, 122);
  grad.addColorStop(0.0, 'rgba(3, 24, 38, 0.5)');
  grad.addColorStop(0.6, 'rgba(3, 24, 38, 0.34)');
  grad.addColorStop(1.0, 'rgba(3, 24, 38, 0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function createFoamRingGeometry() {
  // 平面横向对应船宽、纵向对应船长，预旋转至水平面
  const geo = new THREE.PlaneGeometry(2.6, 5.4);
  geo.rotateX(-Math.PI / 2);
  return geo;
}

function createFoamRingTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(128, 128, 40, 128, 128, 124);
  grad.addColorStop(0.0, 'rgba(255,255,255,0)');
  grad.addColorStop(0.62, 'rgba(255,255,255,0)');
  grad.addColorStop(0.74, 'rgba(255,255,255,0.85)');
  grad.addColorStop(0.84, 'rgba(255,255,255,0.45)');
  grad.addColorStop(1.0, 'rgba(255,255,255,0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// 浮力驱动：在船底前后左右四个点采样波高，推出升沉、纵摇与横摇目标值，
// 再做指数平滑模拟惯性感。船体局部 y 等于 0 即设计水线，直接以波高为目标高度
export function updateBoat(boat, waveField, dt) {
  const yaw = boat.heading;
  const fwdX = Math.sin(yaw);
  const fwdZ = Math.cos(yaw);
  const rightX = Math.cos(yaw);
  const rightZ = -Math.sin(yaw);

  const out = { dx: 0, dy: 0, dz: 0 };
  const sampleHeight = (lateral, longitudinal) => {
    waveField.sample(
      fwdX * longitudinal + rightX * lateral,
      fwdZ * longitudinal + rightZ * lateral,
      out
    );
    return out.dy;
  };

  const hBow = sampleHeight(0, FLOAT_FORE);
  const hStern = sampleHeight(0, -FLOAT_AFT);
  const hRight = sampleHeight(FLOAT_BEAM, 0);
  const hLeft = sampleHeight(-FLOAT_BEAM, 0);
  const hCenter = sampleHeight(0, 0);

  const smoothFast = 1.0 - Math.exp(-dt * 9.0);
  const smoothTilt = 1.0 - Math.exp(-dt * 6.0);

  const g = boat.group;
  const targetY = (hBow + hStern + hRight + hLeft) / 4.0;
  g.position.y += (targetY - g.position.y) * smoothFast;

  const targetPitch = Math.atan2(hStern - hBow, FLOAT_FORE + FLOAT_AFT);
  const targetRoll = Math.atan2(hRight - hLeft, FLOAT_BEAM * 2.0);
  g.rotation.x += (targetPitch - g.rotation.x) * smoothTilt;
  g.rotation.z += (targetRoll - g.rotation.z) * smoothTilt;

  // 航向附加轻微摇摆
  const t = waveField.time;
  g.rotation.y = boat.heading + 0.02 * Math.sin(t * 0.4) + 0.01 * Math.sin(t * 0.9);

  // 旗子飘动
  boat.flag.rotation.y = 0.22 * Math.sin(t * 5.3) + 0.12 * Math.sin(t * 9.1);

  // 水线泡沫环与软船影贴合船中的波面高度
  if (boat.foamRing.visible) {
    boat.foamRing.position.set(0, hCenter + 0.06, 0);
    boat.foamRing.rotation.y = g.rotation.y;
  }
  if (boat.shadowBlob.visible) {
    boat.shadowBlob.position.set(0, hCenter + 0.05, 0);
    boat.shadowBlob.rotation.y = g.rotation.y;
  }
}
