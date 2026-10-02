# Demo 02 · 雨落池塘

完整系列方案见 [WATER_DEMOS.md](WATER_DEMOS.md)。前一个 demo 见
[demo_01 说明](README.md)。

![雨落池塘预览](preview_rain.png)

## 启动

```bash
cd /Users/john/git/CGPlayGround/Taichi
uv sync --locked
uv run python water/demo_02.py
```

默认 960 × 600，每像素四个空间采样。可指定后端、分辨率与采样数：

```bash
uv run python water/demo_02.py --backend metal --width 800 --height 500
uv run python water/demo_02.py --backend cpu --width 480 --height 300
uv run python water/demo_02.py --samples 1 --width 800 --height 500
```

首次启动需要编译 kernel，当前版本冷编译可能需要一分钟或更久。
启动时先在窗口创建前准备首帧，终端显示
`[Startup] Preparing first frame`；完成后才打开窗口并显示场景。
默认启用本地编译缓存，位于 `water/.taichi-cache/`，不纳入版本控制。

## 视角和参数

| 操作 | 功能 |
|---|---|
| 鼠标左键点击水面 | 在点击处生成波纹 |
| 鼠标左键拖动 | 围绕目标旋转 |
| 鼠标右键拖动 | 沿地面平移观察目标 |
| W / S 或上下方向键 | 拉近 / 拉远 |
| 1 / 2 / 3 | 全景 / 低角度 / 俯视 |
| R | 重置镜头、清空水面和雨滴、恢复参数 |
| 空格 | 暂停 / 继续 |
| N | 暂停时推进 1/60 秒 |
| A | 开关自动环绕 |
| H | 显示 / 隐藏面板 |
| P | 保存无面板截图到 `--output` 指定的位置 |
| Esc | 退出 |

左侧面板可调雨量、雨滴冲击强度、雨滴可见度、波速、阻尼、时间速度、
水体清澈度、日间到日落的光照与曝光。波速上限 1.8 m/s，保证显式
差分格式满足稳定性条件。点击 `Calm water` 立即平静水面。

## 截图与验证

```bash
uv run python water/demo_02.py --headless --output output/demo_02.png
uv run python water/demo_02.py --headless --preset waterline --output output/demo_02_low.png
uv run python water/demo_02.py --headless --preset top --time 4 --output output/demo_02_top.png
uv run python water/demo_02.py --headless --frames 60 --width 640 --height 400
```

`--time` 指定首帧前的下雨预热秒数，`--frames` 在无窗口模式下按固定的
1/60 秒推进，最后一帧写入 `--output`。雨滴生成使用固定随机种子，
画面可复现。

```bash
uv run python -m unittest discover -s water/tests -v
```

测试覆盖波动方程的稳定性、传播、阻尼衰减、双线性采样精度、雨水确定
性与区域约束、焦散对水面状态的响应、拾取反投影往返、暂停确定性、
参数校验和启动顺序。

## 实现说明

### 波动模拟

水面高度由二维波动方程的有限差分格式求解。网格 256 × 172 覆盖整个
池面，单元边长 0.025 m。时间积分采用蛙跳格式，固定时间步 1/120 秒，
每显示帧推进两个子步。三个缓冲依次轮转当前、上一、下一时刻，避免
重绑定字段引用。显式格式的稳定性要求波速乘时间步除以单元边长小于
1/√2，波速面板范围据此限制在 0.3 到 1.8 m/s。

池壁采用零梯度边界，波纹到达池壁后反射回来，可以观察到来回交错的
干涉纹样。阻尼默认 0.998，涟漪约三四秒自然消散。

### 雨滴

雨滴生成表由固定种子预先计算，每个雨滴落水后按序取下一个生成表项，
因此整场雨完全确定。落水瞬间按下落速度把高斯形扰动写入高度场，
形成一圈圈扩张的涟漪。雨丝在渲染核心里逐光线解析求最近距离，
只在池面上方区域生成，不遮挡场景物体，也无需额外的遮挡测试。

### 交互

点击水面时，把鼠标位置反投影为视线并与平均水面求交，交点在池内
则注入一个较强的高斯扰动。按下后移动超过阈值视为拖动旋转，
不会产生波纹。

### 渲染

渲染复用 Demo 01 的水材质：Schlick Fresnel、Snell 折射、Beer–Lambert
吸收、GGX 太阳高光、粗糙反射、池底焦散和 HDR 后处理。波面求交改用
牛顿迭代配合双线性采样与单元精确梯度；焦散追踪直接读取模拟高度场，
雨点造成的光纹随模拟同步运动。两组高频细节波只扰动着色法线，
按像素覆盖尺寸衰减，与 Demo 01 相同。

场景几何、环境光照和 PBR 着色从 Demo 01 抽取到 `scene.py`，
两个 demo 共享 `CourtyardScene`。Demo 01 重构后全部原有测试保持通过。

## 验证范围

39 项检查通过：Demo 01 原有 26 项，加上本轮 13 项。Metal 离屏渲染
全景、低角度、俯视画面均已检查。GGUI 窗口完成首帧显示、面板构建及
三帧退出验证，冷编译首帧准备耗时 72 秒，缓存命中后约 6 秒。
实际鼠标点击与键盘操作尚未自动化验证。

本机 Metal 离屏测量（包含逐帧焦散、模拟推进、渲染、后处理提交与
同步，排除窗口界面和 PNG 写入）：

| 配置 | 编译后平均耗时 |
|---|---|
| 960 × 600，全景，默认完整画质 | 182 ms/帧 |
| 960 × 600，全景，关闭雨丝 | 142 ms/帧 |
| 960 × 600，俯视，默认完整画质 | 211 ms/帧 |
| 800 × 500 左右轻量配置，单采样 | 22.3 ms/帧 |

默认画质未达到本机 60 FPS。雨丝渲染约占 40 ms，可先降低雨量或关闭
雨滴可见度；再按需降低空间采样、反射方向数、焦散和 Bloom。
