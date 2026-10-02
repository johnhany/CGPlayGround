# Taichi demos

此目录使用已发布的 Taichi Python 包运行渲染和计算 demo。
`/Users/john/git/taichi` 是引擎源码的只读参考目录，不参与本项目安装。

## 环境与运行

使用 uv 管理 Python 3.11、目录内的 `.venv` 和依赖锁文件 `uv.lock`：

```bash
cd /Users/john/git/CGPlayGround/Taichi
uv sync --locked
uv run python jacobi_iteration.py
uv run python fractal.py
```

从仓库根目录运行时，可使用：

```bash
uv run --project Taichi python Taichi/count_prime.py
```

环境包括 Taichi、NumPy、PyTorch 和 Pillow。无需手动激活环境；若要激活，可执行
`source .venv/bin/activate`。

## 水面渲染系列

- [开发方案](WATER_DEMOS.md)：九个由浅入深的水面、海浪、交互与流体 demo。
- [Demo 01：庭院浅池](water/README.md)：`uv run python water/demo_01.py`。
  包含完整庭院背景、波纹、反射折射、轨道镜头、参数控制和离屏截图。

## 运行条件

- `fractal.py`、`cloth_simulation.py`、`mass_spring.py` 会打开图形窗口，需要桌面会话。
- 多个 demo 请求 Vulkan；运行后应留意 Taichi 输出的实际计算后端和回退提示。
- `convolution.py` 和 `tile_padding.py` 写死了 CUDA 设备，需要 NVIDIA CUDA 环境；当前 Mac 上安装依赖并不会提供 CUDA。
- `tile_padding.py` 保存图片前需要在当前工作目录创建 `output` 目录：`mkdir -p output`。
- `longest_common_subsequence.py` 默认分配约 900 MB 的矩阵，并进行较长时间的计算。

## 引擎源码检索

codebase-memory 的索引目标是 `/Users/john/git/taichi`，项目名为
`Users-john-git-taichi`，与此 demo 项目区分。已使用 full 模式初始化；
索引保存在工具缓存中，未向源码目录写入共享图文件。
使用 `list_projects` 确认索引状态，再通过 `search_graph`、
`trace_path` 和 `get_code_snippet` 检索实现；引用源码前使用
`check_index_coverage` 检查相关文件，解析不完整时直接读取对应源码范围。

## 本机验证记录

- Python 3.11.16、Taichi 1.7.4；八个 demo 的语法解析、依赖导入、
  PyTorch/NumPy 互操作及 Taichi CPU kernel 验证通过。
- `jacobi_iteration.py` 在 CPU 上完成运行并通过脚本中的断言。
- 本机 `count_prime.py` 在 CPU 上得到 `78498`，原脚本请求的 Vulkan
  后端得到 `78504`，存在计算结果差异。CPU 复核命令：
  `TI_ARCH=arm64 uv run python count_prime.py`（此后端名用于本机 Apple Silicon）。
- Codex 沙箱中 Taichi 在计算完成后的退出阶段出现段错误；同样的 CPU
  验证在沙箱外正常退出。后续在 Codex 内运行时需考虑该执行限制。
- 图形窗口和 CUDA benchmark 未做完整运行验证。
