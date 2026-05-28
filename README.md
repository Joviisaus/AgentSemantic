# AgentSematic

基于网格动画的**畸变分析 + 部件分割 + 热力图渲染**一体化流水线。

输入一个带蒙皮动画的 mesh（`.glb` / `.gltf` / `.fbx` / `.obj` / `.ply` / `.stl` 等），自动完成：

1. 对所有动作的所有帧做加权平均，得到参考形状；
2. 逐动作、逐顶点计算 **角度畸变**（1-ring 角度变化）与 **面积畸变**（1-邻域三角形面积变化）；
3. 在 medoid（最具代表性）姿态上分别用 **SEG-MAT**（基于 Medial Axis Transform）与 **谱聚类**（基于网格邻接图 + 几何/畸变特征）做部件分割；
4. 用 pyvista 离屏渲染各动作的 12 视角热力图、medoid 姿态视图以及两种分割结果。

所有输出统一落到 `output/<模型名>/` 下，可直接喂给 VLM 做进一步语义分析。

---

## 目录结构

```
AgentSematic/
├── mesh_distortion_analyzer.py   # 主入口：畸变分析 + 端到端流水线 + CLI
├── heatmap_renderer.py           # 独立的热力图渲染 CLI（仅热力图）
├── spectral_cluster.py           # medoid 姿态上的谱聚类分割
├── seg_mat_runner.py             # SEG-MAT / QMAT 调用封装
├── skeleton_mat.py               # 从蒙皮骨架近似生成 initial.ma
├── config/
│   └── vlm_config.yaml           # VLM API 配置（API Key 走环境变量）
├── third_party/
│   ├── SEG-MAT/                  # submodule：分割算法源码
│   ├── QMAT/                     # 本地构建的 QMAT CLI（被 .gitignore 忽略）
│   └── ...                       # 各类下载件、压缩包（被 .gitignore 忽略）
├── environment.yml               # conda 环境定义
└── output/                       # 运行产物（被 .gitignore 忽略）
```

---

## 安装

### 1. Python 环境

推荐用 conda 一次拉齐：

```bash
conda env create -f environment.yml
conda activate agentsematic

# 谱聚类还需要 scipy（暂未写入 environment.yml）
pip install scipy
```

或直接用 pip：

```bash
pip install "numpy>=1.24" "pyyaml>=6.0" "trimesh>=4.0" "pygltflib>=1.16" \
            "ufbx>=0.0.5" "DracoPy>=2.0.0" "pyvista>=0.43" "pillow>=10.0" "scipy>=1.10"
```

### 2. 拉取 submodule

```bash
git submodule update --init --recursive
```

### 3. SEG-MAT / QMAT 可执行文件（**仅 SEG-MAT 分割需要**）

如果你不需要 SEG-MAT 分割，可以加 `--no-seg-mat` 直接跳过本节，**谱聚类与畸变分析不依赖任何外部 exe**。

需要时按下面准备两个 CLI，默认放到这些位置即可（也可通过环境变量覆盖）：

| 程序 | 默认查找路径 | 环境变量 |
| --- | --- | --- |
| `SEG-MAT.exe` | `third_party/SEG-MAT/exe program/SEG-MAT-Release/SEG-MAT.exe` | `SEG_MAT_EXE` |
| `QMAT.exe`    | `third_party/QMAT/build/Release/QMAT.exe`                       | `QMAT_EXE` |

- **SEG-MAT**：参见 `third_party/SEG-MAT/README.md`，或下载作者发布的 `SEG-MAT-Release.zip` 解压到上述路径。
- **QMAT**：从 [QMAT 仓库](https://github.com/ningnachen/QMAT) 自行 cmake 构建，或解压你本地的 `qmat_x64_sourcecode.rar` 构建。

`initial.ma` 文件按以下优先级自动定位（无需命令行指定）：

1. `--seg-mat-initial-ma` 或环境变量 `SEG_MAT_INITIAL_MA`
2. `output/<模型>/medoid_pose/{render_pose,initial}.ma`
3. 与源 mesh 同目录的 `{stem}.ma` / `{stem}_initial.ma` / `{stem}_mat.ma`
4. `config/mat/{stem}.ma` 或 `data/mat/{stem}.ma`
5. 都没命中且 mesh 含蒙皮骨架时，会自动从骨架近似导出（见 `skeleton_mat.py`）

### 4. VLM 配置（可选）

`config/vlm_config.yaml` 用 `${ENV_VAR_NAME}` 引用 API Key，按需要把对应的环境变量设置好即可，例如：

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

支持 OpenAI / Anthropic / Google / Azure OpenAI / Qwen / 智谱 / Moonshot / DeepSeek / Ollama / 自定义 endpoint。

---

## 快速开始

最常用的一条命令：

```bash
python mesh_distortion_analyzer.py path/to/model.glb
```

会自动在 `output/<模型名>/` 下生成：

```
output/<模型名>/
├── analysis.json                       # 全量畸变统计 + medoid 信息 + 分割标签摘要
├── heatmaps/                           # 每个动作的 12 视角热力图
│   ├── angular/<动作>.png
│   ├── area/<动作>.png
│   ├── weighted/<动作>.png
│   └── combined/<动作>.png
├── medoid_pose/                        # medoid 姿态的单视角概览图与中间 .ma 文件
│   ├── angular_distortion.png
│   ├── area_distortion.png
│   └── ...
├── segmentation/                       # SEG-MAT 分割结果（12 视角）
│   └── medoid_segmentation.png
└── spectral_clustering/                # 谱聚类分割结果（12 视角）
    ├── medoid_spectral_clustering.png
    ├── overview.png
    └── face_cluster_labels.json
```

---

## 常用 CLI 参数

### `mesh_distortion_analyzer.py`（端到端主入口）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `mesh` | — | **必填**，输入 mesh 路径 |
| `-o / --output-dir` | `output/<模型名>` | 输出目录 |
| `--output-base` | `output` | 输出根目录（与 `--output-dir` 二选一） |
| `--reference {weighted_average, rest}` | `weighted_average` | 畸变参考形状 |
| `--fps` | `30.0` | 动画采样帧率 |
| `--max-frames` | 无 | 每个动作最多采样帧数（加速用） |
| `--workers` | `~cpu/2` | 畸变分析并行进程数 |
| `--motion-skip RATIO` | `1e-4` | 相邻帧运动跳过阈值（0 = 逐帧全算） |
| `--medoid-samples` | `128` | medoid 帧子采样上限 |
| `--view ELEV,AZIM` | `22.6,45` | medoid 单视角 |
| `--spectral-clusters K` | 自动估计 | 手动指定谱聚类簇数 |
| `--spectral-scalar-weight W` | `0.35` | 谱聚类中畸变特征的权重 |
| `--metrics` | `angular,area,weighted,combined` | 渲染哪些热力图指标 |
| `--no-seg-mat` | — | 跳过 SEG-MAT 分割 |
| `--no-spectral-cluster` | — | 跳过谱聚类分割 |
| `--no-animation-heatmaps` | — | 跳过各动作 12 视角热力图 |
| `--no-render` | — | 跳过所有渲染（只算数据） |
| `--no-export-json` | — | 不导出 `analysis.json` |
| `--uniform-weight` | — | 动作等权（默认按时长加权） |
| `--verbose` | — | 打印各阶段耗时 |

完整参数见 `python mesh_distortion_analyzer.py -h`。

### `heatmap_renderer.py`（只渲染热力图）

```bash
python heatmap_renderer.py path/to/model.glb --metrics angular,area --max-frames 60
```

仅做参考形状计算 + 热力图渲染，不跑 SEG-MAT / 谱聚类。参数子集与上面同名一致。

---

## 典型使用场景

**只看畸变热力图，跳过所有分割：**

```bash
python mesh_distortion_analyzer.py model.glb --no-seg-mat --no-spectral-cluster
```

**只想要 SEG-MAT 分割结果，并指定 SEG-MAT 路径：**

```bash
python mesh_distortion_analyzer.py model.glb --no-spectral-cluster \
    --seg-mat-exe D:/tools/SEG-MAT.exe --qmat-exe D:/tools/QMAT.exe
```

**手动指定谱聚类簇数 = 8、降低畸变特征权重：**

```bash
python mesh_distortion_analyzer.py model.glb --no-seg-mat \
    --spectral-clusters 8 --spectral-scalar-weight 0.15
```

**只算数据不出图（适合批处理 / CI）：**

```bash
python mesh_distortion_analyzer.py model.glb --no-render
```

---

## 模块说明

| 模块 | 作用 |
| --- | --- |
| `mesh_distortion_analyzer.py` | 加载多种 mesh 格式、采样动画帧、计算畸变指标、选 medoid、串起全流水线 |
| `heatmap_renderer.py` | 基于 pyvista + Pillow 的离屏渲染：12 视角拼图、统一色阶、SEG-MAT 调色板等 |
| `spectral_cluster.py` | 归一化图 Laplacian + KMeans；特征值间隙自动估簇数；融合畸变标量特征 |
| `seg_mat_runner.py` | 写 `.off`、调用 QMAT 简化 MAT、调用 SEG-MAT、解析彩色 OFF 反推 face 标签 |
| `skeleton_mat.py` | 蒙皮骨架 → `initial.ma`：关节→球，父子→边，兄弟关节→三角面（QMAT 拓扑校验） |

---

## License & Third Party

- 本仓库源码：见仓库 `LICENSE`（如缺失请补充）。
- `third_party/SEG-MAT/`：以 git submodule 形式引入，遵循其自身 license。
- QMAT 可执行文件 / 源码、`*.zip` / `*.rar` 等下载件均**不纳入版本控制**，由 `.gitignore` 忽略。
