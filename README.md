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
├── geometry/                     # 纯几何 / 拓扑 / 聚类（无 HTTP，无 pyvista）
│   ├── mesh_distortion_analyzer.py  # 主入口：畸变分析 + 端到端流水线 + CLI
│   ├── spectral_cluster.py          # medoid 姿态上的谱聚类分割
│   ├── seg_mat_runner.py            # SEG-MAT / QMAT 调用封装
│   ├── skeleton_mat.py              # 从蒙皮骨架近似生成 initial.ma
│   ├── blocks.py                    # SemanticBlock 树结构 + 序列化
│   ├── seed_split.py                # 在分块子 mesh 上 FPS / 谱聚类分裂
│   └── grouped_split.py             # 过分割 + VLM 一次性分组
├── rendering/                    # 离屏渲染（pyvista + PIL）
│   ├── heatmap_renderer.py          # 独立的热力图渲染 CLI（仅热力图）
│   ├── render_component.py          # 单分块高亮 / 单视角透明背景渲染
│   └── view_thumbnails.py           # 选视角用的 12 张缩略图拼图
├── vlm/                          # **只**包含 VLM 请求相关代码
│   ├── api_keys.py / config.py      # .env 加载 + provider 配置
│   ├── client.py                    # OpenAI 兼容 / Anthropic / Google / Doubao
│   ├── prompts.py                   # 三分类 / 分组 / 种子数 / 进度自审 提示模板
│   ├── classifier.py                # full / partial / none 判定（HTTP 封装）
│   ├── review.py                    # 全局进度自审（HTTP 封装）
│   ├── view_selector.py             # 最佳视角选择（HTTP 封装）
│   └── cli.py                       # 兼容旧入口的薄转发 → pipeline.cli
├── pipeline/                     # 流水线 / HTML 报告 / CLI
│   ├── pipeline.py                  # 单语义迭代细化主循环（full/partial/none）
│   ├── full_coverage.py             # 全覆盖语义分割主循环（multi-label + 按命中数分裂）
│   ├── report.py                    # 单文件 HTML 报告生成
│   ├── cli.py                       # python -m pipeline.cli 入口（单语义）
│   └── full_coverage_cli.py         # python -m pipeline.full_coverage_cli 入口（全覆盖）
├── scripts/                      # 调试小脚本（ping providers / smoke test）
├── config/
│   └── vlm_config.yaml           # VLM API 配置（API Key 走环境变量）
├── .env.example                  # 复制为 .env 后填 API Key；.env 已被 .gitignore
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

`config/vlm_config.yaml` 用 `${ENV_VAR_NAME}` 引用 API Key，按需要把对应的环境变量设置好即可。
**推荐做法**：把 `.env.example` 复制为 `.env`，把真实 key 写进去——`.env` 已经在 `.gitignore`
中（同时还忽略了 `.env.*`、`.env.local` 等变体），不会被误提交：

```bash
cp .env.example .env
# 然后编辑 .env，按需填 OPENAI_API_KEY / ANTHROPIC_API_KEY / ...
```

也可以直接在 shell 里 `export` / `$env:` 设置，效果等价：

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

支持 OpenAI / Anthropic / Google / Azure OpenAI / Qwen / 智谱 / Moonshot / DeepSeek / Ollama / 自定义 endpoint。

---

## 快速开始

最常用的一条命令：

```bash
python -m geometry.mesh_distortion_analyzer path/to/model.glb
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

### `geometry.mesh_distortion_analyzer`（端到端主入口）

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

完整参数见 `python -m geometry.mesh_distortion_analyzer -h`。

### `rendering.heatmap_renderer`（只渲染热力图）

```bash
python -m rendering.heatmap_renderer path/to/model.glb --metrics angular,area --max-frames 60
```

仅做参考形状计算 + 热力图渲染，不跑 SEG-MAT / 谱聚类。参数子集与上面同名一致。

---

## 典型使用场景

**只看畸变热力图，跳过所有分割：**

```bash
python -m geometry.mesh_distortion_analyzer model.glb --no-seg-mat --no-spectral-cluster
```

**只想要 SEG-MAT 分割结果，并指定 SEG-MAT 路径：**

```bash
python -m geometry.mesh_distortion_analyzer model.glb --no-spectral-cluster \
    --seg-mat-exe D:/tools/SEG-MAT.exe --qmat-exe D:/tools/QMAT.exe
```

**手动指定谱聚类簇数 = 8、降低畸变特征权重：**

```bash
python -m geometry.mesh_distortion_analyzer model.glb --no-seg-mat \
    --spectral-clusters 8 --spectral-scalar-weight 0.15
```

**只算数据不出图（适合批处理 / CI）：**

```bash
python -m geometry.mesh_distortion_analyzer model.glb --no-render
```

---

## 模块说明

| 模块 | 作用 |
| --- | --- |
| `geometry/mesh_distortion_analyzer.py` | 加载多种 mesh 格式、采样动画帧、计算畸变指标、选 medoid、串起全流水线 |
| `rendering/heatmap_renderer.py` | 基于 pyvista + Pillow 的离屏渲染：12 视角拼图、统一色阶、SEG-MAT 调色板等 |
| `geometry/spectral_cluster.py` | 归一化图 Laplacian + KMeans；特征值间隙自动估簇数；融合畸变标量特征 |
| `geometry/seg_mat_runner.py` | 写 `.off`、调用 QMAT 简化 MAT、调用 SEG-MAT、解析彩色 OFF 反推 face 标签 |
| `geometry/skeleton_mat.py` | 蒙皮骨架 → `initial.ma`：关节→球，父子→边，兄弟关节→三角面（QMAT 拓扑校验） |
| `geometry/{blocks,seed_split,grouped_split}.py` | SemanticBlock 树 + 分块子谱聚类 + 过分割分组 |
| `rendering/{render_component,view_thumbnails}.py` | 单分块高亮 / 单视角透明背景 / 12 视角缩略图 |
| `vlm/` | VLM 请求层：客户端、提示模板、三分类 / 进度自审 / 视角选择 |
| `pipeline/` | 主流程编排 + 单文件 HTML 报告 + CLI（`python -m pipeline.cli`） |

---

## VLM 语义细化（`vlm/` 模块）

在已有 SEG-MAT 或谱聚类结果之上，使用 VLM 把每个分块按用户输入的语义判定为
**完全符合 / 局部符合 / 完全不符合** 三类，并对"局部符合"的分块继续做谱聚类
分裂，直到没有"局部符合"为止：

1. **Step 1：高亮 + 三分类**
   每个分块单独渲染一张 4×3 = 12 视角拼图：当前分块用鲜艳红色高亮，其它分块淡灰
   做参考。把这张图喂给 VLM，让其返回 JSON：
   ```json
   {"label": "full" | "partial" | "none", "confidence": 0.9, "reason": "..."}
   ```
2. **Step 2：分裂**
   - `full` → 直接接受
   - `none` → 直接拒绝
   - `partial` → 先问 VLM 在这一块内需要多少子块（`seed_count ≥ 2`），
     然后只在该分块顶点构造的子 mesh 上跑一次谱聚类，得到 `seed_count` 个子分块。
     每个子分块回到 Step 1。
3. **终止条件**：队列空 / 达到 `--max-depth` / 子块面数低于 `--min-block-size`。

### 一行命令

```bash
# 先准备好 .env 中的 API Key
python -m pipeline.cli \
    output/<模型>/medoid_pose/render_pose.off \
    --labels output/<模型>/medoid_pose/face_segment_labels.json \
    --semantic "面部" \
    --provider openai \
    --verbose

# 旧入口 python -m vlm.cli 已转为薄转发，命令行参数完全兼容
```

也可以直接用提供的测试脚本跑一个 smoke check：

```bash
python -m scripts.test_vlm_pipeline --smoke    # 不调 VLM，只验证四个包的 import & 渲染链
```

也可以传 `--labels output/<模型>/spectral_clustering/face_cluster_labels.json` 从
谱聚类结果开始。`--dry-run` 只渲染初始分块拼图、不调用 VLM，可用于排查。

### 输出

`output/<模型>/vlm/<语义>/`：

```
panels/                每个 block 的高亮 12 视角图（block_C0.png ...）
responses/             每次 VLM 调用的原始 JSON 响应（便于复盘 / 调参）
initial_segmentation.png   初始分块拼图
semantic_mask.png      最终：绿色=完全符合语义，淡色=不符合
semantic_mask.json     最终被判为"符合"的 face_id 列表
block_tree.json        完整 block 树（含父子、状态、VLM 决策）
responses_log.json     调用流水时序
```

### 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--semantic STR` | — | **必填**，用户语义（如 "面部"、"右手"） |
| `--labels PATH` | — | **必填**，face_labels JSON（SEG-MAT 或谱聚类输出均可） |
| `--provider NAME` | YAML 中 `active_provider` | 选定 VLM 提供方 |
| `--model NAME` | provider 的 `default_model` | 覆盖模型 |
| `--config PATH` | `config/vlm_config.yaml` | 配置 YAML 路径 |
| `--env-file PATH` | 自动加载 `.env` / `.env.local` | 额外 .env 路径 |
| `--max-depth N` | `3` | 局部符合分裂递归深度上限 |
| `--max-seeds N` | `6` | 单次分裂最多种子数 |
| `--min-block-size N` | `20` | 忽略 face 数少于该值的分块 |
| `--image-size N` | `512` | 单视角分辨率（12 视角共用） |
| `--scalar-weight W` | `0.35` | 子谱聚类畸变特征权重 |
| `--dry-run` | — | 只渲染初始分块，不调用 VLM |
| `--verbose` | — | 打印进度（每个 block 的分类 / 分裂） |

### API Key 安全

- `.env` / `.env.local` / `.env.*` 都在 `.gitignore` 里，**永远不会被提交**；
  只有 `.env.example` 允许进 Git。
- `config/vlm_config.yaml` 全部用 `${ENV_VAR_NAME}` 引用 key，YAML 里**没有任何明文 key**。
- 程序启动时会按顺序加载 `./.env` → `./.env.local`，已有的环境变量不会被覆盖
  （所以 CI 里 `OPENAI_API_KEY=xxx python -m pipeline.cli ...` 也能正常用）。

---

## VLM 全覆盖语义分割（`pipeline/full_coverage.py`）

与上面的 **单语义** 模式（full/partial/none）不同，**全覆盖** 模式让 VLM 自己看
整模型枚举出一份「全包围、互不重叠」的语义清单，再把模型的**每一块面**都分配到
清单中的某一个部位上，等价于一次性完成「语义分块」。

### 算法流程

1. **整模型 overview 渲染 → VLM 枚举**
   渲染一张「整模型 12 视角同色拼图」（`semantics_overview.png`），让 VLM 列出一份
   **全包围、互不重叠** 的语义部位清单。例如人形角色 → `["头部", "颈部", "躯干",
   "左上臂", "右上臂", "左前臂+左手", ..., "左小腿+左脚", "右小腿+右脚"]`。
   也可以用 `--semantic-parts` 或 `--semantic-parts-file` 直接给定，跳过 VLM 枚举。
2. **初始分块**：同单语义模式 —— SEG-MAT/谱聚类 label（或直接整 mesh）→ 按连通分量展开
   → 每个 CC 作为一个 `C{i}` 根 block。
3. **逐块 multi-label**：对每个待定块渲染高亮 12 视角，把封闭候选集喂给 VLM 让它
   选出红色实际覆盖到的语义（multi-label，0~N 项）：
   ```json
   {
     "matched_parts": ["头部", "颈部"],
     "is_background": false,
     "dominant": "头部",
     "confidences": {"头部": 0.85, "颈部": 0.4},
     "notes": "..."
   }
   ```
4. **按命中数分裂**：
   - `len(matched) == 0` / `is_background=true` → 该块为 **background**，
     标 `none` 不再处理；
   - `len(matched) == 1` → 该块直接归到该语义；
   - `len(matched) ≥ 2` → 切成 `K = len(matched)` 个子块（同样走
     CC → segmat → FPS-Voronoi-Lloyd 级联），子块下一轮再 multi-label。
5. **终止条件**：所有叶块的 `matched` 数 ≤ 1，或达到 `--max-depth`，或子块面数低于
   `--min-block-size`。`max-depth` 处仍是多命中时，把块归到 VLM 的 `dominant`
   语义（保守的"best-effort"接受）。

### 一行命令

```bash
# 让 VLM 自己枚举语义清单
python -m pipeline.full_coverage_cli \
    output/<模型>/medoid_pose/render_pose.off \
    --labels output/<模型>/medoid_pose/face_segment_labels.json \
    --enumerate-hint "请按拟人骨架来切，左右对称肢分开" \
    --provider openai \
    --verbose

# 或显式给定语义清单，跳过 VLM 枚举
python -m pipeline.full_coverage_cli \
    output/<模型>/medoid_pose/render_pose.off \
    --semantic-parts "头部,颈部,躯干,左上臂,右上臂,左前臂,右前臂,左大腿,右大腿,左小腿,右小腿" \
    --provider openai \
    --verbose
```

### 输出

`output/<模型>/vlm_full/`：

```
enumeration.json              VLM 枚举到的语义清单（含理由 / 置信度）
semantics_overview.png        整模型 12 视角 overview（枚举所用的输入图）
panels/                       每个 block 的高亮 12 视角图（block_C0.png ...）
responses/                    每次 VLM multi-label 调用的原始 JSON
initial_segmentation.png      初始分块拼图（按 CC 着色）
full_coverage_mask.png        最终：每张面按所属语义上色（自带配色图例）
face_semantic_labels.json     每张面最终的 semantic 索引 / 部位计数
leaf_blocks_overview.png      叶分块总览（每块独立随机色，便于核对切分粒度）
block_tree.json               完整 block 树（含 multi-label 决策 / 分裂记录）
live_report.html              实时 HTML 进度页（自带 meta refresh）
iterations/iter_NN.png        单视角迭代时间线
best_view.json                VLM 选定的视角 + 理由
best_view_overview.png        chosen view 的整模型透明背景 PNG
```

### 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `mesh` | — | **必填**，渲染姿态 mesh（推荐 `render_pose.off`） |
| `--labels PATH` | 整 mesh 连通分量 | 初始分块来源 JSON |
| `--enumerate-hint STR` | `""` | 给 VLM 枚举阶段的额外提示（如 "按拟人骨架切"） |
| `--semantic-parts STR` | — | 显式给定语义清单（逗号 / 中文「、」分隔），跳过 VLM 枚举 |
| `--semantic-parts-file PATH` | — | 从 JSON 读语义清单，跳过 VLM 枚举 |
| `--enumerate-min-parts N` | `2` | VLM 枚举的最少部位数 |
| `--enumerate-max-parts N` | `12` | VLM 枚举的最多部位数 |
| `--enumerate-model NAME` | 默认模型 | 枚举阶段单独指定模型 |
| `--max-depth N` | `6` | 分裂递归深度上限 |
| `--min-block-size N` | `20` | 忽略 face 数少于该值的分块 |
| `--max-seeds-per-split N` | `6` | 单次分裂的种子数上限（默认 `K=len(matched)`） |
| `--parallel-vlm N` | `1` | 每轮 N 个块的 multi-label 用线程池并发 |
| `--no-live-report` | — | 关闭 live_report.html |
| `--no-view-select` | — | 关掉 VLM 选视角，直接用 DEFAULT_OVERVIEW_VIEW |
| `--dry-run` | — | 只渲染 overview + 初始分块，不调用 VLM |
| `--verbose` | — | 打印每个 block 的命中 / 分裂记录 |

完整参数见 `python -m pipeline.full_coverage_cli -h`。

### 与单语义模式的对比

|  | 单语义 (`pipeline.cli`) | 全覆盖 (`pipeline.full_coverage_cli`) |
| --- | --- | --- |
| 用户输入 | 一条语义文本（如「面部」） | 无（VLM 自己枚举）或一份语义清单 |
| VLM 判定 | full / partial / none | 命中候选清单中的哪些部位（multi-label） |
| 分裂触发 | label=partial | matched ≥ 2 |
| 分裂种子数 | VLM 单独询问 `seed_count` | `K = len(matched_parts)` |
| 最终掩码 | 一种颜色 = 符合语义 | 每个语义一种颜色 + background |
| 用途 | 提取**一个**特定语义区域 | 一次性完成**整模型语义分块** |

---

## License & Third Party

- 本仓库源码：见仓库 `LICENSE`（如缺失请补充）。
- `third_party/SEG-MAT/`：以 git submodule 形式引入，遵循其自身 license。
- QMAT 可执行文件 / 源码、`*.zip` / `*.rar` 等下载件均**不纳入版本控制**，由 `.gitignore` 忽略。
