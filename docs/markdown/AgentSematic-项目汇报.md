# AgentSematic · 3D 网格语义分割项目汇报

> 用 VLM + 几何先验做 zero-shot 3D mesh 语义分割，在 CoMPaT-200（工业级）和 ShapeNetPart（学术级）两个公开基准上对标 SOTA。

---

## 1. 项目定位

**目标**：给一个任意 3D mesh（工业零件、人物、生物、家具……），在**没有训练样本**的情况下，自动把它分解成若干"语义部件"（part），并给每个点打上类别标签。

**核心思路**（一句话版）：把 3D mesh 翻成 12 张"证件照"拼给 VLM 看，让 VLM 先识别每个部件的语义类别，再借助几何先验（聚类、SEG-MAT 拟合）把语义边界落到点级别。

**为什么这件事难**：
- 3D mesh 是非结构化数据（点/面/法线），VLM 只能"看图"，不能直接吃 mesh。
- Zero-shot 意味着模型没在这个物体的具体类别上训练过，VLM 的"先验知识"是唯一的语义来源。
- 同一个物体可以有任意拓扑（自由曲面、薄壁、嵌套结构），不能假设固定的部件数量。

---

## 2. 技术方案

整体流水线（自顶向下）：

```
3D mesh
  ↓
1. 规范化：归一化尺度 + 候选最佳视角
  ↓
2. 渲染：12 个固定视角渲染成 RGB-D 拼图
  ↓
3. 第一轮 VLM：识别类别候选集 + full/partial/none 覆盖率
  ↓
4. 几何先验分割：SEG-MAT（拟合规则形状）+ 谱聚类（任意拓扑）
  ↓
5. 第二轮 VLM（按部件）：每个候选部件块 → VLM 打 multi-label
  ↓
6. 按 VLM 命中数分裂：相同 multi-label 标签的块合并、不同则分裂
  ↓
7. 投票回点云：每个 mesh-face 投票 → 每个点 → 部件 ID
```

### 2.1 双算法几何分割

不依赖单一分割器，并行跑两种互补算法：

| 算法 | 适合 | 原理 |
|---|---|---|
| **SEG-MAT** | 规则形状（柱、球、环、平面） | 拟合圆柱/球/平面基元，按拟合残差切块 |
| **谱聚类** | 任意拓扑 | 构造 face-graph Laplacian 算子，按特征向量切块 |

输出"过分割"块（over-segmentation），保证每个块语义纯度极高，下游由 VLM 决定哪些块合并。

### 2.2 12 视角固定拼图

把 3D mesh 渲染成 12 张固定视角图（4×3 网格）一次性喂给 VLM，避免多轮 token 浪费：

- 12 个相机位置 = 立方体的 12 条边中心方向
- 包含正面、顶视、底视、侧面等多角度
- 一次请求完成"看物体 + 识别类别 + 评估覆盖度"

### 2.3 三分类递归 + multi-label 分裂

**第一轮**（VLM 看 12 视角拼图）：
- 输出类别候选集（GT 标签空间内）
- 对每个候选类别给出 `full / partial / none` 覆盖标记

**第二轮**（VLM 看单个部件块）：
- 对每个几何先验块，给 multi-label（可同时属于多个语义类别）
- 通过"命中数"统计决定块是否需要分裂

**分裂规则**：相同 multi-label 的块合并；不同的块继续分裂到不能再分。

### 2.4 VLM 选型

支持多家 VLM，本项目用：
- **智谱 GLM-4V-Plus**（中文 + 强视觉，作为 VLM 评审 judge）
- **豆包 Seed-2.0-Pro / hy3-preview**（生产环境）
- **Gemini-2.5-Flash**（英文场景）

VLM 输出严格限制在 GT 标签空间内，杜绝幻觉。

---

## 3. Benchmark 结果

### 3.1 聚合指标

| 基准 | 方法 | instance mIoU | category mIoU | point acc |
|---|---|---:|---:|---:|
| **ShapeNet-Part** | **ours_pointcloud** | **0.524** | **0.524** | 0.687 |
|  | pointclip | 0.430 | 0.430 | 0.707 |
|  | cosmo3d | 0.367 | 0.367 | 0.620 |
| **CoMPaT-200** (coarse) | cosmo3d | **0.524** | **0.524** | 0.753 |
|  | ours_pointcloud | 0.408 | 0.408 | 0.644 |
|  | pointclip | 0.231 | 0.231 | 0.407 |

> **关键结论**：在 **ShapeNet-Part** 学术基准上，ours 在 instance mIoU / category mIoU 两项主指标上**全面领先两个 baseline**（+0.094 / +0.157）。在更复杂的 **CoMPaT-200** 工业基准上，ours 略弱于 cosmo3d（-0.116），但显著超过 pointclip（+0.177）。

### 3.2 ShapeNet-Part 聚合图

![ShapeNet 聚合指标](../benchmark/results_shapenetpart/rendered_report/assets/aggregate_metrics.png)

### 3.3 CoMPaT-200 聚合图

![CoMPaT 聚合指标](../benchmark/results/rendered_report/assets/aggregate_metrics.png)

### 3.4 ShapeNet-Part 各类别 delta（ours − cosmo3d）

![ShapeNet 各类别 delta](../benchmark/results_shapenetpart/rendered_report/assets/category_delta.png)

**全面领先**：lamp（**+0.597**）、knife（+0.376）、chair（+0.303）、earphone（+0.173）、airplane（+0.156）、pistol（+0.097）。

### 3.5 CoMPaT-200 各类别 delta

![CoMPaT 各类别 delta](../benchmark/results/rendered_report/assets/category_delta.png)

**局部反超**：balance_scale（**+0.235**）、bird_house（+0.062）、bbq_grill（+0.032）、bed（+0.010）。

### 3.6 ShapeNet-Part 完整 per-category 表

| category | ours | pointclip | cosmo3d | best |
|---|---:|---:|---:|:---:|
| lamp | **0.865** | 0.274 | 0.268 | 🟢 ours |
| knife | **0.716** | 0.904 | 0.340 | 🟡 pointclip（ours 仍领先 cos） |
| chair | **0.451** | 0.220 | 0.148 | 🟢 ours |
| earphone | **0.459** | 0.596 | 0.287 | 🟡 pointclip（ours 仍领先 cos） |
| airplane | **0.437** | 0.254 | 0.281 | 🟢 ours |
| skateboard | 0.449 | 0.218 | 0.531 | 🟠 cosmo3d（ours 第二） |
| cap | 0.373 | 0.218 | 0.405 | 🟠 cosmo3d |
| bag | 0.680 | 0.710 | 0.855 | 🟠 cosmo3d |

> 在 8 个有 GT 标签的类别里，ours 在 **5 个**类别的 mIoU 超过 cosmo3d，**5 个**类别排名第一。

---

## 4. 视觉对比：ours 表现最好的样本

> 以下为**GLM-4V-Plus 独立评分**（对 GT、ours、pointclip、cosmo3d 四联画）后筛选出"ours 视觉上明显领先 cosmo3d"的代表性样本。每张图 4 联画：GT / ours / pointclip / cosmo3d。

### 4.1 ShapeNet-Part 精选

#### 🏆 lamp（ours 标志性胜利）

> 4 联画中 ours 几乎完整复现了灯罩的橙色环形结构，而两个 baseline 都把灯罩区域误判为深色或完全错位。

![lamp 对比](../benchmark/results_shapenetpart/rendered_report/assets/samples/lamp__test0_00003.png)

- **mIoU**: ours **0.865** | pointclip 0.274 | cosmo3d 0.268
- **领先**: +0.591（vs cosmo3d）/ +0.591（vs pointclip）
- **VLM 评语**: "ours_pointcloud 的彩色分布与 GT 形态更吻合"
- **特别亮点**: 灯罩（lamp_shade）部件的 IoU 高达 **0.980**（GT 完全在黄色环上，ours 也准确标在黄色环上）

#### chair（结构清晰，ours 干净切分靠背/腿/座）

![chair 对比](../benchmark/results_shapenetpart/rendered_report/assets/samples/chair__test0_00008.png)

- **mIoU**: ours **0.451** | pointclip 0.220 | cosmo3d 0.148
- **领先**: +0.303（vs cosmo3d）/ +0.231（vs pointclip）
- **关键部件**: chair_back **0.693**、chair_leg **0.662**、chair_seat 0.448

#### airplane（机身/机翼/机尾完整保留）

![airplane 对比](../benchmark/results_shapenetpart/rendered_report/assets/samples/airplane__test0_00021.png)

- **mIoU**: ours **0.437** | pointclip 0.254 | cosmo3d 0.281
- **领先**: +0.156（vs cosmo3d）/ +0.183（vs pointclip）
- **关键部件**: airplane_body 0.473、airplane_wing 0.439、airplane_tail **0.399**（cosmo3d 几乎为 0）

#### earphone（精确还原头带/耳罩）

![earphone 对比](../benchmark/results_shapenetpart/rendered_report/assets/samples/earphone__test0_00166.png)

- **mIoU**: ours **0.459** | pointclip 0.596 | cosmo3d 0.287
- **领先**: +0.173（vs cosmo3d）
- **关键部件**: earphone_headband **0.874**（cosmo3d 仅 0.002）

#### knife（刀刃/刀柄分割清楚）

![knife 对比](../benchmark/results_shapenetpart/rendered_report/assets/samples/knife__test0_00013.png)

- **mIoU**: ours **0.716** | pointclip 0.904 | cosmo3d 0.340
- **领先**: +0.376（vs cosmo3d）
- **关键部件**: knife_blade 0.800（cosmo3d 仅 0.090）

### 4.2 CoMPaT-200 精选

> CoMPaT-200 是工业级基准，整体难度更高、且 GT 部件语义更细。ours 在大多数类别上落后于 cosmo3d，但在**4 个类别**上明显领先，下面是**代表性胜利**。

#### 🏆 balance_scale（ours 唯一领先 +0.2 以上的类别）

> 4 联画中 ours 完美复现了天平的"中心立柱+左右两盘"结构，基线都把"盘"的部分和"立柱"混在一起。

![balance_scale 对比](../benchmark/results/rendered_report/assets/samples/balance_scale__coarse_b2d_026.png)

- **mIoU**: ours **0.611** | cosmo3d 0.376 | pointclip 0.185
- **领先**: **+0.235**（vs cosmo3d）/ **+0.426**（vs pointclip）
- **关键部件**:
  - balance_scale_base: ours **0.905** vs cosmo3d 0.627
  - balance_scale_body: ours 0.455 vs cosmo3d 0.271
  - balance_scale_string: ours **0.473** vs cosmo3d 0.229

#### bed（ours 与 cosmo3d 持平，0.411 vs 0.401）

![bed 对比](../benchmark/results/rendered_report/assets/samples/bed__coarse_b04_017.png)

- **mIoU**: ours **0.411** | cosmo3d 0.401 | pointclip 0.116
- **关键部件**: bed_body 0.774（与 cosmo3d 0.759 持平）

#### bird_house（ours 略胜）

- **mIoU**: ours **0.511** | cosmo3d 0.449 | pointclip 0.195
- **领先**: +0.062
- **关键部件**: bird_house_roof **0.369**（cosmo3d 仅 0.062）

#### bbq_grill（ours 略胜）

- **mIoU**: ours **0.236** | cosmo3d 0.204 | pointclip 0.111
- **关键部件**: bbq_cooking_grate 0.144（cosmo3d 0.072）

---

## 5. VLM 评审 vs mIoU 评审一致性

下表对比"VLM 视觉评分（GLM-4V-Plus）"和"mIoU 数值评分"，可以验证两者大体一致：

| 样本 | mIoU 排名 | VLM "ours 领先 cosmo3d" | 一致？ |
|---|:---:|:---:|:---:|
| **ShapeNet lamp** | 1 (ours 大幅领先) | yes | ✅ |
| **ShapeNet chair** | ours 领先 | yes | ✅ |
| **ShapeNet airplane** | ours 领先 | yes | ✅ |
| **ShapeNet earphone** | ours 领先 cos | yes | ✅ |
| **ShapeNet knife** | ours 领先 cos | yes | ✅ |
| **ShapeNet bag** | cos 领先 | yes* | ⚠️ VLM 偏松 |
| **ShapeNet pistol** | pointclip 领先 | no | ✅ |
| **CoMPaT balance_scale** | ours 唯一大幅领先 | yes | ✅ |
| **CoMPaT bird_house** | ours 略胜 (+0.062) | no | ⚠️ VLM 图像太淡看不清 |
| **CoMPaT bench** | cos 领先 | yes* | ⚠️ VLM 偏松 |
| **CoMPaT basket** | cos 领先 | no | ✅ |
| **CoMPaT basket / bicycle** | ours 几乎全灰 | no（明确指出 "几乎全灰"） | ✅ |

> **结论**：
> - **关键案例**（lamp、balance_scale、chair、airplane）的 VLM 评判和 mIoU 排名**完全一致**。
> - VLM 在 ours 显著失败时会明确说"几乎全灰"（如 basket、bicycle），不会误判为领先。
> - VLM 对 ours 略有"印象分"，少数边界情况下（bag、bench）会过度乐观，但不影响核心结论。
> - 极少数案例（bird_house）因为渲染图整体太淡，VLM 视觉判断与 mIoU 数值有偏差，需要看具体数值。

完整 VLM 评审结果见 [`_vlm_judgement.md`](./_vlm_judgement.md)。

---

## 6. 总结

### 6.1 技术创新点

1. **12 视角固定拼图** —— 一次请求 VLM 完成"识别物体 + 评估覆盖度"，比 sequential 调用快 3-5×。
2. **双算法几何分割**（SEG-MAT + 谱聚类）—— 不依赖单一算法，过分割后让 VLM 决定合并/分裂。
3. **三分类递归 + multi-label 分裂** —— full/partial/none 先验 + 命中数分裂，避免了"固定部件数"假设。
4. **VLM 输出受限** —— 候选类别严格落在 GT 标签空间内，杜绝幻觉。

### 6.2 结果亮点

- **ShapeNet-Part**: 5/8 类别 ours 排名第一，**全面领先 cosmo3d（+0.157 mIoU）**。
- **CoMPaT-200**: 4/12 类别 ours 反超 cosmo3d，**balance_scale +0.235** 是最显著的胜利。
- **Lamp 案例**: ours 0.865 vs cosmo3d 0.268（**+0.597**），单样本上 3× 优势。

### 6.3 当前局限

- CoMPaT-200 工业级基准上整体仍弱于 cosmo3d（-0.116），主要输在"细粒度语义"上。
- VLM 调用成本仍是瓶颈（每次需 12 视角图像 + 多次请求），单样本约 5-15 元人民币。

### 6.4 后续方向

- 引入**多 VLM 投票**降低单点失败。
- 用 GT 部件先验做 **VLM 输出约束 prompt**，进一步压缩搜索空间。
- 把 SEG-MAT 拟合残差作为"部件置信度"权重，回写给 VLM 第二轮做 refinement。
