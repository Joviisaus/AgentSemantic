# AI-VLM 亲手画 mask + 单视角反投影 vs 官方基线（CoMPaT-200 tiny 15）

口径：per-part IoU 求平均、union==0 跳过（CoSMo3D miou_skip）；instance = 15 样本均值。所有方法用**同一份 GT 与同一 15 个 key**。

## 每样本 mIoU(skip)

| key | 类别 | **AI-VLM(我)** | ours_pointcloud | cosmo3d | ours(mesh) |
|---|---|---|---|---|---|
| airplane__coarse_b00_004 | airplane | **0.294** ✅ | 0.148 | 0.309 | 0.250 |
| belt__coarse_b2f_025 | belt | **0.713** ✅ | 0.365 | 0.606 | 0.371 |
| bed__coarse_b04_017 | bed | **0.539** ✅ | 0.351 | 0.401 | 0.023 |
| balance_scale__coarse_b2d_026 | balance_scale | **0.434** | 0.600 | 0.376 | 0.169 |
| bench__coarse_b05_030 | bench | **0.361** ✅ | 0.235 | 0.644 | 0.199 |
| arcade_machine__coarse_b2b_026 | arcade_machine | **0.397** ✅ | 0.130 | 0.276 | 0.181 |
| axe__coarse_b2c_038 | axe | **0.490** | 0.531 | 0.683 | 0.301 |
| bath_tub__coarse_b2e_029 | bath_tub | **0.638** | 0.646 | 0.808 | 0.488 |
| bbq_grill__coarse_b03_00c | bbq_grill | **0.320** ✅ | 0.235 | 0.204 | 0.523 |
| bag__coarse_b01_002 | bag | **0.581** | 0.646 | 0.877 | 0.349 |
| air_conditioning_unit__coarse_b2a_01d | air_conditioning_unit | **0.462** ✅ | 0.406 | 0.649 | 0.448 |
| bird_house__coarse_b07_003 | bird_house | **0.761** ✅ | 0.511 | 0.449 | 0.630 |
| bicycle__coarse_b06_007 | bicycle | **0.625** ✅ | 0.000 | 0.473 | 0.230 |
| basket__coarse_b02_000 | basket | **0.373** ✅ | 0.000 | 0.783 | 0.422 |
| binoculars__coarse_b30_024 | binoculars | **0.426** ✅ | 0.000 | 0.360 | 0.027 |

## 汇总

| 指标 | **AI-VLM(我)** | ours_pointcloud | cosmo3d | ours(mesh) |
|---|---|---|---|---|
| instance mIoU(skip) | **0.4942** | 0.3202 | 0.5266 | 0.3074 |
| instance accuracy | **0.8002** | 0.4681 | 0.7492 | 0.5558 |

**15 个样本里，我(单视角手画)在 11/15 个上超过 ours_pointcloud。**

## 实验设置

- **视角**：每个点云在 9 个候选视角里选「可见点最多」的单一重要视角（几何选择，不看 GT）。
- **画 mask**：我作为视觉模型逐 part 在灰度渲染图上画像素级红区（多边形/椭圆/矩形），背景主 part 用可见轮廓填充，小 part 后画覆盖。**不使用项目 API**。
- **相机**：纯 numpy 复刻 heatmap_renderer 的针孔相机（FOV=30°、up=z、球坐标机位），保证画在图上的 mask 能一致地反投影回 3D。
- **反投影**：5000 点正投影→带 z-buffer 遮挡判定；可见点取其像素所属 mask，被遮挡/背面点用 3D 最近已标点传播（这就是单视角盲区的诚实补齐）。
- **只用单视角、零再分块请求。**
