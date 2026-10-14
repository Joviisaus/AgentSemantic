# VLM 视觉评审结果

由智谱 GLM-4V-Plus 对每张 4 联画（GT / ours_pointcloud / pointclip / cosmo3d）独立打分（0–10）。

评审方式：先让 VLM 客观描述每个面板的颜色分布，再基于描述打分。


## compat200

| sample | gt_ours | gt_pointclip | gt_cosmo3d | leading | comment |
|---|---:|---:|---:|:---:|---|
| `bbq_grill__coarse_b03_00c` | 8 | - | 5 | **yes** | ours_pointcloud 的彩色分布与 GT 形态更为吻合 |
| `air_conditioning_unit__coarse_b2a_01d` | 7 | - | 5 | **yes** | ours_pointcloud 的颜色分布和形状与 GT 更为接近 |
| `axe__coarse_b2c_038` | 8 | - | 6 | **yes** | ours_pointcloud 的彩色分布与 GT 更为相似 |
| `bath_tub__coarse_b2e_029` | 8 | - | 6 | **yes** | ours_pointcloud 与 GT 的颜色和形状相似度更高 |
| `bed__coarse_b04_017` | 8 | - | 6 | **yes** | ours_pointcloud 的颜色分布与 GT 更为相似，形态吻合度更高。 |
| `belt__coarse_b2f_025` | 8 | - | 6 | **yes** | ours_pointcloud 的彩色分布与 GT 更为接近 |
| `bench__coarse_b05_030` | 8 | - | 6 | **yes** | ours_pointcloud 与 GT 的颜色和形状更相似 |
| `airplane__coarse_b00_004` | 8 | - | 7 | **yes** | ours_pointcloud 与 GT 的形状和颜色分布最为相似 |
| `arcade_machine__coarse_b2b_026` | 8 | - | 7 | **yes** | ours_pointcloud与GT的形状和颜色分布最为相似 |
| `bag__coarse_b01_002` | 7 | - | 6 | **yes** | ours_pointcloud 的彩色分布与 GT 形态更为吻合 |
| `balance_scale__coarse_b2d_026` | 8 | - | 7 | **yes** | ours_pointcloud 的颜色分布与 GT 更为相似 |
| `bird_house__coarse_b07_003` | 7 | - | 8 | **no** | ours_pointcloud 的颜色分布和形状与 GT 相似度较低 |
| `binoculars__coarse_b30_024` | 5 | - | 7 | **no** | ours_pointcloud 面板颜色较少且分布不均 |
| `basket__coarse_b02_000` | 2 | - | 8 | **no** | ours_pointcloud 面板几乎全灰，cosmo3d 面板与 GT 形态吻合度较高 |
| `bicycle__coarse_b06_007` | 2 | - | 8 | **no** | ours_pointcloud 面板几乎全灰，cosmo3d 面板与 GT 形态吻合 |

## shapenetpart

| sample | gt_ours | gt_pointclip | gt_cosmo3d | leading | comment |
|---|---:|---:|---:|:---:|---|
| `earphone__test0_00166` | 8 | - | 5 | **yes** | ours_pointcloud与GT的颜色和形状更相似 |
| `cap__test0_00186` | 7 | - | 5 | **yes** | ours_pointcloud 的颜色和形状与 GT 更为接近 |
| `knife__test0_00013` | 8 | - | 6 | **yes** | ours_pointcloud 的彩色分布与 GT 形态更吻合 |
| `mug__test0_00033` | 8 | - | 6 | **yes** | ours_pointcloud 的形状与 GT 更相似 |
| `airplane__test0_00021` | 8 | - | 7 | **yes** | ours_pointcloud 与 GT 的形状和颜色分布最为相似 |
| `bag__test0_00142` | 7 | - | 6 | **yes** | ours_pointcloud 的颜色和形状与 GT 更为接近 |
| `car__test0_00000` | 8 | - | 7 | **yes** | ours_pointcloud 的分布与 GT 更为相似 |
| `chair__test0_00008` | 8 | - | 7 | **yes** | ours_pointcloud 与 GT 的颜色和形状相似度更高。 |
| `guitar__test0_00006` | 7 | - | 6 | **yes** | ours_pointcloud 的颜色分布与 GT 更为相似 |
| `lamp__test0_00003` | 8 | - | 7 | **yes** | ours_pointcloud 与 GT 的形状和颜色相似度更高 |
| `laptop__test0_00071` | 7 | - | 6 | **yes** | ours_pointcloud 的颜色分布与 GT 更为相似 |
| `motorbike__test0_00015` | 8 | - | 7 | **yes** | ours_pointcloud 的颜色分布与 GT 形态更为吻合 |
| `rocket__test0_00553` | 7 | - | 6 | **yes** | ours_pointcloud 的形状与 GT 更为相似，但颜色分布不如 GT 明显 |
| `skateboard__test0_00180` | 8 | - | 7 | **yes** | ours_pointcloud 的形状和颜色分布与 GT 更为接近 |
| `pistol__test0_00076` | 5 | - | 7 | **no** | cosmo3d 的彩色分布与 GT 形态更吻合 |