"""
geometry/ —— 纯几何 / 拓扑 / 聚类逻辑（无 HTTP，无 pyvista）。

子模块：
- mesh_distortion_analyzer : Mesh / 动画加载、畸变计算
- seg_mat_runner           : SEG-MAT 外部命令调用 + 后处理
- skeleton_mat             : MAT 骨架后处理
- spectral_cluster         : 谱聚类相关
- blocks                   : SemanticBlock 树结构 + 序列化
- seed_split               : VLM 种子数 / 策略询问 + 分块入口 re-export
- partition                : 各分块策略（connected_components / geodesic_fps / spectral / graph_cut / cvt / sdf / skeleton）
- partition_methods        : 兼容层，转发至 geometry.partition
- topology_repair          : 拓扑几何手术（谱切割 / handle loop / 凹缝 / 图割精修）
- grouped_split            : 过分割 + VLM 一次性分组（VLMClient 由调用方注入）
"""
