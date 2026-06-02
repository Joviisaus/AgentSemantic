"""
rendering/ —— 离屏渲染（pyvista + PIL），输出图片给 VLM 看或拼报告。

子模块：
- heatmap_renderer   : 12 视角拼图 / SEG-MAT 调色 / 单视角渲染
- render_component   : 单分块高亮 / 单视角透明背景 / 过分割候选面板
- view_thumbnails    : 给 VLM 选视角用的 12 张缩略图拼图
"""
