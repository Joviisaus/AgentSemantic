"""
``python -m vlm.cli`` 的薄转发：CLI 已经迁到 ``pipeline.cli``，本文件只是为了
兼容以前的调用方式（旧脚本、README 命令行）保持不变。

新代码请直接用：

    python -m pipeline.cli <mesh> --semantic "<text>" ...
"""

from __future__ import annotations

from pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
